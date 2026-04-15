"""
Time Reversal Test for SpO2 Prediction Pipeline.
Tests whether the model relies on genuine physiological signal or temporal direction bias.
- Original: normal order prediction
- Reversed: reverse each segment, re-select TOP_K models
- Frame-Index Baseline: predict SpO2 using only normalized time position (0→1)
"""
import numpy as np
import pandas as pd
import os
import math
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

from common import (
    all_channel_names, rgb_win_last_col_names, derived_acdc_col_names,
    SEGMENT_LENGTH, TOP_K, USE_NORMALIZATION, GLOBAL_MODEL_BLEND,
    MIN_SpO2_RANGE,
    build_features_from_df, filter_feat_df_by_spo2_range,
    calculate_similarity, calculate_rgb_similarity,
    load_features_from_csv_paths,
)

OUTPUT_DIR = "./output"

DATA_CSV_PATHS = [
    "./data_new/prc-c920.csv",
    "./data_new/prc-i15.csv",
    "./data_new/prc-i15m.csv",
    "./data_new/prc2-c930.csv",
    "./data_new/prc2-i16.csv",
    "./data_new/prc2-i16m.csv",
]


def predict_with_selected_models(top_k_models, X_test_segment_df, feature_cols,
                                  folder_df, start_idx, end_idx,
                                  global_model, use_normalization=USE_NORMALIZATION):
    """Given selected top_k models, predict SpO2 with consensus filtering and GBR blending."""
    if use_normalization:
        test_means = X_test_segment_df.mean(axis=0)
        test_stds = X_test_segment_df.std(axis=0)
        X_test_input = (X_test_segment_df - test_means) / test_stds
    else:
        X_test_input = X_test_segment_df

    segment_preds = []
    sim_weights = []
    for sim_info in top_k_models:
        m = sim_info['model_info']
        y_p = np.dot(X_test_input[m['feature_cols']].values, m['weights']) + m['bias']
        if np.any(np.isnan(y_p)) or np.any(np.isinf(y_p)):
            continue
        segment_preds.append(y_p)
        w = max(sim_info['combined'], 0.01)
        sim_weights.append(w)

    if len(segment_preds) == 0:
        return None

    preds_arr = np.array(segment_preds)
    weights_arr = np.array(sim_weights)

    # Consensus filtering: remove models deviating > median*1.5
    med_pred = np.median(preds_arr, axis=0)
    deviations = np.mean(np.abs(preds_arr - med_pred), axis=1)
    dev_threshold = np.median(deviations) * 1.5
    keep_mask = deviations <= dev_threshold
    if np.sum(keep_mask) >= 3:
        preds_arr = preds_arr[keep_mask]
        weights_arr = weights_arr[keep_mask]

    weights_arr = weights_arr / weights_arr.sum()
    avg_pred = np.average(preds_arr, axis=0, weights=weights_arr)

    # Blend with global LOSO model
    if global_model is not None and GLOBAL_MODEL_BLEND > 0:
        global_feature_cols = feature_cols + [c for c in rgb_win_last_col_names if c not in feature_cols] + derived_acdc_col_names
        rgb_test_seg = folder_df[rgb_win_last_col_names].iloc[start_idx:end_idx].astype(float)
        derived_test_seg = folder_df[derived_acdc_col_names].iloc[start_idx:end_idx].astype(float)
        X_global_test = pd.concat([X_test_segment_df[feature_cols], rgb_test_seg, derived_test_seg], axis=1)
        global_pred = global_model.predict(X_global_test.values)
        avg_pred = (1 - GLOBAL_MODEL_BLEND) * avg_pred + GLOBAL_MODEL_BLEND * global_pred

    avg_pred = np.clip(avg_pred, 70, 100)
    return avg_pred


def select_top_k_models(model_pool, X_test_segment_array, rgb_mean_test_segment,
                         test_subject_group, top_k=TOP_K):
    """Select TOP_K most similar models from pool, excluding same subject_group."""
    similarities = []
    for model_info in model_pool:
        model_group = model_info.get('subject_group', model_info['subject_id'])
        if model_group == test_subject_group:
            continue
        shape_score, range_score = calculate_similarity(
            model_info['X_raw_segment'], X_test_segment_array, model_info['weights'])
        rgb_score = calculate_rgb_similarity(
            model_info['rgb_mean_segment'], rgb_mean_test_segment)
        similarities.append({
            'model_info': model_info,
            'shape_score': shape_score,
            'range_score': range_score,
            'rgb_score': rgb_score,
            'model_id': model_info['model_id'],
        })
    if len(similarities) == 0:
        return None
    for s in similarities:
        s['combined'] = 0.3 * s['shape_score'] + 0.7 * s['range_score']
    similarities.sort(key=lambda x: x['combined'], reverse=True)
    return similarities[:min(top_k, len(similarities))]


def finalize(y_pred):
    """Post-processing: rolling median + EWM smoothing."""
    smooth_window = 1801
    if len(y_pred) >= smooth_window:
        y_pred = pd.Series(y_pred).rolling(window=smooth_window, center=True, min_periods=1).median().values
    ema_span = 901
    if len(y_pred) >= ema_span:
        y_pred = pd.Series(y_pred).ewm(span=ema_span, adjust=False).mean().values
    return y_pred


def evaluate_original_and_reversed(model_pool, feat_df, selected_channels=None,
                                    top_k=TOP_K, segment_length=SEGMENT_LENGTH):
    """
    For each LOSO test subject, evaluate:
    1. Original order prediction
    2. Reversed order prediction (re-select TOP_K for reversed data)
    """
    if selected_channels is None:
        used_channels = all_channel_names
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
    feature_cols = [f"{ch}_acdc" for ch in used_channels]
    req_cols = ["subject_id", "subject_group", "folder_name", "SpO2_win_last"] + \
               feature_cols + rgb_win_last_col_names + derived_acdc_col_names
    feat_df_sel = feat_df[req_cols].copy()

    results = []

    for test_subject_id, test_subdf in feat_df_sel.groupby("subject_id", sort=False):
        test_subject_group = test_subdf["subject_group"].iloc[0]
        print(f"\nTest Subject {test_subject_id} (group {test_subject_group})")

        # Train global GBR model (LOSO)
        global_model = None
        if GLOBAL_MODEL_BLEND > 0:
            train_df = feat_df_sel[feat_df_sel["subject_group"] != test_subject_group]
            global_feature_cols = feature_cols + \
                [c for c in rgb_win_last_col_names if c not in feature_cols] + derived_acdc_col_names
            X_gt = train_df[global_feature_cols].values.astype(float)
            y_gt = train_df["SpO2_win_last"].values.astype(float)
            if len(X_gt) > 10:
                global_model = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=3, learning_rate=0.03,
                    min_samples_leaf=100, l2_regularization=2.0, random_state=42)
                global_model.fit(X_gt, y_gt)

        orig_preds, orig_labels = [], []
        rev_preds, rev_labels = [], []

        for folder_name, folder_df in test_subdf.groupby("folder_name", sort=False):
            n_samples = len(folder_df)
            if n_samples < segment_length:
                continue
            n_seg = n_samples // segment_length

            X_full = folder_df[feature_cols].astype(float)
            rgb_full = folder_df[rgb_win_last_col_names].astype(float)
            y_full = folder_df["SpO2_win_last"].values.astype(float)

            for seg_idx in range(n_seg):
                si = seg_idx * segment_length
                ei = si + segment_length

                X_seg_df = X_full.iloc[si:ei].copy()
                y_seg = y_full[si:ei]
                test_stds = X_seg_df.std(axis=0)
                if (test_stds < 1e-8).any():
                    continue

                X_seg_arr = X_seg_df.values
                rgb_mean_seg = rgb_full.iloc[si:ei].mean(axis=0).values

                # --- Original ---
                top_k_orig = select_top_k_models(
                    model_pool, X_seg_arr, rgb_mean_seg, test_subject_group, top_k)
                if top_k_orig is not None:
                    pred_orig = predict_with_selected_models(
                        top_k_orig, X_seg_df, feature_cols, folder_df, si, ei, global_model)
                    if pred_orig is not None:
                        orig_preds.append(pred_orig)
                        orig_labels.append(y_seg)

                # --- Reversed ---
                X_rev_df = X_seg_df.iloc[::-1].reset_index(drop=True)
                y_rev = y_seg[::-1]
                X_rev_arr = X_rev_df.values
                rgb_rev = rgb_full.iloc[si:ei].iloc[::-1].reset_index(drop=True)
                rgb_mean_rev = rgb_rev.mean(axis=0).values

                # Build a temporary reversed folder_df slice for GBR features
                rev_folder_slice = folder_df.iloc[si:ei].iloc[::-1].reset_index(drop=True)

                top_k_rev = select_top_k_models(
                    model_pool, X_rev_arr, rgb_mean_rev, test_subject_group, top_k)
                if top_k_rev is not None:
                    pred_rev = predict_with_selected_models(
                        top_k_rev, X_rev_df, feature_cols, rev_folder_slice, 0, segment_length, global_model)
                    if pred_rev is not None:
                        rev_preds.append(pred_rev)
                        rev_labels.append(y_rev)

        # Aggregate per subject
        for label, preds_list, labels_list in [
            ("Original", orig_preds, orig_labels),
            ("Reversed", rev_preds, rev_labels),
        ]:
            if len(preds_list) == 0:
                continue
            y_p = finalize(np.concatenate(preds_list))
            y_t = np.concatenate(labels_list)
            subj_range = float(np.max(y_t) - np.min(y_t))
            if subj_range < MIN_SpO2_RANGE:
                print(f"  Skip {test_subject_id} ({label}): range {subj_range:.1f} < {MIN_SpO2_RANGE}")
                continue
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            pcc = pearsonr(y_t, y_p)[0] if np.std(y_p) > 1e-8 else np.nan
            results.append({
                "subject_id": test_subject_id,
                "condition": label,
                "R2": r2, "MAE": mae, "PCC": pcc,
                "n_frames": len(y_t),
            })
            print(f"  {label}: R²={r2:.4f}, MAE={mae:.2f}, PCC={pcc:.4f}")

    return pd.DataFrame(results)


def frame_index_baseline(feat_df, segment_length=SEGMENT_LENGTH):
    """
    Baseline: predict SpO2 using only normalized frame index (0→1).
    LOSO cross-validation with subject_group.
    """
    req_cols = ["subject_id", "subject_group", "folder_name", "SpO2_win_last"]
    df = feat_df[req_cols].copy()

    results = []
    for test_subject_id, test_subdf in df.groupby("subject_id", sort=False):
        test_group = test_subdf["subject_group"].iloc[0]
        train_df = df[df["subject_group"] != test_group]

        # Train: linear regression on normalized frame index
        train_preds_all, train_labels_all = [], []
        for _, fdf in train_df.groupby("folder_name", sort=False):
            n = len(fdf)
            if n < segment_length:
                continue
            x = np.linspace(0, 1, n).reshape(-1, 1)
            y = fdf["SpO2_win_last"].values
            train_preds_all.append(x)
            train_labels_all.append(y)

        if len(train_preds_all) == 0:
            continue
        X_train = np.concatenate(train_preds_all)
        y_train = np.concatenate(train_labels_all)
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Predict on test
        test_preds, test_labels = [], []
        for _, fdf in test_subdf.groupby("folder_name", sort=False):
            n = len(fdf)
            if n < segment_length:
                continue
            x = np.linspace(0, 1, n).reshape(-1, 1)
            y = fdf["SpO2_win_last"].values
            pred = np.clip(lr.predict(x), 70, 100)
            test_preds.append(pred)
            test_labels.append(y)

        if len(test_preds) == 0:
            continue
        y_p = np.concatenate(test_preds)
        y_t = np.concatenate(test_labels)
        subj_range = float(np.max(y_t) - np.min(y_t))
        if subj_range < MIN_SpO2_RANGE:
            continue
        r2 = r2_score(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)
        pcc = pearsonr(y_t, y_p)[0] if np.std(y_p) > 1e-8 else np.nan
        results.append({
            "subject_id": test_subject_id,
            "condition": "Frame-Index Baseline",
            "R2": r2, "MAE": mae, "PCC": pcc,
            "n_frames": len(y_t),
        })

    return pd.DataFrame(results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model pool
    model_pool_path = os.path.join(OUTPUT_DIR, "model", "model_pool.joblib")
    if not os.path.exists(model_pool_path):
        print(f"Model pool not found at {model_pool_path}. Run train.py first.")
        return
    model_pool = joblib.load(model_pool_path)
    print(f"Loaded model pool: {len(model_pool)} models")

    summary_rows = []

    for csv_path in DATA_CSV_PATHS:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        feat_df = load_features_from_csv_paths([csv_path], verbose=True)
        if feat_df is None or len(feat_df) == 0:
            print(f"  No features loaded for {dataset_name}, skipping.")
            continue
        feat_df = filter_feat_df_by_spo2_range(feat_df)

        # Time reversal test
        results_df = evaluate_original_and_reversed(model_pool, feat_df)

        # Frame-index baseline
        baseline_df = frame_index_baseline(feat_df)

        # Combine
        all_df = pd.concat([results_df, baseline_df], ignore_index=True)

        # Save per-subject results
        out_path = os.path.join(OUTPUT_DIR, f"time_reversal_{dataset_name}.csv")
        all_df.to_csv(out_path, index=False)
        print(f"\nSaved per-subject results to {out_path}")

        # Summary statistics
        for condition in ["Original", "Reversed", "Frame-Index Baseline"]:
            cond_df = all_df[all_df["condition"] == condition]
            if len(cond_df) == 0:
                continue
            # Overall PCC
            overall_pcc = np.nan
            if len(cond_df) > 0:
                all_pred_frames = cond_df["n_frames"].sum()
            summary_rows.append({
                "Dataset": dataset_name,
                "Condition": condition,
                "N_Subjects": len(cond_df),
                "Mean_R2": cond_df["R2"].mean(),
                "Mean_MAE": cond_df["MAE"].mean(),
                "Mean_PCC": cond_df["PCC"].mean(),
            })
            print(f"\n  {condition}: N={len(cond_df)}, "
                  f"Mean R²={cond_df['R2'].mean():.4f}, "
                  f"Mean MAE={cond_df['MAE'].mean():.2f}, "
                  f"Mean PCC={cond_df['PCC'].mean():.4f}")

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUTPUT_DIR, "time_reversal_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to {summary_path}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
