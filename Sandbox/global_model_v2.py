"""
Global Model v2: GBR with rich segment features and proper Leave-One-Group-Out CV.
Fixes data leakage: excludes all data from the same physical person across devices.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

from common import (
    SEGMENT_LENGTH,
    MIN_SpO2_RANGE,
    load_features_from_csv_paths,
    filter_feat_df_by_spo2_range,
)

TRAIN_CSV_PATHS = [
    "./data/prc-c920.csv",
    "./data/prc-i15.csv",
    "./data/prc-i15m.csv",
    "./data/prc2-c930.csv",
    "./data/prc2-i16.csv",
    "./data/prc2-i16m.csv",
]

EVAL_DATASETS = {
    "prc2-c930": "prc2-c930",
    "prc2-i16": "prc2-i16",
    "prc2-i16m": "prc2-i16m",
}

ALL_RAW_FEATURES = [
    "R_acdc", "G_acdc", "B_acdc",
    "RoR_RG_acdc", "RoR_RB_acdc",
    "R_acdc_long", "G_acdc_long", "B_acdc_long",
    "RoR_RG_acdc_long", "RoR_RB_acdc_long",
    "POS_Y_acdc", "CHROM_X_acdc",
    "delta_R_acdc", "delta_G_acdc", "delta_B_acdc",
    "sqi",
    "acdc_ratio_long_short_R", "acdc_ratio_long_short_G", "acdc_ratio_long_short_B",
]

SPO2_COL = "SpO2_win_last"


def aggregate_segments_rich(df, feature_cols, segment_length=SEGMENT_LENGTH):
    """
    Aggregate per-window features into rich segment-level features.
    Per feature: mean, std, median, p10, p90, range, slope (linear trend).
    """
    rows = []
    for (sid, folder), gdf in df.groupby(["subject_id", "folder_name"], sort=False):
        n = len(gdf)
        if n < segment_length:
            continue
        subject_group = gdf["subject_group"].iloc[0] if "subject_group" in gdf.columns else sid
        n_segments = n // segment_length
        for seg_idx in range(n_segments):
            start = seg_idx * segment_length
            end = start + segment_length
            seg = gdf.iloc[start:end]
            spo2_vals = seg[SPO2_COL].values
            row = {
                "subject_id": sid,
                "subject_group": subject_group,
                "folder_name": folder,
                "segment_id": seg_idx,
                "SpO2_mean": np.mean(spo2_vals),
                "SpO2_std": np.std(spo2_vals),
            }
            for col in feature_cols:
                vals = seg[col].values
                row[f"{col}_mean"] = np.mean(vals)
                row[f"{col}_std"] = np.std(vals)
                row[f"{col}_median"] = np.median(vals)
                row[f"{col}_p10"] = np.percentile(vals, 10)
                row[f"{col}_p90"] = np.percentile(vals, 90)
                row[f"{col}_range"] = np.ptp(vals)
                # Linear slope (trend over segment)
                x = np.arange(len(vals))
                if np.std(vals) > 1e-10:
                    row[f"{col}_slope"] = np.polyfit(x, vals, 1)[0]
                else:
                    row[f"{col}_slope"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def run_logo_cv(seg_df, x_cols, y_col="SpO2_mean", gbr_params=None):
    """
    Leave-One-Group-Out CV using subject_group (same person across devices).
    Returns per-subject_id results.
    """
    if gbr_params is None:
        gbr_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, min_samples_leaf=5, random_state=42)

    groups = seg_df["subject_group"].unique()
    results = []

    for i, test_group in enumerate(groups):
        if (i + 1) % 20 == 0:
            print(f"  LOGO CV: {i+1}/{len(groups)} groups done", flush=True)
        train_mask = seg_df["subject_group"] != test_group
        test_mask = seg_df["subject_group"] == test_group

        X_train = seg_df.loc[train_mask, x_cols].values
        y_train = seg_df.loc[train_mask, y_col].values
        X_test = seg_df.loc[test_mask, x_cols].values
        y_test = seg_df.loc[test_mask, y_col].values
        test_sids = seg_df.loc[test_mask, "subject_id"].values

        # Handle NaN/Inf
        train_valid = np.isfinite(X_train).all(axis=1)
        test_valid = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_test, y_test = X_test[test_valid], y_test[test_valid]
        test_sids = test_sids[test_valid]

        if len(X_train) < 10 or len(X_test) < 1:
            continue

        model = GradientBoostingRegressor(**gbr_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 70, 100)

        # Report per subject_id within this group
        for sid in np.unique(test_sids):
            sid_mask = test_sids == sid
            y_t = y_test[sid_mask]
            y_p = y_pred[sid_mask]
            if len(y_t) < 2:
                continue
            spo2_range = float(np.max(y_t) - np.min(y_t))
            if spo2_range < MIN_SpO2_RANGE:
                continue
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            pcc = np.nan if np.std(y_p) < 1e-8 else pearsonr(y_t, y_p)[0]
            results.append({
                "subject_id": sid,
                "subject_group": test_group,
                "n_samples": len(y_t),
                "R2": r2, "MAE": mae, "PCC": pcc,
                "spo2_range": spo2_range,
            })

    print(f"  LOGO CV done: {len(groups)} groups, {len(results)} subject results", flush=True)
    return pd.DataFrame(results)


def evaluate_by_dataset(results_df, eval_datasets):
    summary = []
    for ds_name, source_prefix in eval_datasets.items():
        ds_mask = results_df["subject_id"].str.startswith(source_prefix + "_")
        ds_results = results_df[ds_mask]
        if len(ds_results) == 0:
            summary.append({"dataset": ds_name, "n_subjects": 0,
                            "mean_R2": np.nan, "mean_MAE": np.nan, "mean_PCC": np.nan})
            continue
        summary.append({
            "dataset": ds_name,
            "n_subjects": len(ds_results),
            "mean_R2": ds_results["R2"].mean(),
            "mean_MAE": ds_results["MAE"].mean(),
            "mean_PCC": ds_results["PCC"].mean(),
        })
    return pd.DataFrame(summary)


def main():
    print("=" * 60, flush=True)
    print("Global Model v2: GBR + Rich Features + LOGO CV", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading and building features...", flush=True)
    feat_df = load_features_from_csv_paths(TRAIN_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)
    sys.stdout.flush()

    # Use only features that exist
    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]
    print(f"\nUsing {len(raw_features)} raw features: {raw_features}", flush=True)
    print(f"Total samples: {len(feat_df)}, subjects: {feat_df['subject_id'].nunique()}", flush=True)
    if "subject_group" in feat_df.columns:
        print(f"Subject groups: {feat_df['subject_group'].nunique()}", flush=True)
    else:
        print("WARNING: No subject_group column, falling back to subject_id", flush=True)
        feat_df["subject_group"] = feat_df["subject_id"]

    # Aggregate to segments with rich features
    print("\nAggregating to segment-level features...", flush=True)
    seg_df = aggregate_segments_rich(feat_df, raw_features)
    print(f"Segments: {len(seg_df)}, subjects: {seg_df['subject_id'].nunique()}, groups: {seg_df['subject_group'].nunique()}", flush=True)

    # Build x_cols from aggregated features
    agg_suffixes = ["_mean", "_std", "_median", "_p10", "_p90", "_range", "_slope"]
    x_cols = []
    for feat in raw_features:
        for suffix in agg_suffixes:
            col = f"{feat}{suffix}"
            if col in seg_df.columns:
                x_cols.append(col)
    print(f"Total feature columns: {len(x_cols)}", flush=True)

    # Experiment 1: LOGO CV (proper, no leakage)
    print(f"\n{'='*60}", flush=True)
    print("Experiment: GBR + all features + LOGO CV", flush=True)
    print(f"{'='*60}", flush=True)

    results_logo = run_logo_cv(seg_df, x_cols, y_col="SpO2_mean")

    if len(results_logo) > 0:
        ds_summary = evaluate_by_dataset(results_logo, EVAL_DATASETS)
        print("\nLOGO CV Results:", flush=True)
        for _, row in ds_summary.iterrows():
            print(f"  {row['dataset']}: n={row['n_subjects']:.0f}, "
                  f"R2={row['mean_R2']:.4f}, MAE={row['mean_MAE']:.2f}, "
                  f"PCC={row['mean_PCC']:.4f}", flush=True)

    # Experiment 2: LOSO CV (for comparison — potential leakage across devices)
    print(f"\n{'='*60}", flush=True)
    print("Experiment: GBR + all features + LOSO CV (subject_id, may have cross-device leakage)", flush=True)
    print(f"{'='*60}", flush=True)

    # Override subject_group with subject_id for LOSO
    seg_df_loso = seg_df.copy()
    seg_df_loso["subject_group"] = seg_df_loso["subject_id"]
    results_loso = run_logo_cv(seg_df_loso, x_cols, y_col="SpO2_mean")

    if len(results_loso) > 0:
        ds_summary_loso = evaluate_by_dataset(results_loso, EVAL_DATASETS)
        print("\nLOSO CV Results:", flush=True)
        for _, row in ds_summary_loso.iterrows():
            print(f"  {row['dataset']}: n={row['n_subjects']:.0f}, "
                  f"R2={row['mean_R2']:.4f}, MAE={row['mean_MAE']:.2f}, "
                  f"PCC={row['mean_PCC']:.4f}", flush=True)

    # Experiment 3: Try different GBR configs
    print(f"\n{'='*60}", flush=True)
    print("Experiment: GBR hyperparameter sweep (LOGO CV)", flush=True)
    print(f"{'='*60}", flush=True)

    configs = [
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "min_samples_leaf": 10, "random_state": 42},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "min_samples_leaf": 3, "random_state": 42},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "min_samples_leaf": 5, "random_state": 42},
    ]

    for ci, cfg in enumerate(configs):
        print(f"\n  Config {ci}: {cfg}", flush=True)
        results_cfg = run_logo_cv(seg_df, x_cols, y_col="SpO2_mean", gbr_params=cfg)
        if len(results_cfg) > 0:
            ds_summary_cfg = evaluate_by_dataset(results_cfg, EVAL_DATASETS)
            for _, row in ds_summary_cfg.iterrows():
                print(f"    {row['dataset']}: R2={row['mean_R2']:.4f}, MAE={row['mean_MAE']:.2f}, PCC={row['mean_PCC']:.4f}", flush=True)

    # Comparison
    print(f"\n{'='*60}", flush=True)
    print("COMPARISON WITH PREVIOUS BESTS", flush=True)
    print("=" * 60, flush=True)
    current_best = {"prc2-c930": -0.042, "prc2-i16": -0.225, "prc2-i16m": -0.321}
    v1_best = {"prc2-c930": -0.057, "prc2-i16": 0.028, "prc2-i16m": -0.097}

    if len(results_logo) > 0:
        ds_final = evaluate_by_dataset(results_logo, EVAL_DATASETS)
        for _, row in ds_final.iterrows():
            ds = row["dataset"]
            new_r2 = row["mean_R2"]
            print(f"  {ds}: ensemble={current_best.get(ds, 'N/A'):.4f}, v1={v1_best.get(ds, 'N/A'):.4f}, v2(LOGO)={new_r2:.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
