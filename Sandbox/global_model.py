"""
Global Model approach for SpO2 prediction.
Replaces model pool + similarity ensemble with a single global model + LOSO CV.
Uses ratio-of-ratios features motivated by Beer-Lambert law.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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

# Feature sets to experiment with
BASE_FEATURES = ["R_acdc", "G_acdc", "B_acdc"]
RATIO_FEATURES = ["RoR_RG_acdc", "RoR_RB_acdc"]
LONG_FEATURES = ["R_acdc_long", "G_acdc_long", "B_acdc_long", "RoR_RG_acdc_long", "RoR_RB_acdc_long"]
DERIVED_FEATURES = ["POS_Y_acdc", "CHROM_X_acdc"]
DELTA_FEATURES = ["delta_R_acdc", "delta_G_acdc", "delta_B_acdc"]
SQI_FEATURE = ["sqi"]
SCALE_FEATURES = ["acdc_ratio_long_short_R", "acdc_ratio_long_short_G", "acdc_ratio_long_short_B"]

SPO2_COL = "SpO2_win_last"


def aggregate_segments(df, feature_cols, segment_length=SEGMENT_LENGTH):
    """Aggregate per-window features into segment-level (mean+std per feature)."""
    rows = []
    for (sid, folder), gdf in df.groupby(["subject_id", "folder_name"], sort=False):
        n = len(gdf)
        if n < segment_length:
            continue
        n_segments = n // segment_length
        for seg_idx in range(n_segments):
            start = seg_idx * segment_length
            end = start + segment_length
            seg = gdf.iloc[start:end]
            row = {
                "subject_id": sid,
                "folder_name": folder,
                "segment_id": seg_idx,
                "SpO2_mean": seg[SPO2_COL].mean(),
            }
            for col in feature_cols:
                row[f"{col}_seg_mean"] = seg[col].mean()
                row[f"{col}_seg_std"] = seg[col].std()
            rows.append(row)
    return pd.DataFrame(rows)


def run_loso(feat_df, feature_cols, model_type="ridge", use_segments=True):
    """Leave-One-Subject-Out cross-validation. Returns per-subject results."""
    if use_segments:
        df = aggregate_segments(feat_df, feature_cols)
        x_cols = [f"{c}_seg_mean" for c in feature_cols] + [f"{c}_seg_std" for c in feature_cols]
        y_col = "SpO2_mean"
    else:
        df = feat_df.copy()
        x_cols = feature_cols
        y_col = SPO2_COL

    # Verify all feature columns exist
    missing = [c for c in x_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        return pd.DataFrame()

    subjects = df["subject_id"].unique()
    results = []

    for test_sid in subjects:
        train_mask = df["subject_id"] != test_sid
        test_mask = df["subject_id"] == test_sid

        X_train = df.loc[train_mask, x_cols].values
        y_train = df.loc[train_mask, y_col].values
        X_test = df.loc[test_mask, x_cols].values
        y_test = df.loc[test_mask, y_col].values

        if len(X_test) < 2 or np.std(y_test) < 1e-8:
            continue

        spo2_range = float(np.max(y_test) - np.min(y_test))
        if spo2_range < MIN_SpO2_RANGE:
            continue

        if model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "gbr":
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )

        # Handle NaN/Inf
        train_valid = np.isfinite(X_train).all(axis=1)
        test_valid = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_test_clean, y_test_clean = X_test[test_valid], y_test[test_valid]

        if len(X_train) < 10 or len(X_test_clean) < 2:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_clean)
        y_pred = np.clip(y_pred, 70, 100)

        r2 = r2_score(y_test_clean, y_pred)
        mae = mean_absolute_error(y_test_clean, y_pred)
        pcc = np.nan if np.std(y_pred) < 1e-8 else pearsonr(y_test_clean, y_pred)[0]

        results.append({
            "subject_id": test_sid,
            "n_samples": len(y_test_clean),
            "R2": r2,
            "MAE": mae,
            "PCC": pcc,
            "spo2_range": spo2_range,
        })

    return pd.DataFrame(results)


def evaluate_by_dataset(results_df, eval_datasets):
    """Group results by dataset prefix and compute summary metrics."""
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
    print("=" * 60)
    print("Global Model SpO2 Prediction (LOSO CV)")
    print("=" * 60)

    print("\nLoading and building features...")
    feat_df = load_features_from_csv_paths(TRAIN_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)

    n_subjects = feat_df["subject_id"].nunique()
    print(f"\nTotal samples: {len(feat_df)}, Total subjects: {n_subjects}")
    print(f"Available columns: {list(feat_df.columns)}")

    # Feature sets to try
    feature_sets = {
        "base": BASE_FEATURES,
        "ratios": RATIO_FEATURES,
        "base+ratios": BASE_FEATURES + RATIO_FEATURES,
        "all_acdc": BASE_FEATURES + RATIO_FEATURES + LONG_FEATURES + DERIVED_FEATURES,
        "kitchen_sink": BASE_FEATURES + RATIO_FEATURES + LONG_FEATURES + DERIVED_FEATURES + DELTA_FEATURES + SQI_FEATURE + SCALE_FEATURES,
    }

    model_types = ["ridge", "gbr"]
    all_experiment_results = []

    for feat_name, feat_cols in feature_sets.items():
        # Check which columns actually exist
        available = [c for c in feat_cols if c in feat_df.columns]
        if len(available) < len(feat_cols):
            missing = set(feat_cols) - set(available)
            print(f"\n  Skipping features not found: {missing}")
        if not available:
            continue
        feat_cols = available

        for model_type in model_types:
            # Only segment-level (window-level too slow with 2M+ samples)
            exp_name = f"{model_type}_{feat_name}"
            print(f"\n{'=' * 60}")
            print(f"Experiment: {exp_name}")
            print(f"  Features ({len(feat_cols)}): {feat_cols}")
            print(f"{'=' * 60}")

            results_df = run_loso(feat_df, feat_cols, model_type=model_type,
                                  use_segments=True)

            if len(results_df) == 0:
                print("  No valid subjects.")
                continue

            ds_summary = evaluate_by_dataset(results_df, EVAL_DATASETS)

            print(f"\n  Per-dataset results:")
            for _, row in ds_summary.iterrows():
                print(f"    {row['dataset']}: n={row['n_subjects']:.0f}, "
                      f"R2={row['mean_R2']:.4f}, MAE={row['mean_MAE']:.2f}, "
                      f"PCC={row['mean_PCC']:.4f}")

            overall_r2 = results_df["R2"].mean()
            overall_mae = results_df["MAE"].mean()
            overall_pcc = results_df["PCC"].mean()
            print(f"\n  Overall: R2={overall_r2:.4f}, MAE={overall_mae:.2f}, PCC={overall_pcc:.4f}")

            for _, row in ds_summary.iterrows():
                all_experiment_results.append({
                    "experiment": exp_name,
                    "dataset": row["dataset"],
                    "n_subjects": row["n_subjects"],
                    "mean_R2": row["mean_R2"],
                    "mean_MAE": row["mean_MAE"],
                    "mean_PCC": row["mean_PCC"],
                })

    # Summary
    exp_df = pd.DataFrame(all_experiment_results)
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(exp_df.to_string(index=False))

    output_path = os.path.join("./output", "global_model_experiments.csv")
    os.makedirs("./output", exist_ok=True)
    exp_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("BEST CONFIG PER DATASET (by mean R2)")
    print("=" * 60)
    for ds_name in EVAL_DATASETS:
        ds_exp = exp_df[exp_df["dataset"] == ds_name]
        if len(ds_exp) == 0:
            continue
        best = ds_exp.loc[ds_exp["mean_R2"].idxmax()]
        print(f"  {ds_name}: {best['experiment']} -> R2={best['mean_R2']:.4f}, "
              f"MAE={best['mean_MAE']:.2f}, PCC={best['mean_PCC']:.4f}")

    # Comparison with current best
    print("\n" + "=" * 60)
    print("COMPARISON WITH CURRENT BEST (model pool + ensemble)")
    print("=" * 60)
    current_best = {"prc2-c930": -0.042, "prc2-i16": -0.225, "prc2-i16m": -0.321}
    for ds_name, curr_r2 in current_best.items():
        ds_exp = exp_df[exp_df["dataset"] == ds_name]
        if len(ds_exp) == 0:
            continue
        best = ds_exp.loc[ds_exp["mean_R2"].idxmax()]
        delta = best["mean_R2"] - curr_r2
        print(f"  {ds_name}: current={curr_r2:.4f} -> new={best['mean_R2']:.4f} (delta={delta:+.4f})")


if __name__ == "__main__":
    main()
