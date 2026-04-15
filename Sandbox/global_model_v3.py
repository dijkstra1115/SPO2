"""
Global Model v3: Shorter segments + feature selection.
Optimized for speed: fewer GBR estimators, progress tracking.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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


def aggregate_segments(df, feature_cols, segment_length):
    """Aggregate windows into segments with rich stats."""
    rows = []
    for (sid, folder), gdf in df.groupby(["subject_id", "folder_name"], sort=False):
        n = len(gdf)
        if n < segment_length:
            continue
        n_segments = n // segment_length
        vals_dict = {col: gdf[col].values for col in feature_cols}
        spo2_vals = gdf[SPO2_COL].values
        for seg_idx in range(n_segments):
            s, e = seg_idx * segment_length, (seg_idx + 1) * segment_length
            row = {"subject_id": sid, "SpO2_mean": np.mean(spo2_vals[s:e])}
            for col in feature_cols:
                v = vals_dict[col][s:e]
                row[f"{col}_mean"] = np.mean(v)
                row[f"{col}_std"] = np.std(v)
                row[f"{col}_p10"] = np.percentile(v, 10)
                row[f"{col}_p90"] = np.percentile(v, 90)
            rows.append(row)
    return pd.DataFrame(rows)


def get_top_features(seg_df, x_cols, y_col="SpO2_mean", top_n=30):
    """Rank features by GBR importance."""
    X = seg_df[x_cols].values
    y = seg_df[y_col].values
    valid = np.isfinite(X).all(axis=1)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X[valid], y[valid])
    idx = np.argsort(model.feature_importances_)[::-1]
    top = [x_cols[i] for i in idx[:top_n]]
    print(f"  Top {top_n} features:", flush=True)
    for r, i in enumerate(idx[:min(top_n, 10)]):
        print(f"    {r+1}. {x_cols[i]}: {model.feature_importances_[i]:.4f}", flush=True)
    return top


def run_loso(seg_df, x_cols, y_col="SpO2_mean", model_cls=None, model_kw=None):
    """LOSO CV with timing."""
    if model_cls is None:
        model_cls = GradientBoostingRegressor
    if model_kw is None:
        model_kw = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                        subsample=0.8, min_samples_leaf=5, random_state=42)

    subjects = seg_df["subject_id"].unique()
    results = []
    t0 = time.time()

    # Pre-extract arrays for speed
    X_all = seg_df[x_cols].values
    y_all = seg_df[y_col].values
    sid_all = seg_df["subject_id"].values

    for i, test_sid in enumerate(subjects):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(subjects) - i - 1)
            print(f"  LOSO: {i+1}/{len(subjects)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

        train_mask = sid_all != test_sid
        test_mask = sid_all == test_sid

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        # NaN/Inf filter
        tv = np.isfinite(X_train).all(axis=1)
        X_train, y_train = X_train[tv], y_train[tv]
        tv = np.isfinite(X_test).all(axis=1)
        X_test, y_test = X_test[tv], y_test[tv]

        if len(X_test) < 2 or len(X_train) < 10:
            continue
        spo2_range = float(np.ptp(y_test))
        if spo2_range < MIN_SpO2_RANGE:
            continue

        model = model_cls(**model_kw)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 70, 100)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = np.nan if np.std(y_pred) < 1e-8 else pearsonr(y_test, y_pred)[0]
        results.append({"subject_id": test_sid, "n": len(y_test),
                        "R2": r2, "MAE": mae, "PCC": pcc})

    elapsed = time.time() - t0
    print(f"  LOSO done: {len(results)} subjects, {elapsed:.0f}s", flush=True)
    return pd.DataFrame(results)


def report(label, results_df):
    for ds_name, prefix in EVAL_DATASETS.items():
        mask = results_df["subject_id"].str.startswith(prefix + "_")
        ds = results_df[mask]
        if len(ds) > 0:
            print(f"    {ds_name}: n={len(ds)}, R2={ds['R2'].mean():.4f}, "
                  f"MAE={ds['MAE'].mean():.2f}, PCC={ds['PCC'].mean():.4f}", flush=True)
    overall = results_df[["R2", "MAE", "PCC"]].mean()
    print(f"    ALL: R2={overall['R2']:.4f}, MAE={overall['MAE']:.2f}, PCC={overall['PCC']:.4f}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("Global Model v3: Shorter segments + Feature selection", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading features...", flush=True)
    feat_df = load_features_from_csv_paths(TRAIN_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)
    sys.stdout.flush()

    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]
    print(f"\n{len(raw_features)} raw features, {len(feat_df)} samples, "
          f"{feat_df['subject_id'].nunique()} subjects", flush=True)

    # ================================================================
    # Exp 1: Segment length sweep (GBR n=150, 4 agg stats = 76 features)
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Segment length (GBR n=150, 76 features)", flush=True)
    print(f"{'='*60}", flush=True)

    agg_suffixes = ["_mean", "_std", "_p10", "_p90"]

    for seg_len in [300, 450, 600, 900]:
        seg_df = aggregate_segments(feat_df, raw_features, segment_length=seg_len)
        x_cols = [f"{f}{s}" for f in raw_features for s in agg_suffixes if f"{f}{s}" in seg_df.columns]
        print(f"\n  seg={seg_len}: {len(seg_df)} segments, {len(x_cols)} features", flush=True)
        results = run_loso(seg_df, x_cols)
        if len(results) > 0:
            report(f"seg={seg_len}", results)

    # ================================================================
    # Exp 2: Feature selection on seg=300 (best expected)
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: Feature selection (seg=300)", flush=True)
    print(f"{'='*60}", flush=True)

    seg_df_300 = aggregate_segments(feat_df, raw_features, segment_length=300)
    all_x = [f"{f}{s}" for f in raw_features for s in agg_suffixes if f"{f}{s}" in seg_df_300.columns]
    top_feats = get_top_features(seg_df_300, all_x, top_n=40)

    for top_n in [15, 25, 40, len(all_x)]:
        sel = top_feats[:top_n] if top_n <= len(top_feats) else all_x
        label = f"top{top_n}" if top_n <= 40 else f"all{len(all_x)}"
        print(f"\n  {label}:", flush=True)
        results = run_loso(seg_df_300, sel)
        if len(results) > 0:
            report(label, results)

    # ================================================================
    # Exp 3: RF vs GBR (seg=300, top 25)
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: RF vs GBR (seg=300, top25)", flush=True)
    print(f"{'='*60}", flush=True)

    top25 = top_feats[:25]

    print(f"\n  RF:", flush=True)
    results = run_loso(seg_df_300, top25,
                       model_cls=RandomForestRegressor,
                       model_kw=dict(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                     random_state=42, n_jobs=-1))
    if len(results) > 0:
        report("RF", results)

    print(f"\n  GBR deeper:", flush=True)
    results = run_loso(seg_df_300, top25,
                       model_kw=dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=3, random_state=42))
    if len(results) > 0:
        report("GBR_deep", results)

    print(f"\n  GBR conservative:", flush=True)
    results = run_loso(seg_df_300, top25,
                       model_kw=dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                                     subsample=0.7, min_samples_leaf=10, random_state=42))
    if len(results) > 0:
        report("GBR_cons", results)

    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("v2 LOSO baseline: c930=0.018, i16=0.141, i16m=0.033", flush=True)
    print("Target: R2 > 0.2", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
