"""
Global Model v9: Target catastrophic failures.

Analysis shows c930 avg R2≈0 is dragged down by a few subjects with
extreme bias (S082: bias=+10.6, R2=-4.4). These subjects have low SpO2
(83-88) where the model has few training examples.

Exp 1: Sample-weighted GBR (upweight low-SpO2 segments)
Exp 2: SpO2-balanced training (oversample rare SpO2 ranges)
Exp 3: Devnorm + median offset + larger model (combine best elements)
Exp 4: Devnorm + per-subject prediction centering via train residual pattern
Exp 5: Ridge regression instead of GBR (less range compression)
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

from common import (
    SEGMENT_LENGTH,
    MIN_SpO2_RANGE,
    load_features_from_csv_paths,
    filter_feat_df_by_spo2_range,
)

ALL_CSV_PATHS = [
    "./data/prc-c920.csv",
    "./data/prc-i15.csv",
    "./data/prc-i15m.csv",
    "./data/prc2-c930.csv",
    "./data/prc2-i16.csv",
    "./data/prc2-i16m.csv",
]

EVAL_DATASETS = [
    "prc-c920", "prc-i15", "prc-i15m",
    "prc2-c930", "prc2-i16", "prc2-i16m",
]

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
SEG_LEN = 600
AGG_SUFFIXES = ["_mean", "_std", "_p10", "_p90"]

GBR_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=5, random_state=42)

GBR_LARGE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, min_samples_leaf=5, random_state=42)


def extract_device(subject_id):
    idx = subject_id.rfind("_")
    return subject_id[:idx] if idx >= 0 else subject_id


def aggregate_segments(df, feature_cols, segment_length):
    rows = []
    for (sid, folder), gdf in df.groupby(["subject_id", "folder_name"], sort=False):
        n = len(gdf)
        if n < segment_length:
            continue
        n_segments = n // segment_length
        vals_dict = {col: gdf[col].values for col in feature_cols}
        spo2_vals = gdf[SPO2_COL].values
        device = extract_device(sid)
        for seg_idx in range(n_segments):
            s, e = seg_idx * segment_length, (seg_idx + 1) * segment_length
            row = {"subject_id": sid, "device": device,
                   "SpO2_mean": np.mean(spo2_vals[s:e])}
            for col in feature_cols:
                v = vals_dict[col][s:e]
                row[f"{col}_mean"] = np.mean(v)
                row[f"{col}_std"] = np.std(v)
                row[f"{col}_p10"] = np.percentile(v, 10)
                row[f"{col}_p90"] = np.percentile(v, 90)
            rows.append(row)
    return pd.DataFrame(rows)


def report(label, results_df):
    print(f"\n  [{label}]", flush=True)
    all_above = True
    for ds in EVAL_DATASETS:
        mask = results_df["subject_id"].str.startswith(ds + "_")
        sub = results_df[mask]
        if len(sub) > 0:
            r2 = sub["R2"].mean()
            mark = "OK" if r2 >= 0.1 else "X "
            print(f"    {mark} {ds}: n={len(sub)}, R2={r2:.4f}, "
                  f"MAE={sub['MAE'].mean():.2f}, PCC={sub['PCC'].mean():.4f}", flush=True)
            if r2 < 0.1:
                all_above = False
        else:
            print(f"    -- {ds}: no subjects", flush=True)
            all_above = False
    overall = results_df[["R2", "MAE", "PCC"]].mean()
    print(f"    ALL: n={len(results_df)}, R2={overall['R2']:.4f}, MAE={overall['MAE']:.2f}, "
          f"PCC={overall['PCC']:.4f}", flush=True)
    if all_above:
        print("    >>> TARGET MET: all datasets R2 >= 0.1 <<<", flush=True)
    return all_above


def _compute_sample_weights(y, method="inverse_freq"):
    """Compute sample weights to upweight rare SpO2 values."""
    if method == "inverse_freq":
        # Bin SpO2 into 2% bins, weight inversely proportional to bin count
        bins = np.arange(70, 102, 2)
        bin_idx = np.digitize(y, bins) - 1
        bin_counts = np.bincount(bin_idx, minlength=len(bins))
        bin_counts = np.maximum(bin_counts, 1)
        weights = 1.0 / bin_counts[bin_idx]
        weights = weights / weights.mean()  # normalize to mean=1
        return weights
    return np.ones(len(y))


def _loso(seg_df, x_cols, y_col="SpO2_mean",
          use_offset=False, weighted=False, use_ridge=False,
          gbr_params=None):
    """LOSO with device normalization."""
    if gbr_params is None:
        gbr_params = GBR_PARAMS

    subjects = seg_df["subject_id"].unique()
    device_all = seg_df["device"].values
    X_all = seg_df[x_cols].values
    y_all = seg_df[y_col].values
    sid_all = seg_df["subject_id"].values
    results = []
    t0 = time.time()

    for i, test_sid in enumerate(subjects):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(subjects) - i - 1)
            print(f"    LOSO: {i+1}/{len(subjects)} ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

        test_mask = sid_all == test_sid
        train_mask = ~test_mask
        test_dev = extract_device(test_sid)

        X_train = X_all[train_mask].copy()
        y_train = y_all[train_mask].copy()
        dev_train = device_all[train_mask]
        X_test = X_all[test_mask].copy()
        y_test = y_all[test_mask].copy()

        # Device normalization
        for dev in np.unique(dev_train):
            dmask = dev_train == dev
            dm = np.nanmean(X_train[dmask], axis=0)
            ds = np.nanstd(X_train[dmask], axis=0)
            ds[ds < 1e-10] = 1.0
            X_train[dmask] = (X_train[dmask] - dm) / ds
            if dev == test_dev:
                X_test = (X_test - dm) / ds

        # NaN/Inf filter
        tv = np.isfinite(X_train).all(axis=1)
        X_train, y_train, dev_train = X_train[tv], y_train[tv], dev_train[tv]
        tv = np.isfinite(X_test).all(axis=1)
        X_test, y_test = X_test[tv], y_test[tv]

        if len(X_test) < 2 or len(X_train) < 10:
            continue
        if np.ptp(y_test) < MIN_SpO2_RANGE:
            continue

        if use_ridge:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
        else:
            model = GradientBoostingRegressor(**gbr_params)
            if weighted:
                sw = _compute_sample_weights(y_train)
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)

        y_pred_raw = model.predict(X_test)

        if use_offset:
            y_train_pred = model.predict(X_train)
            dev_mask = dev_train == test_dev
            if dev_mask.sum() >= 3:
                offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            else:
                offset = 0.0
            y_pred = np.clip(y_pred_raw + offset, 70, 100)
        else:
            y_pred = np.clip(y_pred_raw, 70, 100)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = np.nan if np.std(y_pred) < 1e-8 else pearsonr(y_test, y_pred)[0]
        results.append({"subject_id": test_sid, "n": len(y_test),
                        "R2": r2, "MAE": mae, "PCC": pcc})

    elapsed = time.time() - t0
    print(f"    LOSO done: {len(results)} subjects, {elapsed:.0f}s", flush=True)
    return pd.DataFrame(results)


def main():
    print("=" * 60, flush=True)
    print("Global Model v9: Target catastrophic failures", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading features...", flush=True)
    feat_df = load_features_from_csv_paths(ALL_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)

    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]
    seg_df = aggregate_segments(feat_df, raw_features, segment_length=SEG_LEN)
    x_cols = [f"{f}{s}" for f in raw_features for s in AGG_SUFFIXES
              if f"{f}{s}" in seg_df.columns]
    print(f"  {len(seg_df)} segments, {seg_df['subject_id'].nunique()} subjects, "
          f"{len(x_cols)} features", flush=True)

    # Exp 1: Sample-weighted GBR
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Devnorm + sample-weighted GBR", flush=True)
    print(f"{'='*60}", flush=True)
    res1 = _loso(seg_df, x_cols, weighted=True)
    report("weighted", res1)

    # Exp 2: Sample-weighted + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: Devnorm + weighted + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res2 = _loso(seg_df, x_cols, weighted=True, use_offset=True)
    report("weighted_offset", res2)

    # Exp 3: Larger model + offset (combine best from v8 + v5)
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: Devnorm + larger model + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res3 = _loso(seg_df, x_cols, use_offset=True, gbr_params=GBR_LARGE)
    report("large_offset", res3)

    # Exp 4: Weighted + larger model + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 4: Devnorm + weighted + larger model + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res4 = _loso(seg_df, x_cols, weighted=True, use_offset=True, gbr_params=GBR_LARGE)
    report("weighted_large_offset", res4)

    # Exp 5: Ridge regression (linear, no range compression)
    print(f"\n{'='*60}", flush=True)
    print("Exp 5: Devnorm + Ridge regression", flush=True)
    print(f"{'='*60}", flush=True)
    res5 = _loso(seg_df, x_cols, use_ridge=True)
    report("ridge", res5)

    # Exp 6: Ridge + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 6: Devnorm + Ridge + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res6 = _loso(seg_df, x_cols, use_ridge=True, use_offset=True)
    report("ridge_offset", res6)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
