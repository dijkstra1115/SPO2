"""
Global Model v4: Device-aware modeling for R2 > 0.1 on all 6 datasets.

Exp 0: Training pool ablation (which datasets help?)
Exp 1: Per-device calibration (post-hoc linear correction)
Exp 2: Device-normalized features (z-score per device)
Exp 3: Device-normalized + calibration combined
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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


def extract_device(subject_id):
    """'prc2-c930_S001-1' -> 'prc2-c930'"""
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


def _loso_core(seg_df, x_cols, y_col="SpO2_mean", train_filter=None,
               calibrate=False, normalize_by_device=False):
    """Unified LOSO loop with optional device calibration and normalization.

    Args:
        train_filter: function(device) -> bool, to filter training pool
        calibrate: apply per-device post-hoc linear correction
        normalize_by_device: z-score features per device inside CV loop
    """
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
        train_mask = sid_all != test_sid

        # Optional: filter training pool by device
        if train_filter is not None:
            dev_ok = np.array([train_filter(d) for d in device_all])
            train_mask = train_mask & dev_ok

        X_train = X_all[train_mask].copy()
        y_train = y_all[train_mask].copy()
        dev_train = device_all[train_mask]
        X_test = X_all[test_mask].copy()
        y_test = y_all[test_mask].copy()

        # Optional: per-device z-score normalization
        if normalize_by_device:
            for dev in np.unique(dev_train):
                dmask = dev_train == dev
                dm = np.nanmean(X_train[dmask], axis=0)
                ds = np.nanstd(X_train[dmask], axis=0)
                ds[ds < 1e-10] = 1.0
                X_train[dmask] = (X_train[dmask] - dm) / ds
                # Apply same stats to test if same device
                test_dev = extract_device(test_sid)
                if dev == test_dev:
                    X_test = (X_test - dm) / ds

        # NaN/Inf filter
        tv = np.isfinite(X_train).all(axis=1)
        X_train, y_train = X_train[tv], y_train[tv]
        dev_train = dev_train[tv] if calibrate else dev_train
        tv = np.isfinite(X_test).all(axis=1)
        X_test, y_test = X_test[tv], y_test[tv]

        if len(X_test) < 2 or len(X_train) < 10:
            continue
        if np.ptp(y_test) < MIN_SpO2_RANGE:
            continue

        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_train, y_train)
        y_pred_raw = model.predict(X_test)

        # Optional: per-device calibration
        if calibrate:
            y_train_pred = model.predict(X_train)
            test_dev = extract_device(test_sid)
            dev_mask = dev_train == test_dev
            if dev_mask.sum() >= 5:
                cal = LinearRegression()
                cal.fit(y_train_pred[dev_mask].reshape(-1, 1), y_train[dev_mask])
                y_pred = np.clip(cal.predict(y_pred_raw.reshape(-1, 1)), 70, 100)
            else:
                y_pred = np.clip(y_pred_raw, 70, 100)
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
    print("Global Model v4: Device-aware modeling", flush=True)
    print("Target: R2 > 0.1 on ALL 6 datasets", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading features...", flush=True)
    feat_df = load_features_from_csv_paths(ALL_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)

    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]
    print(f"\nAggregating segments (seg={SEG_LEN})...", flush=True)
    seg_df = aggregate_segments(feat_df, raw_features, segment_length=SEG_LEN)
    x_cols = [f"{f}{s}" for f in raw_features for s in AGG_SUFFIXES
              if f"{f}{s}" in seg_df.columns]
    print(f"  {len(seg_df)} segments, {seg_df['subject_id'].nunique()} subjects, "
          f"{len(x_cols)} features", flush=True)

    print("\n  Device distribution:", flush=True)
    for dev in EVAL_DATASETS:
        mask = seg_df["device"] == dev
        n_seg = mask.sum()
        n_subj = seg_df[mask]["subject_id"].nunique()
        print(f"    {dev}: {n_seg} segs, {n_subj} subjects", flush=True)

    # ================================================================
    # Exp 0: Training pool ablation
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 0: Training pool ablation", flush=True)
    print(f"{'='*60}", flush=True)

    # Pool A: all 6 datasets
    print("\n  Pool A: all 6 datasets", flush=True)
    res_a = _loso_core(seg_df, x_cols)
    report("pool_all", res_a)

    # Pool B: prc only (train on prc, eval all)
    print("\n  Pool B: prc only → eval all", flush=True)
    res_b = _loso_core(seg_df, x_cols,
                       train_filter=lambda d: d.startswith("prc-"))
    report("pool_prc", res_b)

    # Pool C: prc2 only (train on prc2, eval all)
    print("\n  Pool C: prc2 only → eval all", flush=True)
    res_c = _loso_core(seg_df, x_cols,
                       train_filter=lambda d: d.startswith("prc2-"))
    report("pool_prc2", res_c)

    # Pool D: same-family (webcams→webcams, phones→phones)
    print("\n  Pool D: same-family (webcam→webcam, phone→phone)", flush=True)
    # For this we need per-subject filtering. Use device type matching.
    def same_family_filter_for(test_sid):
        test_dev = extract_device(test_sid)
        test_is_webcam = "c920" in test_dev or "c930" in test_dev
        def f(d):
            d_is_webcam = "c920" in d or "c930" in d
            return d_is_webcam == test_is_webcam
        return f

    # Need custom loop for Pool D since filter depends on test subject
    subjects = seg_df["subject_id"].unique()
    device_all = seg_df["device"].values
    X_all = seg_df[x_cols].values
    y_all = seg_df["SpO2_mean"].values
    sid_all = seg_df["subject_id"].values
    results_d = []
    t0 = time.time()
    for i, test_sid in enumerate(subjects):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(subjects) - i - 1)
            print(f"    LOSO: {i+1}/{len(subjects)} ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

        test_mask = sid_all == test_sid
        filt = same_family_filter_for(test_sid)
        train_mask = (sid_all != test_sid) & np.array([filt(d) for d in device_all])

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        tv = np.isfinite(X_train).all(axis=1)
        X_train, y_train = X_train[tv], y_train[tv]
        tv = np.isfinite(X_test).all(axis=1)
        X_test, y_test = X_test[tv], y_test[tv]

        if len(X_test) < 2 or len(X_train) < 10:
            continue
        if np.ptp(y_test) < MIN_SpO2_RANGE:
            continue

        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 70, 100)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = np.nan if np.std(y_pred) < 1e-8 else pearsonr(y_test, y_pred)[0]
        results_d.append({"subject_id": test_sid, "n": len(y_test),
                          "R2": r2, "MAE": mae, "PCC": pcc})
    print(f"    LOSO done: {len(results_d)} subjects, {time.time()-t0:.0f}s", flush=True)
    report("pool_family", pd.DataFrame(results_d))

    # ================================================================
    # Exp 1: Per-device calibration (using full pool)
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Per-device calibration", flush=True)
    print(f"{'='*60}", flush=True)
    res_cal = _loso_core(seg_df, x_cols, calibrate=True)
    report("calibrated", res_cal)

    # ================================================================
    # Exp 2: Device-normalized features
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: Device-normalized features", flush=True)
    print(f"{'='*60}", flush=True)
    res_norm = _loso_core(seg_df, x_cols, normalize_by_device=True)
    report("dev_norm", res_norm)

    # ================================================================
    # Exp 3: Device-normalized + calibration
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: Device-normalized + calibration", flush=True)
    print(f"{'='*60}", flush=True)
    res_nc = _loso_core(seg_df, x_cols, normalize_by_device=True, calibrate=True)
    report("dev_norm_cal", res_nc)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
