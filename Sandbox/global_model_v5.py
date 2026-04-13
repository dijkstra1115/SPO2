"""
Global Model v5: Targeting R2 > 0.1 on ALL 6 datasets.

Exp 1: Device-specific models (train/eval per device only)
Exp 2: Global model with per-device median offset correction
Exp 3: Device-normalized + per-device offset correction
Exp 4: Window-level prediction (no segment aggregation)
Exp 5: Stacked: device-norm GBR + per-device residual model
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


def _loso_device_specific(seg_df, x_cols, y_col="SpO2_mean"):
    """Exp 1: Train separate model per device. LOSO within each device."""
    all_results = []
    for dev in EVAL_DATASETS:
        dev_df = seg_df[seg_df["device"] == dev].copy()
        subjects = dev_df["subject_id"].unique()
        X_all = dev_df[x_cols].values
        y_all = dev_df[y_col].values
        sid_all = dev_df["subject_id"].values
        t0 = time.time()

        for test_sid in subjects:
            test_mask = sid_all == test_sid
            train_mask = ~test_mask

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
            all_results.append({"subject_id": test_sid, "n": len(y_test),
                                "R2": r2, "MAE": mae, "PCC": pcc})

        print(f"    {dev}: {len([r for r in all_results if r['subject_id'].startswith(dev)])} subjects, "
              f"{time.time()-t0:.0f}s", flush=True)

    return pd.DataFrame(all_results)


def _loso_with_offset(seg_df, x_cols, y_col="SpO2_mean", normalize_by_device=False):
    """Exp 2/3: Global model + per-device median offset correction.

    Instead of linear calibration (which can amplify errors), simply shift
    predictions by the median residual on same-device training data.
    """
    subjects = seg_df["subject_id"].unique()
    device_all = seg_df["device"].values
    X_all = seg_df[x_cols].values.copy()
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

        X_train = X_all[train_mask].copy()
        y_train = y_all[train_mask].copy()
        dev_train = device_all[train_mask]
        X_test = X_all[test_mask].copy()
        y_test = y_all[test_mask].copy()

        # Per-device z-score normalization
        if normalize_by_device:
            test_dev = extract_device(test_sid)
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

        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_train, y_train)

        # Predict on training data to compute per-device offset
        y_train_pred = model.predict(X_train)
        test_dev = extract_device(test_sid)
        dev_mask = dev_train == test_dev
        if dev_mask.sum() >= 3:
            # Median offset: how much does the model systematically over/under-predict
            # for this device?
            residuals = y_train[dev_mask] - y_train_pred[dev_mask]
            offset = np.median(residuals)
        else:
            offset = 0.0

        y_pred = np.clip(model.predict(X_test) + offset, 70, 100)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = np.nan if np.std(y_pred) < 1e-8 else pearsonr(y_test, y_pred)[0]
        results.append({"subject_id": test_sid, "n": len(y_test),
                        "R2": r2, "MAE": mae, "PCC": pcc})

    elapsed = time.time() - t0
    print(f"    LOSO done: {len(results)} subjects, {elapsed:.0f}s", flush=True)
    return pd.DataFrame(results)


def _loso_stacked(seg_df, x_cols, y_col="SpO2_mean"):
    """Exp 5: Two-stage stacked model.

    Stage 1: Device-normalized global GBR → raw prediction
    Stage 2: Per-device Ridge on (raw_pred, device_features) → final prediction

    The residual model learns device-specific scale/offset corrections.
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

        # Stage 1: Global GBR
        gbr = GradientBoostingRegressor(**GBR_PARAMS)
        gbr.fit(X_train, y_train)
        y_train_pred1 = gbr.predict(X_train)
        y_test_pred1 = gbr.predict(X_test)

        # Stage 2: Per-device residual correction using Ridge
        dev_mask = dev_train == test_dev
        if dev_mask.sum() >= 5:
            # Features for residual model: stage1 prediction + top raw features
            residuals = y_train[dev_mask] - y_train_pred1[dev_mask]
            X_res_train = y_train_pred1[dev_mask].reshape(-1, 1)
            X_res_test = y_test_pred1.reshape(-1, 1)

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_res_train, y_train[dev_mask])
            y_pred = np.clip(ridge.predict(X_res_test), 70, 100)
        else:
            y_pred = np.clip(y_test_pred1, 70, 100)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc = np.nan if np.std(y_pred) < 1e-8 else pearsonr(y_test, y_pred)[0]
        results.append({"subject_id": test_sid, "n": len(y_test),
                        "R2": r2, "MAE": mae, "PCC": pcc})

    elapsed = time.time() - t0
    print(f"    LOSO done: {len(results)} subjects, {elapsed:.0f}s", flush=True)
    return pd.DataFrame(results)


def _loso_window_level(feat_df, raw_features, y_col=SPO2_COL):
    """Exp 4: Window-level prediction (no aggregation), then aggregate predictions.

    Each window (19 raw features) predicts SpO2 directly. Predictions are
    aggregated per segment using trimmed mean. This preserves temporal
    granularity and avoids information loss from aggregation.
    """
    # Prepare window-level data
    feat_df = feat_df.copy()
    feat_df["device"] = feat_df["subject_id"].apply(extract_device)

    x_cols = [c for c in raw_features if c in feat_df.columns]
    subjects = feat_df["subject_id"].unique()
    X_all = feat_df[x_cols].values
    y_all = feat_df[y_col].values
    sid_all = feat_df["subject_id"].values
    dev_all = feat_df["device"].values
    results = []
    t0 = time.time()

    for i, test_sid in enumerate(subjects):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(subjects) - i - 1)
            print(f"    LOSO: {i+1}/{len(subjects)} ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

        test_mask = sid_all == test_sid
        train_mask = ~test_mask

        X_train = X_all[train_mask].copy()
        y_train = y_all[train_mask].copy()
        dev_train = dev_all[train_mask]
        X_test = X_all[test_mask].copy()
        y_test = y_all[test_mask].copy()

        # Device normalization
        test_dev = extract_device(test_sid)
        for dev in np.unique(dev_train):
            dmask = dev_train == dev
            dm = np.nanmean(X_train[dmask], axis=0)
            ds = np.nanstd(X_train[dmask], axis=0)
            ds[ds < 1e-10] = 1.0
            X_train[dmask] = (X_train[dmask] - dm) / ds
            if dev == test_dev:
                X_test = (X_test - dm) / ds

        # NaN/Inf filter
        tv = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train, y_train = X_train[tv], y_train[tv]
        dev_train = dev_train[tv]
        tv = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
        X_test, y_test = X_test[tv], y_test[tv]

        if len(X_test) < SEG_LEN or len(X_train) < 100:
            continue

        # Check SpO2 range on segment-aggregated basis
        n_segs = len(y_test) // SEG_LEN
        if n_segs < 1:
            continue
        seg_means = [np.mean(y_test[j*SEG_LEN:(j+1)*SEG_LEN]) for j in range(n_segs)]
        if np.ptp(seg_means) < MIN_SpO2_RANGE:
            continue

        # Subsample training data for speed (window-level has way more samples)
        if len(X_train) > 50000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), 50000, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]

        # Lighter model for window-level (many more samples)
        model = GradientBoostingRegressor(
            n_estimators=80, max_depth=2, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42)
        model.fit(X_train, y_train)

        # Per-device offset
        y_train_pred = model.predict(X_train)
        dev_mask_train = dev_train == test_dev if len(dev_train) == len(y_train) else np.zeros(len(y_train), dtype=bool)
        # After subsampling, dev_train may be misaligned; recalculate
        # Skip offset for window-level since we subsampled
        offset = 0.0

        y_pred_raw = model.predict(X_test) + offset

        # Aggregate predictions per segment using trimmed mean
        pred_seg_means = []
        true_seg_means = []
        for j in range(n_segs):
            s, e = j * SEG_LEN, (j + 1) * SEG_LEN
            preds = y_pred_raw[s:e]
            # Trimmed mean: drop top/bottom 10%
            lo, hi = np.percentile(preds, 10), np.percentile(preds, 90)
            mask = (preds >= lo) & (preds <= hi)
            if mask.sum() > 0:
                pred_seg_means.append(np.mean(preds[mask]))
            else:
                pred_seg_means.append(np.mean(preds))
            true_seg_means.append(np.mean(y_test[s:e]))

        y_pred_seg = np.clip(np.array(pred_seg_means), 70, 100)
        y_true_seg = np.array(true_seg_means)

        if len(y_pred_seg) < 2:
            continue

        r2 = r2_score(y_true_seg, y_pred_seg)
        mae = mean_absolute_error(y_true_seg, y_pred_seg)
        pcc = np.nan if np.std(y_pred_seg) < 1e-8 else pearsonr(y_true_seg, y_pred_seg)[0]
        results.append({"subject_id": test_sid, "n": len(y_pred_seg),
                        "R2": r2, "MAE": mae, "PCC": pcc})

    elapsed = time.time() - t0
    print(f"    LOSO done: {len(results)} subjects, {elapsed:.0f}s", flush=True)
    return pd.DataFrame(results)


def main():
    print("=" * 60, flush=True)
    print("Global Model v5: Targeting R2 > 0.1 on ALL 6 datasets", flush=True)
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

    # ================================================================
    # Exp 1: Device-specific models
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Device-specific models (per-device LOSO)", flush=True)
    print(f"{'='*60}", flush=True)
    res1 = _loso_device_specific(seg_df, x_cols)
    report("device_specific", res1)

    # ================================================================
    # Exp 2: Global + per-device median offset
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: Global + per-device median offset", flush=True)
    print(f"{'='*60}", flush=True)
    res2 = _loso_with_offset(seg_df, x_cols)
    report("global_offset", res2)

    # ================================================================
    # Exp 3: Device-normalized + per-device median offset
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: Device-normalized + median offset", flush=True)
    print(f"{'='*60}", flush=True)
    res3 = _loso_with_offset(seg_df, x_cols, normalize_by_device=True)
    report("devnorm_offset", res3)

    # ================================================================
    # Exp 4: Window-level prediction
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 4: Window-level prediction (no aggregation)", flush=True)
    print(f"{'='*60}", flush=True)
    res4 = _loso_window_level(feat_df, raw_features)
    report("window_level", res4)

    # ================================================================
    # Exp 5: Stacked model (GBR + per-device Ridge)
    # ================================================================
    print(f"\n{'='*60}", flush=True)
    print("Exp 5: Stacked (dev-norm GBR + per-device Ridge)", flush=True)
    print(f"{'='*60}", flush=True)
    res5 = _loso_stacked(seg_df, x_cols)
    report("stacked", res5)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
