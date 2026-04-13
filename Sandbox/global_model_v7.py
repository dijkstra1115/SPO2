"""
Global Model v7: Device indicator features + segment length sweep.

Key idea: Instead of post-hoc calibration, encode device identity directly
as features so GBR can learn device-specific offsets internally.
Also sweep segment lengths to find optimal granularity.

Exp 1: seg=600, devnorm + device indicator features
Exp 2: seg=300, devnorm + device indicator features
Exp 3: seg=450, devnorm + device indicator features
Exp 4: seg=300, devnorm + device indicators + median offset
Exp 5: seg=300, devnorm + device indicators + family indicator
"""
import sys
import time
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
AGG_SUFFIXES = ["_mean", "_std", "_p10", "_p90"]

GBR_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=5, random_state=42)

DEVICE_LIST = ["prc-c920", "prc-i15", "prc-i15m", "prc2-c930", "prc2-i16", "prc2-i16m"]


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


def add_device_features(seg_df, include_family=False):
    """Add one-hot device indicator columns and optionally family indicator."""
    for dev in DEVICE_LIST:
        seg_df[f"dev_{dev}"] = (seg_df["device"] == dev).astype(float)
    if include_family:
        seg_df["is_webcam"] = seg_df["device"].isin(["prc-c920", "prc2-c930"]).astype(float)
    return seg_df


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


def _loso(seg_df, x_cols, y_col="SpO2_mean", use_offset=False):
    """LOSO with device-normalized features and optional median offset."""
    subjects = seg_df["subject_id"].unique()
    device_all = seg_df["device"].values
    X_all = seg_df[x_cols].values
    y_all = seg_df[y_col].values
    sid_all = seg_df["subject_id"].values

    # Identify which columns are device indicators (don't normalize those)
    dev_col_mask = np.array([c.startswith("dev_") or c == "is_webcam" for c in x_cols])
    feat_col_mask = ~dev_col_mask

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

        # Device normalization on feature columns only (not device indicators)
        if feat_col_mask.any():
            for dev in np.unique(dev_train):
                dmask = dev_train == dev
                feat_idx = np.where(feat_col_mask)[0]
                dm = np.nanmean(X_train[np.ix_(dmask, feat_idx)], axis=0)
                ds = np.nanstd(X_train[np.ix_(dmask, feat_idx)], axis=0)
                ds[ds < 1e-10] = 1.0
                X_train[np.ix_(dmask, feat_idx)] = (X_train[np.ix_(dmask, feat_idx)] - dm) / ds
                if dev == test_dev:
                    X_test[:, feat_idx] = (X_test[:, feat_idx] - dm) / ds

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
        y_pred_raw = model.predict(X_test)

        # Optional median offset correction
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
    print("Global Model v7: Device indicators + segment sweep", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading features...", flush=True)
    feat_df = load_features_from_csv_paths(ALL_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)
    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]

    for seg_len in [600, 450, 300]:
        print(f"\n{'='*60}", flush=True)
        print(f"Segment length = {seg_len}", flush=True)
        print(f"{'='*60}", flush=True)

        seg_df = aggregate_segments(feat_df, raw_features, segment_length=seg_len)
        seg_df = add_device_features(seg_df, include_family=False)

        agg_cols = [f"{f}{s}" for f in raw_features for s in AGG_SUFFIXES
                    if f"{f}{s}" in seg_df.columns]
        dev_cols = [f"dev_{d}" for d in DEVICE_LIST]
        x_cols = agg_cols + dev_cols

        print(f"  {len(seg_df)} segments, {seg_df['subject_id'].nunique()} subjects, "
              f"{len(x_cols)} features ({len(agg_cols)} agg + {len(dev_cols)} dev)", flush=True)

        # Device distribution
        for dev in EVAL_DATASETS:
            n = (seg_df["device"] == dev).sum()
            print(f"    {dev}: {n} segments", flush=True)

        # Devnorm + device indicators
        print(f"\n  Devnorm + device indicators (seg={seg_len}):", flush=True)
        res = _loso(seg_df, x_cols)
        met = report(f"devind_seg{seg_len}", res)

        # Devnorm + device indicators + offset
        print(f"\n  Devnorm + device indicators + offset (seg={seg_len}):", flush=True)
        res2 = _loso(seg_df, x_cols, use_offset=True)
        met2 = report(f"devind_offset_seg{seg_len}", res2)

        if met or met2:
            print("\n  !!! TARGET MET - stopping early !!!", flush=True)
            break

    # Extra: seg=300 with family indicator
    print(f"\n{'='*60}", flush=True)
    print("Extra: seg=300, devnorm + device + family indicators", flush=True)
    print(f"{'='*60}", flush=True)
    seg_df = aggregate_segments(feat_df, raw_features, segment_length=300)
    seg_df = add_device_features(seg_df, include_family=True)
    agg_cols = [f"{f}{s}" for f in raw_features for s in AGG_SUFFIXES
                if f"{f}{s}" in seg_df.columns]
    dev_cols = [f"dev_{d}" for d in DEVICE_LIST] + ["is_webcam"]
    x_cols = agg_cols + dev_cols
    print(f"  {len(seg_df)} segments, {len(x_cols)} features", flush=True)

    res3 = _loso(seg_df, x_cols)
    report("devind_family_seg300", res3)

    res4 = _loso(seg_df, x_cols, use_offset=True)
    report("devind_family_offset_seg300", res4)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
