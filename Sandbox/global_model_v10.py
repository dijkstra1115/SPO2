"""
Global Model v10: Robust GBR + hybrid approaches.

Exp 1: GBR with Huber loss (robust to outliers in training)
Exp 2: GBR Huber + devnorm + offset
Exp 3: Two-model blend: global GBR + webcam-only GBR (for webcam subjects)
Exp 4: Two-model blend + offset
Exp 5: Devnorm + offset + feature interaction terms (RoR products)
Exp 6: Devnorm + offset + exclude low-importance features
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
SEG_LEN = 600
AGG_SUFFIXES = ["_mean", "_std", "_p10", "_p90"]

GBR_BASE = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42)

GBR_HUBER = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                 subsample=0.8, min_samples_leaf=5, random_state=42,
                 loss="huber", alpha=0.9)

WEBCAM_DEVICES = {"prc-c920", "prc2-c930"}


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


def add_interaction_features(seg_df, raw_features):
    """Add interaction terms between key features."""
    # RoR product: RG * RB (captures combined ratio behavior)
    for suffix in AGG_SUFFIXES:
        rg = f"RoR_RG_acdc{suffix}"
        rb = f"RoR_RB_acdc{suffix}"
        if rg in seg_df.columns and rb in seg_df.columns:
            seg_df[f"RoR_product{suffix}"] = seg_df[rg] * seg_df[rb]
        # Long window versions
        rg_l = f"RoR_RG_acdc_long{suffix}"
        rb_l = f"RoR_RB_acdc_long{suffix}"
        if rg_l in seg_df.columns and rb_l in seg_df.columns:
            seg_df[f"RoR_product_long{suffix}"] = seg_df[rg_l] * seg_df[rb_l]
        # R/G ratio
        r = f"R_acdc{suffix}"
        g = f"G_acdc{suffix}"
        if r in seg_df.columns and g in seg_df.columns:
            seg_df[f"RG_ratio{suffix}"] = seg_df[r] / (seg_df[g] + 1e-10)
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


def _loso(seg_df, x_cols, y_col="SpO2_mean",
          use_offset=False, gbr_params=None, blend_webcam=False):
    """LOSO with device normalization.

    blend_webcam: train a second model on webcam-only data and blend
    predictions for webcam test subjects.
    """
    if gbr_params is None:
        gbr_params = GBR_BASE

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

        # Global model
        model = GradientBoostingRegressor(**gbr_params)
        model.fit(X_train, y_train)
        y_pred_global = model.predict(X_test)

        if blend_webcam and test_dev in WEBCAM_DEVICES:
            # Train webcam-specific model
            webcam_mask = np.array([d in WEBCAM_DEVICES for d in dev_train])
            if webcam_mask.sum() >= 10:
                webcam_model = GradientBoostingRegressor(**gbr_params)
                webcam_model.fit(X_train[webcam_mask], y_train[webcam_mask])
                y_pred_webcam = webcam_model.predict(X_test)
                # Blend: 50% global + 50% webcam-specific
                y_pred_raw = 0.5 * y_pred_global + 0.5 * y_pred_webcam
            else:
                y_pred_raw = y_pred_global
        else:
            y_pred_raw = y_pred_global

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
    print("Global Model v10: Robust GBR + hybrid approaches", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading features...", flush=True)
    feat_df = load_features_from_csv_paths(ALL_CSV_PATHS, verbose=True)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=True)

    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]
    seg_df = aggregate_segments(feat_df, raw_features, segment_length=SEG_LEN)
    x_cols_base = [f"{f}{s}" for f in raw_features for s in AGG_SUFFIXES
                   if f"{f}{s}" in seg_df.columns]

    # Add interaction features
    seg_df = add_interaction_features(seg_df, raw_features)
    interaction_cols = [c for c in seg_df.columns
                       if c.startswith("RoR_product") or c.startswith("RG_ratio")]
    x_cols_ext = x_cols_base + interaction_cols

    print(f"  {len(seg_df)} segments, {seg_df['subject_id'].nunique()} subjects", flush=True)
    print(f"  {len(x_cols_base)} base features, {len(x_cols_ext)} extended features", flush=True)

    # Exp 1: Huber loss GBR
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Devnorm + Huber loss GBR", flush=True)
    print(f"{'='*60}", flush=True)
    res1 = _loso(seg_df, x_cols_base, gbr_params=GBR_HUBER)
    report("huber_gbr", res1)

    # Exp 2: Huber + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: Devnorm + Huber GBR + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res2 = _loso(seg_df, x_cols_base, gbr_params=GBR_HUBER, use_offset=True)
    report("huber_offset", res2)

    # Exp 3: Two-model blend (global + webcam-specific)
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: Devnorm + webcam blend", flush=True)
    print(f"{'='*60}", flush=True)
    res3 = _loso(seg_df, x_cols_base, blend_webcam=True)
    report("webcam_blend", res3)

    # Exp 4: Two-model blend + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 4: Devnorm + webcam blend + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res4 = _loso(seg_df, x_cols_base, blend_webcam=True, use_offset=True)
    report("webcam_blend_offset", res4)

    # Exp 5: Extended features + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 5: Devnorm + interaction features + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res5 = _loso(seg_df, x_cols_ext, use_offset=True)
    report("interaction_offset", res5)

    # Exp 6: Huber + webcam blend + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 6: Huber + webcam blend + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res6 = _loso(seg_df, x_cols_base, gbr_params=GBR_HUBER,
                 blend_webcam=True, use_offset=True)
    report("huber_blend_offset", res6)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
