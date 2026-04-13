"""
Global Model v8: Fix prediction range compression.

Root cause: GBR compresses predictions toward the mean. For c930,
truth std=6.5 but pred std=3.5 — predictions don't span enough range.

Fix: After predicting, rescale predictions to match the expected variance
for the target device, computed from training data.

Exp 1: Devnorm + variance rescaling (match pred std to device training std)
Exp 2: Devnorm + z-score rescaling (normalize pred, re-scale to device stats)
Exp 3: Devnorm + variance rescaling + median offset
Exp 4: Larger model (n=200, depth=4) for better range coverage
Exp 5: Devnorm + variance rescaling + larger model
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

GBR_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=5, random_state=42)

GBR_PARAMS_LARGE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
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


def _loso(seg_df, x_cols, y_col="SpO2_mean",
          rescale_mode="none", use_offset=False, gbr_params=None):
    """LOSO with device normalization and prediction rescaling.

    rescale_mode:
        "none" - no rescaling
        "variance" - scale predictions to match device training variance
        "zscore" - z-score normalize predictions, re-scale to device stats
    """
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

        model = GradientBoostingRegressor(**gbr_params)
        model.fit(X_train, y_train)
        y_pred_raw = model.predict(X_test)

        # Compute device-specific training stats for rescaling
        dev_mask = dev_train == test_dev
        if dev_mask.sum() >= 3:
            dev_y_mean = np.mean(y_train[dev_mask])
            dev_y_std = np.std(y_train[dev_mask])

            # Compute prediction stats on same-device training data
            y_train_pred = model.predict(X_train)
            dev_pred_mean = np.mean(y_train_pred[dev_mask])
            dev_pred_std = np.std(y_train_pred[dev_mask])
        else:
            dev_y_mean = np.mean(y_train)
            dev_y_std = np.std(y_train)
            y_train_pred = model.predict(X_train)
            dev_pred_mean = np.mean(y_train_pred)
            dev_pred_std = np.std(y_train_pred)

        if rescale_mode == "variance":
            # Scale predictions to match device training variance
            # pred_rescaled = (pred - pred_mean) * (true_std / pred_std) + true_mean
            if dev_pred_std > 0.1:
                scale = dev_y_std / dev_pred_std
                # Clamp scale to avoid wild amplification
                scale = np.clip(scale, 0.5, 3.0)
                y_pred = (y_pred_raw - dev_pred_mean) * scale + dev_y_mean
            else:
                y_pred = y_pred_raw

        elif rescale_mode == "zscore":
            # Z-score normalize predictions, then re-scale to device stats
            if dev_pred_std > 0.1:
                y_pred_z = (y_pred_raw - dev_pred_mean) / dev_pred_std
                y_pred = y_pred_z * dev_y_std + dev_y_mean
            else:
                y_pred = y_pred_raw

        else:
            y_pred = y_pred_raw

        # Optional median offset (on top of rescaling)
        if use_offset and dev_mask.sum() >= 3:
            if rescale_mode != "none":
                # Recompute offset after rescaling
                if rescale_mode == "variance":
                    y_train_pred_rescaled = (y_train_pred[dev_mask] - dev_pred_mean) * scale + dev_y_mean
                elif rescale_mode == "zscore":
                    y_train_pred_rescaled = ((y_train_pred[dev_mask] - dev_pred_mean) / dev_pred_std) * dev_y_std + dev_y_mean
                offset = np.median(y_train[dev_mask] - y_train_pred_rescaled)
            else:
                offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            y_pred = y_pred + offset

        y_pred = np.clip(y_pred, 70, 100)

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
    print("Global Model v8: Fix prediction range compression", flush=True)
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

    # Exp 1: Variance rescaling
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Devnorm + variance rescaling", flush=True)
    print(f"{'='*60}", flush=True)
    res1 = _loso(seg_df, x_cols, rescale_mode="variance")
    met1 = report("var_rescale", res1)

    # Exp 2: Z-score rescaling
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: Devnorm + z-score rescaling", flush=True)
    print(f"{'='*60}", flush=True)
    res2 = _loso(seg_df, x_cols, rescale_mode="zscore")
    met2 = report("zscore_rescale", res2)

    # Exp 3: Variance rescaling + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: Devnorm + variance rescaling + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res3 = _loso(seg_df, x_cols, rescale_mode="variance", use_offset=True)
    met3 = report("var_rescale_offset", res3)

    # Exp 4: Larger model
    print(f"\n{'='*60}", flush=True)
    print("Exp 4: Devnorm + larger model (n=200, depth=4)", flush=True)
    print(f"{'='*60}", flush=True)
    res4 = _loso(seg_df, x_cols, gbr_params=GBR_PARAMS_LARGE)
    met4 = report("larger_model", res4)

    # Exp 5: Larger model + variance rescaling
    print(f"\n{'='*60}", flush=True)
    print("Exp 5: Devnorm + larger model + variance rescaling", flush=True)
    print(f"{'='*60}", flush=True)
    res5 = _loso(seg_df, x_cols, rescale_mode="variance", gbr_params=GBR_PARAMS_LARGE)
    met5 = report("larger_var_rescale", res5)

    # Exp 6: Z-score rescaling + offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 6: Devnorm + z-score rescaling + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res6 = _loso(seg_df, x_cols, rescale_mode="zscore", use_offset=True)
    met6 = report("zscore_offset", res6)

    if any([met1, met2, met3, met4, met5, met6]):
        print("\n  !!! TARGET MET !!!", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
