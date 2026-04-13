"""
Diagnose why c930 has PCC=0.71 but R2≈0.
Print per-subject prediction stats to understand the failure mode.
"""
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


def main():
    print("Loading...", flush=True)
    feat_df = load_features_from_csv_paths(ALL_CSV_PATHS, verbose=False)
    feat_df = filter_feat_df_by_spo2_range(feat_df, segment_length=SEGMENT_LENGTH, verbose=False)

    raw_features = [c for c in ALL_RAW_FEATURES if c in feat_df.columns]
    seg_df = aggregate_segments(feat_df, raw_features, segment_length=SEG_LEN)
    x_cols = [f"{f}{s}" for f in raw_features for s in AGG_SUFFIXES
              if f"{f}{s}" in seg_df.columns]

    device_all = seg_df["device"].values
    X_all = seg_df[x_cols].values
    y_all = seg_df["SpO2_mean"].values
    sid_all = seg_df["subject_id"].values

    # Only test c930 subjects
    c930_subjects = [s for s in seg_df["subject_id"].unique() if s.startswith("prc2-c930_")]
    print(f"\n{len(c930_subjects)} c930 subjects to analyze\n", flush=True)

    print(f"{'Subject':<25} {'n':>3} {'y_range':>7} {'y_mean':>6} {'p_mean':>6} "
          f"{'bias':>6} {'p_range':>7} {'R2':>7} {'PCC':>6}", flush=True)
    print("-" * 95, flush=True)

    all_y = []
    all_p = []

    for test_sid in c930_subjects:
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

        tv = np.isfinite(X_train).all(axis=1)
        X_train, y_train = X_train[tv], y_train[tv]
        tv = np.isfinite(X_test).all(axis=1)
        X_test, y_test = X_test[tv], y_test[tv]

        if len(X_test) < 2 or len(X_train) < 10:
            continue
        if np.ptp(y_test) < MIN_SpO2_RANGE:
            print(f"{test_sid:<25} {len(X_test):>3} {np.ptp(y_test):>7.1f}  SKIPPED (range<{MIN_SpO2_RANGE})", flush=True)
            continue

        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        bias = np.mean(y_pred) - np.mean(y_test)
        r2 = r2_score(y_test, y_pred)
        pcc = pearsonr(y_test, y_pred)[0] if np.std(y_pred) > 1e-8 else np.nan

        all_y.extend(y_test.tolist())
        all_p.extend(y_pred.tolist())

        print(f"{test_sid:<25} {len(y_test):>3} {np.ptp(y_test):>7.1f} {np.mean(y_test):>6.1f} "
              f"{np.mean(y_pred):>6.1f} {bias:>+6.1f} {np.ptp(y_pred):>7.1f} "
              f"{r2:>7.3f} {pcc:>6.3f}", flush=True)

    all_y = np.array(all_y)
    all_p = np.array(all_p)
    print(f"\n{'OVERALL':<25} {len(all_y):>3} {np.ptp(all_y):>7.1f} {np.mean(all_y):>6.1f} "
          f"{np.mean(all_p):>6.1f} {np.mean(all_p)-np.mean(all_y):>+6.1f} {np.ptp(all_p):>7.1f} "
          f"{r2_score(all_y, all_p):>7.3f} {pearsonr(all_y, all_p)[0]:>6.3f}", flush=True)

    # Also check: what's the variance of bias across subjects?
    print(f"\n--- Diagnosis ---", flush=True)
    print(f"Mean prediction: {np.mean(all_p):.2f}, Mean truth: {np.mean(all_y):.2f}", flush=True)
    print(f"Pred std: {np.std(all_p):.2f}, Truth std: {np.std(all_y):.2f}", flush=True)
    print(f"Pred range: {np.ptp(all_p):.2f}, Truth range: {np.ptp(all_y):.2f}", flush=True)

    # Check: if we apply a global affine transform, what R2 would we get?
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(all_p.reshape(-1, 1), all_y)
    all_p_cal = lr.predict(all_p.reshape(-1, 1))
    print(f"\nAfter global affine (slope={lr.coef_[0]:.3f}, intercept={lr.intercept_:.2f}):", flush=True)
    print(f"  R2 = {r2_score(all_y, all_p_cal):.4f}", flush=True)

    # Compare with other datasets
    print(f"\n--- Comparison across devices ---", flush=True)
    for dev_name in ["prc-c920", "prc-i15", "prc-i15m", "prc2-c930", "prc2-i16", "prc2-i16m"]:
        dev_mask_seg = seg_df["device"] == dev_name
        dev_y = seg_df[dev_mask_seg]["SpO2_mean"].values
        if len(dev_y) > 0:
            print(f"  {dev_name}: n_segs={len(dev_y)}, SpO2 mean={np.mean(dev_y):.1f}, "
                  f"std={np.std(dev_y):.1f}, range={np.ptp(dev_y):.1f}", flush=True)


if __name__ == "__main__":
    main()
