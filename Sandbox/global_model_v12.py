"""
Global Model v12: Fix c930 catastrophic subjects.

Best so far: huber_large+offset passes 5/6, only c930=-0.014 fails.
c930 median R2=0.127, but S010 (R2=-2.6) and S082 (R2=-4.4) destroy the mean.
Both have extreme mean SpO2 (96.5 and 83.4) vs device avg 91.7.
The model predicts them toward ~92, creating massive bias.

Strategy: Detect when predictions are unreliable and apply correction.

Exp 1: Huber large + offset + per-subject prediction spread check
        (if pred range << expected, blend with KNN)
Exp 2: Huber large + offset + ensemble with RF (diversity reduces variance)
Exp 3: Huber large + two-stage offset (global offset then device offset)
Exp 4: Huber large + offset + prediction confidence weighting
Exp 5: Huber large + offset + GBR trained on absolute residuals for weighting
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
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

GBR_HUBER_LARGE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                       subsample=0.8, min_samples_leaf=5, random_state=42,
                       loss="huber", alpha=0.9)

RF_PARAMS = dict(n_estimators=200, max_depth=6, min_samples_leaf=5,
                 random_state=42, n_jobs=-1)


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


def _loso_v12(seg_df, x_cols, y_col="SpO2_mean", mode="base"):
    """LOSO with Huber large + offset as base, plus experimental fixes.

    Modes:
        base: huber_large + offset (baseline from v11)
        knn_blend: blend with KNN when pred range is compressed
        ensemble: average GBR + RF predictions
        two_stage_offset: global offset, then device offset
        confidence: weight predictions by training residual similarity
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

        dev_mask = dev_train == test_dev

        if mode == "base":
            model = GradientBoostingRegressor(**GBR_HUBER_LARGE)
            model.fit(X_train, y_train)
            y_pred_raw = model.predict(X_test)
            # Offset
            y_train_pred = model.predict(X_train)
            if dev_mask.sum() >= 3:
                offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            else:
                offset = 0.0
            y_pred = np.clip(y_pred_raw + offset, 70, 100)

        elif mode == "knn_blend":
            # GBR prediction
            model = GradientBoostingRegressor(**GBR_HUBER_LARGE)
            model.fit(X_train, y_train)
            y_pred_gbr = model.predict(X_test)

            # KNN prediction (same-device neighbors)
            if dev_mask.sum() >= 5:
                k = min(15, dev_mask.sum())
                knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
                knn.fit(X_train[dev_mask], y_train[dev_mask])
                y_pred_knn = knn.predict(X_test)

                # Adaptive blending: if GBR pred range is compressed vs KNN,
                # give more weight to KNN
                gbr_range = np.ptp(y_pred_gbr)
                knn_range = np.ptp(y_pred_knn)
                if gbr_range > 0.1:
                    # If KNN has broader range, it may be more reliable for extremes
                    knn_weight = min(0.5, max(0.2, knn_range / (gbr_range + knn_range)))
                else:
                    knn_weight = 0.5
                y_pred_raw = (1 - knn_weight) * y_pred_gbr + knn_weight * y_pred_knn
            else:
                y_pred_raw = y_pred_gbr

            # Offset using GBR predictions
            y_train_pred = model.predict(X_train)
            if dev_mask.sum() >= 3:
                offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            else:
                offset = 0.0
            y_pred = np.clip(y_pred_raw + offset, 70, 100)

        elif mode == "ensemble":
            # GBR + RF ensemble
            gbr = GradientBoostingRegressor(**GBR_HUBER_LARGE)
            gbr.fit(X_train, y_train)
            y_pred_gbr = gbr.predict(X_test)

            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            # Average
            y_pred_raw = 0.5 * y_pred_gbr + 0.5 * y_pred_rf

            # Offset using ensemble predictions on training
            y_train_pred = 0.5 * gbr.predict(X_train) + 0.5 * rf.predict(X_train)
            if dev_mask.sum() >= 3:
                offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            else:
                offset = 0.0
            y_pred = np.clip(y_pred_raw + offset, 70, 100)

        elif mode == "two_stage_offset":
            model = GradientBoostingRegressor(**GBR_HUBER_LARGE)
            model.fit(X_train, y_train)
            y_pred_raw = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            # Stage 1: Global offset (all training data)
            global_offset = np.median(y_train - y_train_pred)
            y_pred_raw = y_pred_raw + global_offset
            y_train_pred = y_train_pred + global_offset

            # Stage 2: Device-specific residual offset
            if dev_mask.sum() >= 3:
                dev_offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            else:
                dev_offset = 0.0
            y_pred = np.clip(y_pred_raw + dev_offset, 70, 100)

        elif mode == "confidence":
            model = GradientBoostingRegressor(**GBR_HUBER_LARGE)
            model.fit(X_train, y_train)
            y_pred_raw = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            # Offset
            if dev_mask.sum() >= 3:
                offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
            else:
                offset = 0.0
            y_pred_raw = y_pred_raw + offset

            # Confidence: compute training residual for nearest neighbors
            # If nearby training points have high residual, shrink prediction
            # toward device mean
            if dev_mask.sum() >= 5:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(10, dev_mask.sum()))
                nn.fit(X_train[dev_mask])
                dists, idxs = nn.kneighbors(X_test)

                dev_residuals = np.abs(y_train[dev_mask] - y_train_pred[dev_mask])
                dev_y_mean = np.mean(y_train[dev_mask])

                for j in range(len(X_test)):
                    # Mean absolute residual of nearest neighbors
                    neighbor_residual = np.mean(dev_residuals[idxs[j]])
                    # If neighbors have high residual, shrink toward device mean
                    # confidence in [0, 1]: 1=trust model, 0=use device mean
                    confidence = np.clip(1.0 - neighbor_residual / 10.0, 0.3, 1.0)
                    y_pred_raw[j] = confidence * y_pred_raw[j] + (1 - confidence) * dev_y_mean

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
    print("Global Model v12: Fix c930 catastrophic subjects", flush=True)
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

    # Exp 1: Base (huber_large + offset) - reproduce v11 best
    print(f"\n{'='*60}", flush=True)
    print("Exp 1: Base (Huber large + offset)", flush=True)
    print(f"{'='*60}", flush=True)
    res1 = _loso_v12(seg_df, x_cols, mode="base")
    report("base", res1)

    # Exp 2: KNN blend (adaptive weight)
    print(f"\n{'='*60}", flush=True)
    print("Exp 2: KNN blend (adaptive)", flush=True)
    print(f"{'='*60}", flush=True)
    res2 = _loso_v12(seg_df, x_cols, mode="knn_blend")
    report("knn_blend", res2)

    # Exp 3: GBR + RF ensemble
    print(f"\n{'='*60}", flush=True)
    print("Exp 3: GBR + RF ensemble + offset", flush=True)
    print(f"{'='*60}", flush=True)
    res3 = _loso_v12(seg_df, x_cols, mode="ensemble")
    report("ensemble", res3)

    # Exp 4: Two-stage offset
    print(f"\n{'='*60}", flush=True)
    print("Exp 4: Two-stage offset (global + device)", flush=True)
    print(f"{'='*60}", flush=True)
    res4 = _loso_v12(seg_df, x_cols, mode="two_stage_offset")
    report("two_stage", res4)

    # Exp 5: Confidence-weighted prediction
    print(f"\n{'='*60}", flush=True)
    print("Exp 5: Confidence-weighted (shrink uncertain preds to device mean)", flush=True)
    print(f"{'='*60}", flush=True)
    res5 = _loso_v12(seg_df, x_cols, mode="confidence")
    report("confidence", res5)

    print(f"\n{'='*60}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
