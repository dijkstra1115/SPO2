#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPO2 Low/Normal/High classifier quick probe
- Features: six-channel color transforms + CCT (per-frame)
- Models: Logistic Regression (with standardization), RandomForest
- Input: 1+ CSV files, each with columns COLOR_R, COLOR_G, COLOR_B, SPO2
- Output: per-CSV metrics CSV + confusion matrices as PNGs
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Utils ----------------

def log(msg: str):
    print(msg, flush=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_csv_chunked(path: str, usecols=None, chunksize=100_000, target_n: int = None, random_state: int = 42) -> pd.DataFrame:
    """Chunked loader with optional reservoir-style sampling to a target size."""
    if usecols is None:
        usecols = ["COLOR_R", "COLOR_G", "COLOR_B", "SPO2"]
    assert os.path.exists(path), f"File not found: {path}"
    dfs = []
    total = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=usecols)
        if len(chunk) == 0:
            continue
        if target_n is None:
            dfs.append(chunk)
        else:
            # sample a limited amount per chunk to avoid huge memory
            need = max(1, min(len(chunk), max(10_000, target_n // 10)))
            dfs.append(chunk.sample(n=need, random_state=random_state))
            total += need
            if total >= target_n:
                break
    if not dfs:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if target_n is not None and len(df) > target_n:
        df = df.sample(n=target_n, random_state=random_state)
    return df

def build_features_from_rgb(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return X (N,7) features and SPO2 array (N,)"""
    R = df["COLOR_R"].astype(np.float32).values
    G = df["COLOR_G"].astype(np.float32).values
    B = df["COLOR_B"].astype(np.float32).values
    rgb = np.vstack([R, G, B])

    # Six-channel transforms
    ycgcr_cg = 128.0 -  81.085 * rgb[0, :] / 255.0 + 112.000 * rgb[1, :] / 255.0 - 30.915 * rgb[2, :] / 255.0
    ycgcr_cr = 128.0 + 112.000 * rgb[0, :] / 255.0 -  93.786 * rgb[1, :] / 255.0 - 18.214 * rgb[2, :] / 255.0
    yiq_i    =  0.595716 * rgb[0, :] - 0.274453 * rgb[1, :] - 0.321263 * rgb[2, :]
    ydbdr_dr = -1.333 * rgb[0, :] + 1.116 * rgb[1, :] + 0.217 * rgb[2, :]
    pos_y    = -2*rgb[0, :] + rgb[1, :] + rgb[2, :]
    chrom_x  =   3*rgb[0, :] - 2*rgb[1, :]

    # CCT (McCamy)
    eps = 1e-8
    R01, G01, B01 = rgb / 255.0
    X = 0.4124*R01 + 0.3576*G01 + 0.1805*B01
    Y = 0.2126*R01 + 0.7152*G01 + 0.0722*B01
    Z = 0.0193*R01 + 0.1192*G01 + 0.9505*B01
    den = np.maximum(X + Y + Z, eps)
    x = X / den
    y = Y / den
    n = (x - 0.3320) / np.maximum(0.1858 - y, eps)
    CCT = 437*(n**3) + 3601*(n**2) + 6861*n + 5517

    Xfeat = np.vstack([ycgcr_cg, ycgcr_cr, yiq_i, ydbdr_dr, pos_y, chrom_x, CCT]).T
    spo2 = df["SPO2"].astype(np.float32).values
    return Xfeat.astype(np.float32), spo2

def normalize_features_per_csv(X: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    對單一 CSV 的特徵做 feature-wise normalization。
    mode="zscore": (X - mean) / std ；"none": 原樣返回
    """
    if mode == "zscore":
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mu) / std
    return X

def label_spo2(spo2: np.ndarray, low_thr: float = 85.0, high_thr: float = 95.0) -> np.ndarray:
    labels = np.full(len(spo2), "Normal", dtype=object)
    labels[spo2 < low_thr] = "Low"
    labels[spo2 > high_thr] = "High"
    return labels

def fit_and_eval_models_cross_csv(all_data: Dict[str, Tuple[np.ndarray, np.ndarray]], test_csv: str, seed: int) -> Dict[str, Dict]:
    """Cross-CSV evaluation: train on all CSVs except test_csv, test on test_csv"""
    
    # Prepare training data (all CSVs except test_csv)
    X_train_list = []
    y_train_list = []
    
    for csv_name, (X, y) in all_data.items():
        if csv_name != test_csv:
            X_train_list.append(X)
            y_train_list.append(y)
    
    if not X_train_list:
        raise ValueError(f"No training data available for testing {test_csv}")
    
    # Combine training data
    X_tr = np.vstack(X_train_list)
    y_tr = np.hstack(y_train_list)
    
    # Get test data
    X_te, y_te = all_data[test_csv]
    
    # RandomForest only
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    rep_rf = classification_report(y_te, pred_rf, output_dict=True, zero_division=0)
    cm_rf = confusion_matrix(y_te, pred_rf, labels=["Low","Normal","High"])

    return {
        "RandomForest": {"report": rep_rf, "cm": cm_rf, "y_true": y_te, "y_pred": pred_rf},
    }

def cm_to_png(cm: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(4.2, 4.0))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1,2], ["Low","Normal","High"])
    plt.yticks([0,1,2], ["Low","Normal","High"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def extract_summary(report: Dict) -> Tuple[float, float, float, Dict[str, float]]:
    acc = report.get("accuracy", float("nan"))
    # macro average f1 across Low/Normal/High
    f1s = []
    per_class_f1 = {}
    for cls in ["Low","Normal","High"]:
        f1 = report.get(cls, {}).get("f1-score", float("nan"))
        per_class_f1[cls] = f1
        if not math.isnan(f1):
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return acc, macro_f1, report.get("macro avg", {}).get("f1-score", float("nan")), per_class_f1

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="One or more CSV files")
    ap.add_argument("--outdir", type=str, default="clf_results", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=100_000, help="CSV read chunksize")
    ap.add_argument("--target_n", type=int, default=200_000, help="Max rows to sample per CSV (None = all)")
    ap.add_argument("--low_thr", type=float, default=85.0, help="Low threshold for SPO2 class")
    ap.add_argument("--high_thr", type=float, default=95.0, help="High threshold for SPO2 class")
    ap.add_argument("--per_csv_norm", type=str, default="zscore",
                    choices=["none", "zscore"],
                    help="Per-CSV feature normalization before training/eval")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    rows = []

    # First, load all CSV files and prepare data
    log(f"\n[LOAD] Loading all CSV files...")
    all_data = {}
    
    for csv_path in args.csvs:
        tag = os.path.splitext(os.path.basename(csv_path))[0]
        log(f"[LOAD] {csv_path}")
        df = load_csv_chunked(csv_path, chunksize=args.chunksize, target_n=args.target_n, random_state=args.seed)
        if df.empty:
            log(f"[WARN] Empty or invalid data: {csv_path}")
            continue

        # Build features + labels
        X, spo2 = build_features_from_rgb(df)

        # ★ Per-CSV normalization：對「這個 CSV 的特徵」先做 z-score
        X = normalize_features_per_csv(X, mode=args.per_csv_norm)

        y = label_spo2(spo2, args.low_thr, args.high_thr)

        # Guard: need at least two classes and enough samples
        uniq, counts = np.unique(y, return_counts=True)
        log(f"[INFO] {tag} class distribution: {dict(zip(uniq, counts))}")
        if len(uniq) < 2:
            log(f"[SKIP] {tag} needs at least 2 classes. Got: {dict(zip(uniq, counts))}")
            continue
            
        all_data[tag] = (X, y)

    if len(all_data) < 2:
        log(f"[ERROR] Need at least 2 valid CSV files for cross-CSV evaluation. Got: {len(all_data)}")
        return

    log(f"\n[EVAL] Starting cross-CSV evaluation with {len(all_data)} files...")
    
    # Cross-CSV evaluation: test each CSV using others for training
    for test_csv in all_data.keys():
        log(f"\n[TEST] Testing on {test_csv}, training on {[k for k in all_data.keys() if k != test_csv]}")
        
        try:
            # Fit models using cross-CSV approach
            res = fit_and_eval_models_cross_csv(all_data, test_csv, seed=args.seed)

            # Save confusion matrices and assemble summary
            for model_name, payload in res.items():
                report = payload["report"]
                cm = payload["cm"]
                acc, macro_f1, macro_f1_sklearn, per_cls = extract_summary(report)

                cm_path = os.path.join(args.outdir, f"{test_csv}_{model_name}_cross_csv_cm.png")
                cm_to_png(cm, f"Cross-CSV Confusion Matrix - {model_name} (Test: {test_csv})", cm_path)

                row = {
                    "test_csv": test_csv,
                    "model": model_name,
                    "accuracy": acc,
                    "macro_f1(mean3)": macro_f1,
                    "macro_f1(sklearn)": macro_f1_sklearn,
                    "f1_low": per_cls.get("Low", float("nan")),
                    "f1_normal": per_cls.get("Normal", float("nan")),
                    "f1_high": per_cls.get("High", float("nan")),
                    "cm_path": cm_path,
                }
                rows.append(row)
                log(f"[RESULT] {test_csv} | {model_name} | Acc={acc:.4f} | MacroF1={macro_f1:.4f} | per-class F1: {per_cls}")
                
        except Exception as e:
            log(f"[ERROR] Failed to evaluate {test_csv}: {e}")
            continue

    if rows:
        out_csv = os.path.join(args.outdir, "summary_cross_csv.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log(f"\nSaved cross-CSV summary: {out_csv}")
    else:
        log("No results to save.")

if __name__ == "__main__":
    main()
