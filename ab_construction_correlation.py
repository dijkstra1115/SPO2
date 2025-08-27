# -*- coding: utf-8 -*-
"""
ab_construction_correction.py

放置位置：與 Data 同層
用法：python ab_construction_correction.py

功能：
1) 讀取 ./Data/CCT_Output/CCT_xxxx/data.csv
2) 對每個 CCT 和每個通道：
   - 以 ch 與 gt 為該 CCT 的 12 位受試者做 y = a*x + b
   - 輸出 ./model_para/CCT_xxxx/ch_{channel_name}.csv (含每位受試者的 a,b)
3) Construction：
   - 在每個 CCT 內用 (a,b) 回歸 b ≈ s·a + i，並記錄該 CCT 的 min(a), max(a)
   - 彙整所有 CCT 的 s,i，對 CCT 做一次線性擬合 → slope_trend, intercept_trend
   - 輸出 ./model_line/model_line_{channel_name}.csv
4) Correction：
   - 以每筆受試者片段的 (x_min,x_max) 與 (y_min,y_max) 反推 a 範圍
   - 更新各 CCT 的 min(a), max(a)
   - 輸出 ./model_line/corr_model_line_{channel_name}.csv

假設：
- 每個 data.csv 有欄位：ch（通道值）、gt（SpO₂）
- 受試者欄位若無，則假設每 1800 列為一人
- CCT 資料夾命名為 CCT_3200, CCT_3500, ...，內含 data.csv
"""

import os
import re
import math
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

# ====== 使用者可調參數 ======
# 重要：在這裡設定你要處理的通道名稱
# 程式會自動檢查哪些通道在你的資料中存在，並只處理存在的通道
CHANNEL_COLS = [
    "cg",      # 藍色通道
    "cr",      # 色相通道
    "iq",      # 飽和度通道
    "dr",      # 明度通道
    "pos",      # Y通道
    "chrom"      # Cr通道
]

# 其他參數
DATA_ROOT = "./Data/CCT_Output_holding"
GT_COL = "gt"        # SpO2 欄位
SUBJECT_COL_CANDIDATES = ["Folder"]  # 嘗試自動偵測的受試者欄名
ROWS_PER_SUBJECT = 1800  # 若無受試者欄，將每 1800 列視為一位受試者
OUTPUT_PARA_DIR = "./model_para_holding"   # 輸出每個 CCT 的 a,b
OUTPUT_LINE_DIR = "./model_line_holding"   # 輸出 model_line 與 correction 結果
SAVE_INTERMEDIATE = True           # 是否輸出每個 CCT 的 ch.csv (a,b)
# ==========================

@dataclass
class AB:
    a: float
    b: float
    sub: str

@dataclass
class SI:
    s: float
    i: float
    a_min: float
    a_max: float
    cct: int

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_cct_from_folder(folder_name: str) -> Optional[int]:
    # 支援 "CCT_5600" 或 "5600" 類型
    m = re.search(r"(\d{3,5})", os.path.basename(folder_name))
    if m:
        return int(m.group(1))
    return None

def detect_subject_column(df: pd.DataFrame) -> Optional[str]:
    for col in SUBJECT_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None

def split_into_subjects(df: pd.DataFrame, subject_col: Optional[str]) -> List[Tuple[str, pd.DataFrame]]:
    """
    回傳 [(subject_id, df_subject), ...]
    若無 subject_col，則以 ROWS_PER_SUBJECT 分割，主鍵為 sub_000, sub_001, ...
    """
    if subject_col:
        groups = []
        for sid, g in df.groupby(subject_col):
            groups.append((str(sid), g.reset_index(drop=True)))
        return groups
    else:
        n = len(df)
        if n % ROWS_PER_SUBJECT != 0:
            # 仍然硬切，最後一段可能不足 1800
            pass
        groups = []
        num_sub = math.ceil(n / ROWS_PER_SUBJECT)
        for k in range(num_sub):
            start = k * ROWS_PER_SUBJECT
            end = min((k + 1) * ROWS_PER_SUBJECT, n)
            g = df.iloc[start:end].reset_index(drop=True)
            groups.append((f"sub_{k:03d}", g))
        return groups

def fit_line_xy(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    以 numpy polyfit 擬合 y = a*x + b
    回傳 (a,b)
    """
    if len(x) < 2:
        return np.nan, np.nan
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)

def fit_b_on_a(a_vals: np.ndarray, b_vals: np.ndarray) -> Tuple[float, float]:
    """
    擬合 b = s*a + i
    回傳 (s, i)
    """
    if len(a_vals) < 2:
        return np.nan, np.nan
    s, i = np.polyfit(a_vals, b_vals, deg=1)
    return float(s), float(i)

def compute_a_range_from_ranges(x_range: Tuple[float,float],
                                y_range: Tuple[float,float],
                                s: float, i: float) -> Tuple[float,float]:
    """
    a = (y - i) / (x + s)，取四個角 (x,y) = (xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)
    避免除零
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    cand = []
    for x in (x_min, x_max):
        denom = x + s
        if abs(denom) < 1e-12:
            continue
        for y in (y_min, y_max):
            cand.append((y - i) / denom)
    if not cand:
        return np.nan, np.nan
    return float(np.nanmin(cand)), float(np.nanmax(cand))

def compute_a_range_aligned(x: np.ndarray, y: np.ndarray, s: float, i: float) -> Tuple[float, float]:
    """
    對齊每個時間點 t，計算 a_t = (y_t - i) / (x_t + s)，取全片段的 min/max。
    會自動忽略 NaN 與接近除零的樣本。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = x + s
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    if not np.any(valid):
        return np.nan, np.nan
    a_vals = (y[valid] - i) / denom[valid]
    return float(np.nanmin(a_vals)), float(np.nanmax(a_vals))

def compute_a_range_aligned_robust(
    x: np.ndarray, y: np.ndarray, s: float, i: float,
    denom_thresh: float = 1e-3,  # 依你的 ch 尺度調整
    q_low: float = 0.01, q_high: float = 0.99
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = x + s
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(denom) & (np.abs(denom) > denom_thresh)
    if not np.any(valid):
        return np.nan, np.nan
    a_vals = (y[valid] - i) / denom[valid]

    # 可選：再做一次 sigma-clipping（排除極端離群）
    # med = np.nanmedian(a_vals)
    # mad = np.nanmedian(np.abs(a_vals - med)) + 1e-12
    # valid2 = np.abs(a_vals - med) < 6.0 * 1.4826 * mad  # 6*MAD
    # a_vals = a_vals[valid2]

    a_min = float(np.nanquantile(a_vals, q_low))
    a_max = float(np.nanquantile(a_vals, q_high))
    return a_min, a_max

def test_with_a_range(x: np.ndarray, y: np.ndarray, s: float, i: float,
                      a_min: float, a_max: float) -> Tuple[float, float, float, float]:
    """
    以 a 的區間 [a_min, a_max] 評估預測：
      ŷ_min = a_min * (x + s) + i
      ŷ_max = a_max * (x + s) + i
      ŷ_mid = ((a_min+a_max)/2) * (x + s) + i

    回傳：
      coverage:  y 落在 [ŷ_min, ŷ_max] 的比例
      mae:       使用 a_mid 的 MAE
      rmse:      使用 a_mid 的 RMSE
      mean_band: 區間平均寬度的 L1（對每個樣本的 |ŷ_max - ŷ_min| 取平均）
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = float(s); i = float(i); a_min = float(a_min); a_max = float(a_max)

    y_pred_min = a_min * (x + s) + i
    y_pred_max = a_max * (x + s) + i

    low = np.minimum(y_pred_min, y_pred_max)
    high = np.maximum(y_pred_min, y_pred_max)
    inside = (y >= low) & (y <= high)
    coverage = float(np.mean(inside)) if inside.size > 0 else np.nan

    a_mid = 0.5 * (a_min + a_max)
    y_pred_mid = a_mid * (x + s) + i
    mae = float(np.mean(np.abs(y - y_pred_mid))) if y.size > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - y_pred_mid) ** 2))) if y.size > 0 else np.nan

    mean_band = float(np.mean(np.abs(y_pred_max - y_pred_min))) if y.size > 0 else np.nan
    return coverage, mae, rmse, mean_band

def evaluate_with_corr(corr_df: pd.DataFrame,
                       cct_folders: List[str],
                       channel_col: str,  # 添加通道参数
                       gt_col: str,
                       output_dir: str,
                       detect_subject_column_fn,
                       split_into_subjects_fn):
    """
    使用 correction 後的每個 CCT 之 (s, i, min(a), max(a))，對所有 data.csv 做評估。
    會輸出兩份報表：
      1) per-subject：./model_line/eval_with_a_range_subject_{channel}.csv
      2) per-CCT    ：./model_line/eval_with_a_range_cct_{channel}.csv
      3) overall    ：./model_line/eval_with_a_range_overall_{channel}.txt（簡單摘要）

    欄位說明：
      coverage:   y 落在 [ŷ_min, ŷ_max] 的比例
      mae / rmse: 使用 a_mid 的誤差
      mean_band:  平均預測區間寬度（越小越好，代表區間不寬）
    """
    # 準備 CCT -> (s, i, a_min, a_max)
    need_cols = {"CCT", "slope", "intercept", "min(a)", "max(a)"}
    if not need_cols.issubset(set(corr_df.columns)):
        raise KeyError(f"corr_df 需包含欄位：{need_cols}")

    corr_map = {}
    for _, r in corr_df.iterrows():
        cct = int(r["CCT"])
        corr_map[cct] = (float(r["slope"]), float(r["intercept"]),
                         float(r["min(a)"]), float(r["max(a)"]))

    # 蒐集逐 subject 的指標
    subject_rows = []

    for folder in cct_folders:
        cct = parse_cct_from_folder(folder)
        if cct is None or cct not in corr_map:
            continue
        csv_path = os.path.join(folder, "data.csv")
        if not os.path.exists(csv_path):
            continue

        s, i, a_min, a_max = corr_map[cct]

        df = pd.read_csv(csv_path)
        subj_col = detect_subject_column_fn(df)
        groups = split_into_subjects_fn(df, subj_col)

        for sid, g in groups:
            x = g[channel_col].to_numpy(dtype=float)
            y = g[gt_col].to_numpy(dtype=float)

            coverage, mae, rmse, mean_band = test_with_a_range(x, y, s, i, a_min, a_max)

            subject_rows.append({
                "CCT": cct,
                "subject": sid,
                "n": int(len(g)),
                "coverage": coverage,
                "mae": mae,
                "rmse": rmse,
                "mean_band": mean_band,
                "slope": s,
                "intercept": i,
                "a_min": a_min,
                "a_max": a_max
            })

    if not subject_rows:
        print("[WARN] evaluate_with_corr: 沒有可用的 subject 評估資料。")
        return

    ensure_dir(output_dir)
    df_subj = pd.DataFrame(subject_rows).sort_values(["CCT", "subject"])
    
    # 修复：在文件名中添加通道识别
    subj_csv = os.path.join(output_dir, f"eval_with_a_range_subject_{channel_col}.csv")
    df_subj.to_csv(subj_csv, index=False)

    # 依 CCT 聚合
    agg_funcs = {
        "n": "sum",
        "coverage": "mean",
        "mae": "mean",
        "rmse": "mean",
        "mean_band": "mean"
    }
    df_cct = (df_subj.groupby("CCT", as_index=False)
              .agg(agg_funcs)
              .sort_values("CCT"))
    
    # 修复：在文件名中添加通道识别
    cct_csv = os.path.join(output_dir, f"eval_with_a_range_cct_{channel_col}.csv")
    df_cct.to_csv(cct_csv, index=False)

    # Overall 摘要
    overall = {
        "subjects": int(len(df_subj)),
        "total_samples": int(df_subj["n"].sum()),
        "coverage_mean": float(df_subj["coverage"].mean()),
        "mae_mean": float(df_subj["mae"].mean()),
        "rmse_mean": float(df_subj["rmse"].mean()),
        "mean_band_mean": float(df_subj["mean_band"].mean())
    }
    
    # 修复：在文件名中添加通道识别
    overall_txt = os.path.join(output_dir, f"eval_with_a_range_overall_{channel_col}.txt")
    with open(overall_txt, "w", encoding="utf-8") as f:
        for k, v in overall.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] Saved subject metrics: {subj_csv}")
    print(f"[OK] Saved CCT metrics    : {cct_csv}")
    print(f"[OK] Saved overall summary: {overall_txt}")

def plot_ab_lines_with_ranges(df, title="a-b lines per CCT", filename=None, resolution=400):
    """
    df 需包含欄位：
      - 'CCT' 或 'cct_midpoints'：CCT 數值
      - 'slope'：每個 CCT 對應的 s（建議用 trend 後或 correction 後的值）
      - 'intercept'：每個 CCT 對應的 i（同上）
      - 'min(a)'、'max(a)'：該 CCT 的 a 範圍
    作用：
      畫出每個 CCT 的 b = s*a + i 直線；並在各自的 a 範圍上加粗顯示。
    """
    # 兼容欄名
    cct_col = 'CCT' if 'CCT' in df.columns else 'cct_midpoints'
    req_cols = [cct_col, 'slope', 'intercept', 'min(a)', 'max(a)']
    for c in req_cols:
        if c not in df.columns:
            raise KeyError(f"plot_ab_lines_with_ranges: need column '{c}'")

    # 全域 a 範圍（便於把所有線畫在同一個 a 範圍上）
    a_min_global = np.nanmin(df['min(a)'].to_numpy(float))
    a_max_global = np.nanmax(df['max(a)'].to_numpy(float))
    a_vals = np.linspace(a_min_global, a_max_global, resolution)

    plt.figure(figsize=(10, 6))
    for _, r in df.sort_values(cct_col).iterrows():
        cct = int(r[cct_col])
        s = float(r['slope'])
        i = float(r['intercept'])
        a0, a1 = float(r['min(a)']), float(r['max(a)'])
        if not (np.isfinite(s) and np.isfinite(i) and np.isfinite(a0) and np.isfinite(a1)):
            continue

        # 全域範圍的細線
        b_full = s * a_vals + i
        (line_full,) = plt.plot(a_vals, b_full, alpha=0.5, label=f"CCT {cct}: b={s:.3g}·a+{i:.3g}")
        color = line_full.get_color()

        # 自身 a 範圍的加粗線段
        a_seg = np.linspace(a0, a1, max(2, resolution // 4))
        b_seg = s * a_seg + i
        plt.plot(a_seg, b_seg, linewidth=3, color=color, label="_nolegend_", zorder=5)

    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    # plt.show()

def plot_a_ranges_bars(df, title="a ranges per CCT", filename=None):
    """
    df 需包含欄位：
      - 'CCT' 或 'cct_midpoints'
      - 'min(a)', 'max(a)'
    作用：
      以橫條方式顯示每個 CCT 的 a 範圍。
    """
    cct_col = 'CCT' if 'CCT' in df.columns else 'cct_midpoints'
    for c in [cct_col, 'min(a)', 'max(a)']:
        if c not in df.columns:
            raise KeyError(f"plot_a_ranges_bars: need column '{c}'")

    dff = df[[cct_col, 'min(a)', 'max(a)']].dropna().sort_values(cct_col)
    labels = dff[cct_col].astype(int).astype(str).tolist()
    y = np.arange(len(labels))
    min_as = dff['min(a)'].to_numpy(float)
    max_as = dff['max(a)'].to_numpy(float)
    widths = max_as - min_as

    plt.figure(figsize=(7, 3.2))
    plt.barh(y, widths, left=min_as, height=0.45, edgecolor='black', alpha=0.15)
    plt.yticks(y, labels)
    plt.ylabel("CCT")
    plt.xlabel("a")
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    # plt.show()

def plot_cct_vs_slope_intercept(df, title="CCT vs Slope/Intercept", filename=None):
    """
    繪製 CCT 與 slope 和 intercept 的關係折線圖
    
    df 需包含欄位：
      - 'CCT' 或 'cct_midpoints'
      - 'slope', 'intercept'
      - 'slope_trend', 'intercept_trend' (可選，用於顯示趨勢線)
    
    作用：
      以折線圖方式顯示 slope 和 intercept 隨 CCT 的變化趨勢
      包含擬合線和相關係數
    """
    # 兼容欄名
    cct_col = 'CCT' if 'CCT' in df.columns else 'cct_midpoints'
    for c in [cct_col, 'slope', 'intercept']:
        if c not in df.columns:
            raise KeyError(f"plot_cct_vs_slope_intercept: need column '{c}'")
    
    # 過濾有效數據並排序
    dff = df[[cct_col, 'slope', 'intercept']].dropna().sort_values(cct_col)
    if dff.empty:
        print("[WARN] plot_cct_vs_slope_intercept: 沒有有效數據")
        return
    
    cct_vals = dff[cct_col].astype(float).to_numpy()
    slope_vals = dff['slope'].astype(float).to_numpy()
    intercept_vals = dff['intercept'].astype(float).to_numpy()
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左圖：CCT vs Slope
    ax1.plot(cct_vals, slope_vals, 'o-', linewidth=2, markersize=8, 
             color='blue', label='Actual slope')
    
    # 計算 slope 與 CCT 的擬合線和相關係數
    if len(cct_vals) >= 2:
        # 線性擬合
        slope_coeffs = np.polyfit(cct_vals, slope_vals, 1)
        slope_fit = np.poly1d(slope_coeffs)
        slope_fit_line = slope_fit(cct_vals)
        
        # Pearson 相關係數
        slope_corr = np.corrcoef(cct_vals, slope_vals)[0, 1]
        
        # 繪製擬合線
        ax1.plot(cct_vals, slope_fit_line, '--', linewidth=2, 
                 color='red', alpha=0.8, 
                 label=f'Fitted line: y={slope_coeffs[0]:.2e}x+{slope_coeffs[1]:.3f}')
        
        # 在圖上顯示相關係數
        ax1.text(0.05, 0.95, f'Pearson r = {slope_corr:.3f}', 
                 transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('CCT (K)', fontsize=12)
    ax1.set_ylabel('Slope', fontsize=12)
    ax1.set_title(f'CCT vs Slope - {title}', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=10)
    
    # 右圖：CCT vs Intercept
    ax2.plot(cct_vals, intercept_vals, 's-', linewidth=2, markersize=8, 
             color='green', label='Actual intercept')
    
    # 計算 intercept 與 CCT 的擬合線和相關係數
    if len(cct_vals) >= 2:
        # 線性擬合
        intercept_coeffs = np.polyfit(cct_vals, intercept_vals, 1)
        intercept_fit = np.poly1d(intercept_coeffs)
        intercept_fit_line = intercept_fit(cct_vals)
        
        # Pearson 相關係數
        intercept_corr = np.corrcoef(cct_vals, intercept_vals)[0, 1]
        
        # 繪製擬合線
        ax2.plot(cct_vals, intercept_fit_line, '--', linewidth=2, 
                 color='orange', alpha=0.8, 
                 label=f'Fitted line: y={intercept_coeffs[0]:.2e}x+{intercept_coeffs[1]:.3f}')
        
        # 在圖上顯示相關係數
        ax2.text(0.05, 0.95, f'Pearson r = {intercept_corr:.3f}', 
                 transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('CCT (K)', fontsize=12)
    ax2.set_ylabel('Intercept', fontsize=12)
    ax2.set_title(f'CCT vs Intercept - {title}', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    
    # plt.show()

def process_single_channel(channel_col: str, cct_folders: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    處理單一通道的完整流程
    回傳 (model_df, corr_df)
    """
    print(f"\n=== 處理通道: {channel_col} ===")
    
    # 1) 對每個 CCT 讀 data.csv，對每位受試者做 y=a*x+b
    all_cct_ab: Dict[int, List[AB]] = {}
    per_cct_minmax_x_y: Dict[int, List[Tuple[Tuple[float,float], Tuple[float,float]]]] = {}

    for folder in cct_folders:
        cct = parse_cct_from_folder(folder)
        if cct is None:
            print(f"[WARN] Skip folder (no CCT parsed): {folder}")
            continue
        csv_path = os.path.join(folder, "data.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing data.csv in {folder}, skip.")
            continue

        df = pd.read_csv(csv_path)
        # 基本欄位檢查
        if channel_col not in df.columns or GT_COL not in df.columns:
            print(f"[WARN] Channel '{channel_col}' or '{GT_COL}' not found in {csv_path}, skip.")
            continue

        subj_col = detect_subject_column(df)
        groups = split_into_subjects(df, subj_col)

        ab_list: List[AB] = []
        xy_ranges: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []

        for sid, g in groups:
            x = g[channel_col].to_numpy(dtype=float)
            y = g[GT_COL].to_numpy(dtype=float)
            a, b = fit_line_xy(x, y)
            if not (np.isfinite(a) and np.isfinite(b)):
                print(f"[WARN] subject {sid} at CCT {cct}: insufficient or invalid data; skip AB.")
                continue
            ab_list.append(AB(a=a, b=b, sub=sid))

            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
            xy_ranges.append(((x_min, x_max), (y_min, y_max)))

        if not ab_list:
            print(f"[WARN] CCT {cct}: no valid (a,b) computed; skip.")
            continue

        all_cct_ab[cct] = ab_list
        per_cct_minmax_x_y[cct] = xy_ranges

        # 輸出每個 CCT 的 a,b（等同於 model_para）
        if SAVE_INTERMEDIATE:
            out_dir = os.path.join(OUTPUT_PARA_DIR, f"CCT_{cct}")
            ensure_dir(out_dir)
            df_ab = pd.DataFrame({"key": [ab.sub for ab in ab_list],
                                  "cct_mean": [cct]*len(ab_list),
                                  "a": [ab.a for ab in ab_list],
                                  "b": [ab.b for ab in ab_list]})
            df_ab.to_csv(os.path.join(out_dir, f"ch_{channel_col}.csv"), index=False)

    if not all_cct_ab:
        print(f"[WARN] Channel {channel_col}: No AB data collected from any CCT folder.")
        return pd.DataFrame(), pd.DataFrame()

    # 2) 在每個 CCT 內做 b ≈ s·a + i，並記錄 a 範圍（construction）
    rows = []
    for cct, ab_list in sorted(all_cct_ab.items()):
        a_vals = np.array([ab.a for ab in ab_list], dtype=float)
        b_vals = np.array([ab.b for ab in ab_list], dtype=float)
        valid = np.isfinite(a_vals) & np.isfinite(b_vals)
        a_vals = a_vals[valid]
        b_vals = b_vals[valid]
        if len(a_vals) < 2:
            print(f"[WARN] CCT {cct}: <2 valid (a,b); cannot fit s,i.")
            s, i = np.nan, np.nan
            a_min, a_max = (np.nan, np.nan) if len(a_vals)==0 else (float(np.min(a_vals)), float(np.max(a_vals)))
        else:
            s, i = fit_b_on_a(a_vals, b_vals)
            a_min, a_max = float(np.min(a_vals)), float(np.max(a_vals))

        rows.append({"cct_midpoints": cct,
                     "slope": s,
                     "intercept": i,
                     "min(a)": a_min,
                     "max(a)": a_max})

    model_df = pd.DataFrame(rows).sort_values("cct_midpoints").reset_index(drop=True)

    # 3) 對 s 與 i 分別做隨 CCT 的線性趨勢擬合（slope_trend, intercept_trend）
    def safe_polyfit_x_y(x: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
        v = np.isfinite(x) & np.isfinite(y)
        xx, yy = x[v], y[v]
        if len(xx) < 2:
            return (np.nan, np.nan)
        k, b = np.polyfit(xx, yy, deg=1)
        return float(k), float(b)

    k_s, b_s = safe_polyfit_x_y(model_df["cct_midpoints"].to_numpy(float),
                                model_df["slope"].to_numpy(float))
    k_i, b_i = safe_polyfit_x_y(model_df["cct_midpoints"].to_numpy(float),
                                model_df["intercept"].to_numpy(float))

    if np.isfinite(k_s):
        model_df["slope_trend"] = model_df["cct_midpoints"]*k_s + b_s
    else:
        model_df["slope_trend"] = np.nan

    if np.isfinite(k_i):
        model_df["intercept_trend"] = model_df["cct_midpoints"]*k_i + b_i
    else:
        model_df["intercept_trend"] = np.nan

    # 4) Correction：用每個 CCT 的資料片段範圍反推 a 範圍，更新 min(a), max(a)
    #    先建一個查表：CCT -> (s_trend, i_trend)
    si_trend_by_cct: Dict[int, Tuple[float,float]] = {}
    for _, r in model_df.iterrows():
        cct = int(r["cct_midpoints"])
        si_trend_by_cct[cct] = (float(r["slope_trend"]), float(r["intercept_trend"]))

    # 以 construction 的 a 範圍為起點
    corr = {int(r["cct_midpoints"]): {"slope": float(r["slope_trend"]),
                                      "intercept": float(r["intercept_trend"]),
                                      "min(a)": float(r["min(a)"]),
                                      "max(a)": float(r["max(a)"])}
            for _, r in model_df.iterrows()}

    # 再讀一次資料，用各 CCT 的 (x_range, y_range) 反推 a_range 來擴充
    for folder in cct_folders:
        cct = parse_cct_from_folder(folder)
        if cct is None or cct not in si_trend_by_cct:
            continue
        csv_path = os.path.join(folder, "data.csv")
        if not os.path.exists(csv_path):
            continue
        s_tr, i_tr = si_trend_by_cct[cct]
        # 若趨勢無效，跳過
        if not (np.isfinite(s_tr) and np.isfinite(i_tr)):
            continue

        df = pd.read_csv(csv_path)
        if channel_col not in df.columns:
            continue
        subj_col = detect_subject_column(df)
        groups = split_into_subjects(df, subj_col)

        for sid, g in groups:
            x = g[channel_col].to_numpy(dtype=float)
            y = g[GT_COL].to_numpy(dtype=float)
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

            # a_min_new, a_max_new = compute_a_range_from_ranges((x_min, x_max), (y_min, y_max), s_tr, i_tr)
            # a_min_new, a_max_new = compute_a_range_aligned(x, y, s_tr, i_tr)
            a_min_new, a_max_new = compute_a_range_aligned_robust(
                x, y, s_tr, i_tr,
                denom_thresh=1e-3,   # 視資料尺度微調，如 1e-4~1e-2
                q_low=0.01, q_high=0.99  # 或 0.05/0.95
            )
            if not (np.isfinite(a_min_new) and np.isfinite(a_max_new)):
                continue

            if cct in corr:
                if np.isfinite(corr[cct]["min(a)"]):
                    corr[cct]["min(a)"] = min(corr[cct]["min(a)"], a_min_new)
                else:
                    corr[cct]["min(a)"] = a_min_new

                if np.isfinite(corr[cct]["max(a)"]):
                    corr[cct]["max(a)"] = max(corr[cct]["max(a)"], a_max_new)
                else:
                    corr[cct]["max(a)"] = a_max_new
            else:
                corr[cct] = {"slope": s_tr, "intercept": i_tr,
                             "min(a)": a_min_new, "max(a)": a_max_new}

    # 轉換為 DataFrame
    corr_df = (pd.DataFrame.from_dict(corr, orient="index")
               .rename_axis("CCT")
               .reset_index()
               .sort_values("CCT"))
    
    return model_df, corr_df

def detect_available_channels(cct_folders: List[str]) -> List[str]:
    """
    自動檢測在資料中存在的通道
    回傳實際存在的通道列表
    """
    available_channels = set()
    
    for folder in cct_folders:
        csv_path = os.path.join(folder, "data.csv")
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # 檢查哪些通道存在
            for channel in CHANNEL_COLS:
                if channel in df.columns:
                    available_channels.add(channel)
        except Exception as e:
            print(f"[WARN] Error reading {csv_path}: {str(e)}")
            continue
    
    # 按原始順序排序
    detected_channels = [ch for ch in CHANNEL_COLS if ch in available_channels]
    
    if not detected_channels:
        raise RuntimeError("No channels found in any data files!")
    
    print(f"檢測到的可用通道: {detected_channels}")
    print(f"跳過的通道: {[ch for ch in CHANNEL_COLS if ch not in available_channels]}")
    
    return detected_channels

def main():
    # 1) 列出所有 CCT_xxxx 資料夾
    cct_folders = sorted([p for p in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(p)])
    if not cct_folders:
        raise FileNotFoundError(f"No CCT folders found under {DATA_ROOT}")

    # 2) 自動檢測可用的通道
    available_channels = detect_available_channels(cct_folders)
    
    # 3) 對每個可用通道進行處理
    ensure_dir(OUTPUT_LINE_DIR)
    
    # 收集所有通道的结果用于评估
    all_results = {}
    
    for channel_col in available_channels:
        try:
            # 處理單一通道
            model_df, corr_df = process_single_channel(channel_col, cct_folders)
            
            if model_df.empty or corr_df.empty:
                print(f"[WARN] Channel {channel_col}: No valid data, skipping...")
                continue
            
            # 保存结果
            all_results[channel_col] = (model_df, corr_df)
            
            # 輸出 model_line
            model_line_csv = os.path.join(OUTPUT_LINE_DIR, f"model_line_{channel_col}.csv")
            model_df.to_csv(model_line_csv, index=False)
            
            # 輸出 correction 結果
            corr_csv = os.path.join(OUTPUT_LINE_DIR, f"corr_model_line_{channel_col}.csv")
            corr_df.to_csv(corr_csv, index=False)
            
            # 生成視覺化圖表
            plot_ab_lines_with_ranges(corr_df, title=f"AB lines (corrected) per CCT - {channel_col}",
                                    filename=os.path.join(OUTPUT_LINE_DIR, f"viz_ab_lines_corrected_{channel_col}.png"))
            plot_a_ranges_bars(corr_df, title=f"A ranges (corrected) - {channel_col}",
                            filename=os.path.join(OUTPUT_LINE_DIR, f"viz_a_ranges_corrected_{channel_col}.png"))
            
            # 繪製 CCT vs Slope/Intercept 關係圖
            plot_cct_vs_slope_intercept(model_df, title=f"Channel: {channel_col}",
                                      filename=os.path.join(OUTPUT_LINE_DIR, f"viz_cct_vs_slope_intercept_{channel_col}.png"))
            
            print(f"[OK] Channel {channel_col}: Saved model_line and correction results")
            
        except Exception as e:
            print(f"[ERROR] Channel {channel_col}: {str(e)}")
            continue

    print(f"\n[OK] All available channels processed. Results saved under: {OUTPUT_LINE_DIR}")
    if SAVE_INTERMEDIATE:
        print(f"[OK] Per-CCT AB files saved under: {OUTPUT_PARA_DIR}")

    # ===== 測試：使用 min(a)/max(a) 評估預測表現 =====
    # corr_df 含欄位：CCT, slope, intercept, min(a), max(a)
    # 使用剛剛跑過的 cct_folders 與同一份資料逐 subject 計算 coverage/MAE/RMSE
    for channel_col, (model_df, corr_df) in all_results.items():
        try:
            print(f"\n=== 評估通道: {channel_col} ===")
            evaluate_with_corr(
                corr_df=corr_df,
                cct_folders=cct_folders,
                channel_col=channel_col,
                gt_col=GT_COL,
                output_dir=OUTPUT_LINE_DIR,
                detect_subject_column_fn=detect_subject_column,
                split_into_subjects_fn=split_into_subjects
            )
        except Exception as e:
            print(f"[ERROR] 評估通道 {channel_col} 時發生錯誤: {str(e)}")
            continue


if __name__ == "__main__":
    main()
