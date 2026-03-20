import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def safe_pearsonr(x, y):
    """當任一陣列無變異時回傳 nan，避免報錯。"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < 2 or x_clean.std() == 0 or y_clean.std() == 0:
        return np.nan, np.nan
    return pearsonr(x_clean, y_clean)

# ================= 配置區 =================
SOURCE_A_DIR = "//172.16.1.2/Algorithm/FDA HR Illumination Data/"
SOURCE_B_DIR = "//172.16.1.3/Public/temp/Sqi_dev"
META_FILE = "filtered_subject_v3.csv"
OUTPUT_DIR = "./ANALYSIS_SPO2_METRICS"

# 指標門檻設定
SPO2_NORMAL_MIN = 94          # 正常血氧下限 (%) 假設還是 0-100 的尺度
SCORE_ANOMALY_THRESH = 0.7    # Score 低於 0.7 視為「異常低分」
SKIP_HEAD_FRAMES = 900        # 跳過前 N 幀（暖機階段）

# 欄位名稱設定
COL_SPO2 = "SpO2"             # 血氧數值的欄位名稱
COL_SCORE = "SpO2 Score"      # 分數的欄位名稱
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    meta_path = os.path.join(SOURCE_B_DIR, META_FILE)
    meta = pd.read_csv(meta_path)
except Exception as e:
    print(f"讀取 meta 失敗: {e}，使用測試模式")
    meta = pd.DataFrame({"Folder": ["Test_Folder_01"]})

summary_results = []
invalid_subjects = [] # 紀錄被剃除的受試者名單

for i in range(len(meta)):
    folder_name = meta.iloc[i]["Folder"]

    print(f"Processing subject: {folder_name}")
    
    PATH_A = os.path.join(SOURCE_A_DIR, "Dump(5.8.3)", folder_name, "C920.csv")
    PATH_B = os.path.join(SOURCE_B_DIR, "Dump_new(5.8.6)", folder_name, "C920.csv")
    
    if not os.path.exists(PATH_A) or not os.path.exists(PATH_B):
        continue
        
    df_A = pd.read_csv(PATH_A)
    df_B = pd.read_csv(PATH_B)
    
    min_len = min(len(df_A), len(df_B))
    
    combined_df = pd.DataFrame({
        "SpO2_Val": df_A[COL_SPO2].iloc[:min_len],
        "Score_A": df_A[COL_SCORE].iloc[:min_len],
        "Score_B": df_B[COL_SCORE].iloc[:min_len]
    })
    
    # 1. 剃除暖機階段的資料
    combined_df = combined_df.iloc[SKIP_HEAD_FRAMES:].reset_index(drop=True)
    if len(combined_df) == 0:
        continue

    # ==========================================
    # 2. ★ 新增：過濾無效受試者 ★
    # 如果暖機後，SpO2_Val 幾乎全為 -1，或者任一版本的 Score 全為 0，就把這個受試者剃除
    # (使用 .all() 代表整段都是這個數值)
    # ==========================================
    if (combined_df["SpO2_Val"] == -1).all() or \
       (combined_df["Score_A"] == 0).all() or \
       (combined_df["Score_B"] == 0).all():
        
        print(f"[過濾] 剃除受試者: {folder_name} (原因: 暖機後 SpO2 或 Score 皆為無效預設值 -1 / 0)")
        invalid_subjects.append(folder_name)
        continue  # 直接跳過，不往下進行計算與畫圖

    # ==========================================
    # 核心指標計算
    # ==========================================
    normal_mask = combined_df["SpO2_Val"] >= SPO2_NORMAL_MIN
    normal_data = combined_df[normal_mask]
    
    if len(normal_data) > 0:
        discrepancy_A = (normal_data["Score_A"] < SCORE_ANOMALY_THRESH).mean() * 100
        discrepancy_B = (normal_data["Score_B"] < SCORE_ANOMALY_THRESH).mean() * 100
    else:
        discrepancy_A, discrepancy_B = np.nan, np.nan

    # 真實異常捕捉率 (Sensitivity to Hypoxia): SpO2 < 94 時 Score < 0.7 的比例
    hypoxia_mask = combined_df["SpO2_Val"] < SPO2_NORMAL_MIN
    n_hypoxia = hypoxia_mask.sum()
    if n_hypoxia > 0:
        sens_A = ((combined_df["Score_A"] < SCORE_ANOMALY_THRESH) & hypoxia_mask).sum() / n_hypoxia
        sens_B = ((combined_df["Score_B"] < SCORE_ANOMALY_THRESH) & hypoxia_mask).sum() / n_hypoxia
    else:
        sens_A, sens_B = np.nan, np.nan

    # 相對改善率 (以 A 為基準的 FAR 改善 %)
    far_A, far_B = discrepancy_A, discrepancy_B
    if not np.isnan(far_A) and far_A > 0:
        rel_improve_far = (far_A - far_B) / far_A * 100.0
    else:
        rel_improve_far = np.nan

    corr_A, _ = safe_pearsonr(combined_df["SpO2_Val"], combined_df["Score_A"])
    corr_B, _ = safe_pearsonr(combined_df["SpO2_Val"], combined_df["Score_B"])

    cv_A = (combined_df["Score_A"].std() / (combined_df["Score_A"].mean() + 1e-9)) * 100
    cv_B = (combined_df["Score_B"].std() / (combined_df["Score_B"].mean() + 1e-9)) * 100
    
    swing_A = combined_df["Score_A"].diff().abs().max()
    swing_B = combined_df["Score_B"].diff().abs().max()

    summary_results.append({
        "Folder": folder_name,
        "Valid_Frames": len(combined_df),
        "Normal_SpO2_Frames": len(normal_data),
        "N_Hypoxia_Frames": int(n_hypoxia),
        "A_Discrepancy(%)": discrepancy_A,
        "B_Discrepancy(%)": discrepancy_B,
        "Rel_Improvement_FAR(%)": rel_improve_far,
        "A_Sensitivity_Hypoxia": sens_A,
        "B_Sensitivity_Hypoxia": sens_B,
        "A_Corr_SpO2": corr_A,
        "B_Corr_SpO2": corr_B,
        "A_CV(%)": cv_A,
        "B_CV(%)": cv_B,
        "A_Max_Swing": swing_A,
        "B_Max_Swing": swing_B
    })

    # ==========================
    # 視覺化報表
    # ==========================
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 1)

    # 圖 1: SpO2 值與 A/B Score 時序對比
    ax1 = fig.add_subplot(gs[0])
    ax1_twin = ax1.twinx()
    
    ax1.plot(combined_df["Score_A"], color='tab:red', alpha=0.6, label=f"Ver A Score (CV: {cv_A:.1f}%)")
    ax1.plot(combined_df["Score_B"], color='tab:blue', alpha=0.8, label=f"Ver B Score (CV: {cv_B:.1f}%)")
    ax1.set_ylabel("SpO2 Score", fontsize=12)
    ax1.set_ylim(-0.05, 1.05) 
    
    # SpO2 數值（綠色）
    ax1_twin.plot(combined_df["SpO2_Val"], color='green', alpha=0.7, linewidth=2, label="SpO2 Value")
    ax1_twin.set_ylabel("SpO2 (%)", fontsize=12)
    ax1_twin.set_ylim(85, 102) 
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')
    ax1.set_title(f"Timeseries: SpO2 vs Scores - {folder_name}", fontsize=14)

    # 圖 2: SpO2 vs Score 散佈圖 
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(combined_df["SpO2_Val"], combined_df["Score_A"], color='red', alpha=0.2, s=20, label="Ver A")
    ax2.scatter(combined_df["SpO2_Val"], combined_df["Score_B"], color='blue', alpha=0.3, s=20, marker='x', label="Ver B")
    
    ax2.axvline(x=SPO2_NORMAL_MIN, color='green', linestyle='--')
    ax2.axhline(y=SCORE_ANOMALY_THRESH, color='gray', linestyle='--')
    
    ax2.fill_between([SPO2_NORMAL_MIN, 100], 0, SCORE_ANOMALY_THRESH, color='orange', alpha=0.1, label='Customer Confusion Zone')
    
    ax2.set_title(f"Correlation Scatter (A_Corr: {corr_A:.3f} | B_Corr: {corr_B:.3f})")
    ax2.set_xlabel("SpO2 Value (%)")
    ax2.set_ylabel("SpO2 Score")
    ax2.set_xlim(85, 102)
    ax2.set_ylim(-0.05, 1.05) 
    ax2.legend()

    # 圖 3: Score 的一階差分
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(combined_df["Score_A"].diff().abs(), color='red', alpha=0.5, label=f"Ver A Swing (Max: {swing_A:.3f})")
    ax3.plot(combined_df["Score_B"].diff().abs(), color='blue', alpha=0.7, label=f"Ver B Swing (Max: {swing_B:.3f})")
    ax3.set_title("Frame-to-Frame Score Swing (Absolute Difference 0~1 scale)")
    ax3.set_ylabel("Score Change")
    ax3.set_ylim(0, 0.5) 
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{folder_name}_SpO2_Metrics.png"))
    plt.close()

# 輸出總結
if len(summary_results) > 0:
    summary_df = pd.DataFrame(summary_results)
    print("=== 整體改善總結 ===")
    print(f"舊版 A 異常率 (FAR): {summary_df['A_Discrepancy(%)'].mean():.2f}%")
    print(f"新版 B 異常率 (FAR): {summary_df['B_Discrepancy(%)'].mean():.2f}%")
    valid_rel = summary_df["Rel_Improvement_FAR(%)"].dropna()
    if len(valid_rel) > 0:
        print(f"FAR 相對改善 (以 A 為基準): 平均 {valid_rel.mean():.1f}%")
    print("-" * 20)
    # 真實異常捕捉率：僅對「有低血氧幀」的受試者平均
    has_hypoxia = summary_df["N_Hypoxia_Frames"] > 0
    if has_hypoxia.any():
        sub = summary_df[has_hypoxia]
        print(f"真實異常捕捉率 (僅有低血氧之受試者): A={sub['A_Sensitivity_Hypoxia'].mean():.2%} | B={sub['B_Sensitivity_Hypoxia'].mean():.2%}")
    else:
        print("真實異常捕捉率: 無受試者含低血氧幀，略過")
    print("-" * 20)
    print(f"舊版 A 平均相關性: {summary_df['A_Corr_SpO2'].mean():.3f}")
    print(f"新版 B 平均相關性: {summary_df['B_Corr_SpO2'].mean():.3f}")
    print("-" * 20)
    print(f"舊版 A 平均 CV: {summary_df['A_CV(%)'].mean():.2f}%")
    print(f"新版 B 平均 CV: {summary_df['B_CV(%)'].mean():.2f}%")
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Metrics_Comparison_Summary.csv"), index=False)