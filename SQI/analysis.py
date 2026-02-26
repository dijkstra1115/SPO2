import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 參數設定
# ==========================================
SOURCE_DIR = "//172.16.1.2/Algorithm/FDA HR Illumination Data/"
META_FILE = os.path.join("filtered_subject_v3.csv")

# ==========================================
# 1. 資料收集與特徵萃取
# ==========================================
print("開始讀取並分析資料...")
meta = pd.read_csv(META_FILE)

# 用來儲存所有受試者的統計結果
results = []
# 用來儲存一個範例受試者的時序資料，畫圖用
example_ts_data = None 

for i in range(len(meta)):
    meta_tmp = meta.iloc[i]

    print(f"Processing subject: {meta_tmp['Folder']}")
    
    DATA_PATH = os.path.join(SOURCE_DIR, "Dump(5.8.3)", str(meta_tmp["Folder"]), "C920.csv")
    
    if not os.path.exists(DATA_PATH):
        continue
        
    # 讀取需要的欄位
    df = pd.read_csv(DATA_PATH, usecols=["COLOR_R", "COLOR_G", "COLOR_B", "SpO2 Score"])

    # ==========================================
    # ★ 新增：略過前 30 秒 (暖機期 WARMUP)
    # ==========================================
    WARMUP_FRAMES = 30 * 30  # 900 frames
    
    if len(df) <= WARMUP_FRAMES:
        print(f"[{meta_tmp['Folder']}] 影片長度不足 30 秒，無有效數據，跳過。")
        continue
        
    # 截斷前 30 秒，只保留真正有在計算的片段
    df = df.iloc[WARMUP_FRAMES:].reset_index(drop=True)
    
    # ----------------------------------
    # 特徵計算
    # ----------------------------------
    # 判斷無效值 (假設小於0 即為無效，因原程式碼沒找到波形會回傳 -1.0)
    invalid_mask = df["SpO2 Score"] < 0
    dropout_rate = invalid_mask.mean() * 100  # 遺失率 (%)
    
    valid_scores = df.loc[~invalid_mask, "SpO2 Score"]
    
    if len(valid_scores) > 0:
        mean_score = valid_scores.mean()
        # 計算分數跳動度 (相鄰幀絕對差值的平均)
        score_jitter = valid_scores.diff().abs().mean() 
    else:
        mean_score = np.nan
        score_jitter = np.nan
        
    mean_R = df["COLOR_R"].mean()
    mean_G = df["COLOR_G"].mean()
    
    # 簡單評估綠光的雜訊程度 (利用 rolling standard deviation 減去大趨勢)
    # 若綠光高頻雜訊大，此值會較高
    g_noise_proxy = df["COLOR_G"].rolling(window=15).std().mean()

    results.append({
        "Subject": meta_tmp.get("Folder", f"Sub_{i}"),
        "Mean_R": mean_R,
        "Mean_G": mean_G,
        "Mean_Score": mean_score,
        "Dropout_Rate_%": dropout_rate,
        "Score_Jitter": score_jitter,
        "G_Noise_Proxy": g_noise_proxy
    })
    
    # 保留第一位有效受試者的時序資料作範例圖
    if example_ts_data is None and len(valid_scores) > 500:
        example_ts_data = df.copy()

results_df = pd.DataFrame(results).dropna()
print(f"成功分析 {len(results_df)} 位受試者資料。")

# ==========================================
# 2. 視覺化分析 (產生 4 張分析圖)
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 圖 1: 亮度偏差分析 (證明絕對殘差的 Bug)
sns.scatterplot(data=results_df, x="Mean_R", y="Mean_Score", ax=axes[0, 0], alpha=0.7)
# 加上趨勢線
sns.regplot(data=results_df, x="Mean_R", y="Mean_Score", ax=axes[0, 0], scatter=False, color='red')
axes[0, 0].set_title("1. Brightness Bias (Mean Red vs Mean SQI)\n(Ideal: Flat line. If tilted, it proves normalization bug)")
axes[0, 0].set_xlabel("Average Red Channel Intensity (Brightness)")
axes[0, 0].set_ylabel("Average Valid SpO2 Score")

# 圖 2: 訊號遺失率分佈 (Dropout Rate)
sns.histplot(results_df["Dropout_Rate_%"], bins=20, kde=True, ax=axes[0, 1], color="orange")
axes[0, 1].set_title("2. Distribution of Dropout Rate (-1.0 occurence)\n(Complaints about 'Cannot measure')")
axes[0, 1].set_xlabel("Dropout Rate (%)")
axes[0, 1].set_ylabel("Number of Subjects")

# 圖 3: 分數跳動度分佈 (Jitter)
sns.histplot(results_df["Score_Jitter"], bins=20, kde=True, ax=axes[1, 0], color="green")
axes[1, 0].set_title("3. Temporal Jitter of SpO2 Score\n(High value means the score jumps wildly frame-by-frame)")
axes[1, 0].set_xlabel("Mean Absolute Frame-to-Frame Score Difference")
axes[1, 0].set_ylabel("Number of Subjects")

# 圖 4: 綠光品質與分數的關聯 (驗證綠光盲區)
sns.scatterplot(data=results_df, x="G_Noise_Proxy", y="Mean_Score", ax=axes[1, 1], alpha=0.7, color="purple")
axes[1, 1].set_title("4. Green Channel Noise vs SQI\n(If score is high even when G noise is high, algorithm is ignoring G)")
axes[1, 1].set_xlabel("Green Channel Noise Proxy (Rolling Std)")
axes[1, 1].set_ylabel("Average Valid SpO2 Score")

plt.tight_layout()
plt.savefig("SQI_Analysis_Dashboard.png", dpi=150)
print("已將分析圖表儲存為 'SQI_Analysis_Dashboard.png'")
plt.show()

# ==========================================
# 3. 範例受試者時序折線圖 (Time-series)
# ==========================================
if example_ts_data is not None:
    fig2, ax1 = plt.subplots(figsize=(12, 4))
    
    # 畫出 R channel 趨勢 (標準化方便顯示)
    r_norm = (example_ts_data["COLOR_R"] - example_ts_data["COLOR_R"].min()) / (example_ts_data["COLOR_R"].max() - example_ts_data["COLOR_R"].min() + 1e-5)
    ax1.plot(r_norm, color='red', alpha=0.5, label='R Channel (Normalized)')
    
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Normalized Intensity")
    ax1.tick_params(axis='y', labelcolor='red')
    
    # 畫出 SQI 趨勢
    ax2 = ax1.twinx()
    # 濾除 -1 方便觀看
    score_plot = example_ts_data["SpO2 Score"].replace(to_replace=-1.0, value=np.nan)
    ax2.plot(score_plot, color='blue', linewidth=2, label='SpO2 SQI')
    ax2.set_ylabel("SpO2 Score", color='blue')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Example Subject: Temporal Stability of SQI")
    fig2.tight_layout()
    plt.savefig("SQI_Example_Timeseries.png", dpi=150)
    print("已將時序範例圖儲存為 'SQI_Example_Timeseries.png'")
    plt.show()