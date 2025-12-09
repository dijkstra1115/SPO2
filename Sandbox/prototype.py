import numpy as np
import pandas as pd
import re
import os
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 tkinter 错误
import matplotlib.pyplot as plt

# Config
WIN_LEN = 1
STEP = 1
OUTPUT_DIR = "./output"  # 输出目录
SAVE_PLOTS = False  # 是否保存每个 subject 的预测图

# L1 + L2 正则化参数
USE_REGULARIZATION = True  # 是否使用正则化
ALPHA = 1.0  # 正则化强度 (alpha)
L1_RATIO = 0.5  # L1 正则化比例 (0.0 = 纯 L2, 1.0 = 纯 L1, 0.5 = Elastic Net)

# 选择要使用的通道 (可选: "cg", "cr", "yiq_i", "dbdr_dr", "pos_y", "chrom_x")
# 例如: SELECTED_CHANNELS = ["cg", "cr"]  # 只使用 cg 和 cr
#      SELECTED_CHANNELS = ["cg"]         # 只使用 cg
#      SELECTED_CHANNELS = None           # 使用所有6个通道
# 如果设置为 "all_single"，将测试所有单个通道并生成汇总报告
SELECTED_CHANNELS = "all_single"  # None 表示使用所有通道, "all_single" 表示测试所有单个通道

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_subject_id(folder):
    """从 Folder 中提取 [] 内的 id"""
    m = re.search(r'\[(.*?)\]', str(folder))
    return m.group(1) if m else None

def getsixchannels(rgb):
    yiq_i =  0.595716 * rgb[0, :] - 0.274453 * rgb[1, :] - 0.321263 * rgb[2, :]
    ycgcr_cg = 128.0 -  81.085 * rgb[0, :] / 255.0 + 112.000 * rgb[1, :] / 255.0 - 30.915 * rgb[2, :] / 255.0
    ycgcr_cr = 128.0 + 112.000 * rgb[0, :] / 255.0 -  93.786 * rgb[1, :] / 255.0 - 18.214 * rgb[2, :] / 255.0
    ydbdr_dr = -1.333 * rgb[0, :] + 1.116 * rgb[1, :] + 0.217 * rgb[2, :]
    pos_y = -2*rgb[0, :] + rgb[1, :] + rgb[2, :]
    chrom_x =   3*rgb[0, :] - 2*rgb[1, :]
    return [ycgcr_cg, ycgcr_cr, yiq_i, ydbdr_dr, pos_y, chrom_x]

# Load raw data
df = pd.read_csv("./data/prc-c920.csv")

# 提取 Subject ID
df['Subject'] = df['Folder'].apply(extract_subject_id)

all_channel_names = ["cg", "cr", "yiq_i", "dbdr_dr", "pos_y", "chrom_x"]

# Build features once (包含所有通道)
print("=" * 60)
print("Processing all subjects and building features...")
print("=" * 60)
rows = []
for subject_id, g in df.groupby("Subject", sort=False):
    print(f"Processing subject {subject_id}")
    R = g["COLOR_R"].values.astype(float)
    G = g["COLOR_G"].values.astype(float)
    B = g["COLOR_B"].values.astype(float)
    SPO2 = g["SPO2"].values.astype(float)
    N = len(g)
    if N < WIN_LEN: continue

    rgb = np.vstack([R, G, B])
    six = getsixchannels(rgb)
    
    # 构建所有通道的特征
    for start in range(0, N - WIN_LEN + 1, STEP):
        end = start + WIN_LEN
        y = float(SPO2[start:end].mean())
        row = {"subject_id": subject_id, "SPO2_win_mean": y}
        for idx, name in zip(range(len(all_channel_names)), all_channel_names):
            ch = six[idx]
            row[f"{name}_mean"] = ch[start:end].mean()
        rows.append(row)

feat_df_all = pd.DataFrame(rows)
print(f"Feature building completed. Total samples: {len(feat_df_all)}")
print("=" * 60)

def train_and_evaluate(selected_channels, feat_df, save_plots=False):
    """训练模型并评估，返回结果 DataFrame"""
    # 打印使用的通道信息
    if selected_channels is None:
        used_channels = all_channel_names
        print(f"使用所有通道: {used_channels}")
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
        print(f"使用选定通道: {used_channels}")
        if len(used_channels) != len(selected_channels):
            print(f"警告: 部分通道名称无效，已忽略")
    
    # 从完整特征中选择指定通道的特征
    feature_cols = [f"{ch}_mean" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols].copy()

    # Per-person Linear Regression training & backtest
    results_lr_raw = []
    for subject_id, subdf in feat_df_selected.groupby("subject_id", sort=False):
        print(f"Training subject {subject_id}")
        X = subdf.drop(columns=["subject_id", "SPO2_win_mean"])
        y = subdf["SPO2_win_mean"].values
        if len(subdf) < 10: continue
        
        # 根据配置选择使用正则化或普通线性回归
        if USE_REGULARIZATION:
            lr = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=10000, random_state=42)
        else:
            lr = LinearRegression()
        lr.fit(X, y)
        yhat = lr.predict(X)
        
        # 计算评估指标
        r2 = r2_score(y, yhat)
        mae = mean_absolute_error(y, yhat)
        rmse = np.sqrt(mean_squared_error(y, yhat))
        pcc, _ = pearsonr(y, yhat)  # PCC (Pearson Correlation Coefficient)
        
        results_lr_raw.append({
            "subject_id": subject_id,
            "n_frames": len(subdf),
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "PCC": pcc
        })
        
        # 为每个 subject 生成并保存预测曲线图（如果启用）
        if save_plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y, label="True SpO2", linewidth=1.5, color='black')
            plt.plot(yhat, label="Predicted SpO2 (LR)", linewidth=1.5, color='blue', linestyle='--')
            plt.title(f"Subject {subject_id} - Backtest (Linear Regression, R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}, PCC={pcc:.3f})")
            plt.xlabel("Frame index")
            plt.ylabel("SpO2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            # 生成通道标识符用于文件名
            if selected_channels is None:
                ch_suffix = "all"
            else:
                ch_suffix = "_".join(selected_channels)
            output_path = os.path.join(OUTPUT_DIR, f"vis_{subject_id}_{ch_suffix}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()

    results_lr_raw_df = pd.DataFrame(results_lr_raw).sort_values("R2", ascending=False)
    return results_lr_raw_df

# 主程序逻辑
if SELECTED_CHANNELS == "all_single":
    # 测试所有单个通道并生成汇总报告
    print("\n" + "=" * 60)
    print("测试所有单个通道...")
    print("=" * 60)
    
    summary_results = []
    
    # 测试每个单个通道
    for ch in all_channel_names:
        print(f"\n{'='*60}")
        print(f"测试通道: {ch}")
        print(f"{'='*60}")
        results_df = train_and_evaluate([ch], feat_df_all, save_plots=SAVE_PLOTS)
        
        # 计算平均值
        avg_r2 = results_df["R2"].mean()
        avg_mae = results_df["MAE"].mean()
        avg_pcc = results_df["PCC"].mean()
        
        # 计算 PCC 低于阈值的 subject 比例
        total_subjects = len(results_df)
        pcc_lt_0_5_ratio = (results_df["PCC"] < 0.5).sum() / total_subjects
        pcc_lt_0_4_ratio = (results_df["PCC"] < 0.4).sum() / total_subjects
        pcc_lt_0_3_ratio = (results_df["PCC"] < 0.3).sum() / total_subjects
        
        summary_results.append({
            "channel": ch,
            "avg_R2": avg_r2,
            "avg_MAE": avg_mae,
            "avg_PCC": avg_pcc,
            "pcc_lt_0.5_ratio": pcc_lt_0_5_ratio,
            "pcc_lt_0.4_ratio": pcc_lt_0_4_ratio,
            "pcc_lt_0.3_ratio": pcc_lt_0_3_ratio
        })
        
        print(f"通道 {ch} - 平均 R²: {avg_r2:.4f}, 平均 MAE: {avg_mae:.4f}, 平均 PCC: {avg_pcc:.4f}")
        print(f"  PCC < 0.5 比例: {pcc_lt_0_5_ratio:.3f}, PCC < 0.4 比例: {pcc_lt_0_4_ratio:.3f}, PCC < 0.3 比例: {pcc_lt_0_3_ratio:.3f}")
    
    # 测试所有通道组合
    print(f"\n{'='*60}")
    print(f"测试所有通道组合")
    print(f"{'='*60}")
    results_df_all = train_and_evaluate(None, feat_df_all, save_plots=SAVE_PLOTS)
    avg_r2_all = results_df_all["R2"].mean()
    avg_mae_all = results_df_all["MAE"].mean()
    avg_pcc_all = results_df_all["PCC"].mean()
    
    # 计算 PCC 低于阈值的 subject 比例
    total_subjects_all = len(results_df_all)
    pcc_lt_0_5_ratio_all = (results_df_all["PCC"] < 0.5).sum() / total_subjects_all
    pcc_lt_0_4_ratio_all = (results_df_all["PCC"] < 0.4).sum() / total_subjects_all
    pcc_lt_0_3_ratio_all = (results_df_all["PCC"] < 0.3).sum() / total_subjects_all
    
    summary_results.append({
        "channel": "all",
        "avg_R2": avg_r2_all,
        "avg_MAE": avg_mae_all,
        "avg_PCC": avg_pcc_all,
        "pcc_lt_0.5_ratio": pcc_lt_0_5_ratio_all,
        "pcc_lt_0.4_ratio": pcc_lt_0_4_ratio_all,
        "pcc_lt_0.3_ratio": pcc_lt_0_3_ratio_all
    })
    print(f"所有通道 - 平均 R²: {avg_r2_all:.4f}, 平均 MAE: {avg_mae_all:.4f}, 平均 PCC: {avg_pcc_all:.4f}")
    print(f"  PCC < 0.5 比例: {pcc_lt_0_5_ratio_all:.3f}, PCC < 0.4 比例: {pcc_lt_0_4_ratio_all:.3f}, PCC < 0.3 比例: {pcc_lt_0_3_ratio_all:.3f}")
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(OUTPUT_DIR, "channel_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"汇总结果已保存到: {summary_path}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
else:
    # 使用指定的通道配置
    results_lr_raw_df = train_and_evaluate(SELECTED_CHANNELS, feat_df_all, save_plots=SAVE_PLOTS)
    
    # 保存结果表格
    # 生成通道标识符用于文件名
    if SELECTED_CHANNELS is None:
        ch_suffix = "all"
    else:
        ch_suffix = "_".join(SELECTED_CHANNELS)
    results_path = os.path.join(OUTPUT_DIR, f"results_summary_lr_{ch_suffix}.csv")
    results_lr_raw_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    print(f"Total subjects: {len(results_lr_raw_df)}")
    if SAVE_PLOTS:
        print(f"All prediction curves saved to {OUTPUT_DIR}/")