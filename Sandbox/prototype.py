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

# Normalization 参数
USE_NORMALIZATION = True  # 是否使用归一化

# L1 + L2 正则化参数
USE_REGULARIZATION = False  # 是否使用正则化
ALPHA = 1.0  # 正则化强度 (alpha)
L1_RATIO = 0.5  # L1 正则化比例 (0.0 = 纯 L2, 1.0 = 纯 L1, 0.5 = Elastic Net)

# 选择要使用的通道 (可选: "cg", "cr", "yiq_i", "dbdr_dr", "pos_y", "chrom_x")
# 例如: SELECTED_CHANNELS = ["cg", "cr"]  # 只使用 cg 和 cr
#      SELECTED_CHANNELS = ["cg"]         # 只使用 cg
#      SELECTED_CHANNELS = None           # 使用所有6个通道
# 如果设置为 "all_single"，将测试所有单个通道并生成汇总报告
# 如果设置为 "ensemble"，将使用 ensemble model pool 进行预测
SELECTED_CHANNELS = "ensemble"  # None 表示使用所有通道, "all_single" 表示测试所有单个通道, "ensemble" 表示使用 ensemble 模型池

# Ensemble 模型池配置
TOP_K = 5  # 从其他模型中选择 TOP_K 个最相似的模型进行 ensemble
SEGMENT_LENGTH = 900  # 每个模型训练时使用的固定样本数（段长度）

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

def find_global_valid_features(feat_df, min_subject_ratio=0.0):
    """
    方案一：全局特征筛选
    找出在所有 subject 中都不是常数的特征（或至少在 min_subject_ratio 比例的 subject 中不是常数）
    
    Parameters:
    -----------
    feat_df : DataFrame
        特征数据，包含 'subject_id' 列和所有特征列
    min_subject_ratio : float
        最小 subject 比例阈值。0.0 表示必须在所有 subject 中都不是常数（交集）
        0.8 表示至少在 80% 的 subject 中不是常数（并集）
    
    Returns:
    --------
    valid_features : list
        有效的特征列名列表
    feature_stats : dict
        每个特征的统计信息，包括：
        - 'constant_in_subjects': 在哪些 subject 中是常数
        - 'valid_subject_count': 在多少个 subject 中不是常数
        - 'valid_subject_ratio': 在多少比例的 subject 中不是常数
    """
    print("\n" + "=" * 60)
    print("全局特征筛选分析...")
    print("=" * 60)
    
    # 获取所有特征列（排除 subject_id 和 SPO2_win_mean）
    all_feature_cols = [col for col in feat_df.columns 
                       if col not in ['subject_id', 'SPO2_win_mean']]
    
    print(f"总特征数: {len(all_feature_cols)}")
    print(f"特征列表: {all_feature_cols}")
    
    # 统计每个特征在每个 subject 中的情况
    feature_stats = {}
    for feat_col in all_feature_cols:
        feature_stats[feat_col] = {
            'constant_in_subjects': [],
            'valid_subject_count': 0,
            'valid_subject_ratio': 0.0
        }
    
    total_subjects = 0
    subject_constant_features = {}  # 记录每个 subject 的常数特征
    
    # 遍历每个 subject
    for subject_id, subdf in feat_df.groupby("subject_id", sort=False):
        total_subjects += 1
        X_subject = subdf[all_feature_cols].astype(float)
        
        # 计算每个特征的标准差
        stds = X_subject.std(axis=0)
        constant_mask = stds < 1e-8
        
        # 记录这个 subject 的常数特征
        constant_features = X_subject.columns[constant_mask].tolist()
        subject_constant_features[subject_id] = constant_features
        
        # 更新每个特征的统计信息
        for feat_col in all_feature_cols:
            if feat_col in constant_features:
                feature_stats[feat_col]['constant_in_subjects'].append(subject_id)
            else:
                feature_stats[feat_col]['valid_subject_count'] += 1
    
    # 计算每个特征的 valid_subject_ratio
    for feat_col in all_feature_cols:
        feature_stats[feat_col]['valid_subject_ratio'] = (
            feature_stats[feat_col]['valid_subject_count'] / total_subjects
        )
    
    # 根据 min_subject_ratio 筛选有效特征
    valid_features = []
    for feat_col in all_feature_cols:
        ratio = feature_stats[feat_col]['valid_subject_ratio']
        if ratio >= (1.0 - min_subject_ratio):
            valid_features.append(feat_col)
    
    # 打印详细统计信息
    print(f"\n总 subject 数: {total_subjects}")
    print(f"筛选阈值: 至少在 {(1.0 - min_subject_ratio) * 100:.1f}% 的 subject 中不是常数")
    print(f"\n有效特征数: {len(valid_features)} / {len(all_feature_cols)}")
    
    if len(valid_features) > 0:
        print(f"\n有效特征列表:")
        for feat in valid_features:
            stats = feature_stats[feat]
            print(f"  - {feat}: 在 {stats['valid_subject_count']}/{total_subjects} "
                  f"({stats['valid_subject_ratio']*100:.1f}%) subject 中不是常数")
    else:
        print("\n警告: 没有找到符合条件的特征！")
    
    # 打印被移除的特征
    removed_features = [f for f in all_feature_cols if f not in valid_features]
    if len(removed_features) > 0:
        print(f"\n被移除的特征 ({len(removed_features)} 个):")
        for feat in removed_features:
            stats = feature_stats[feat]
            const_subjects = stats['constant_in_subjects']
            print(f"  - {feat}: 在 {stats['valid_subject_count']}/{total_subjects} "
                  f"({stats['valid_subject_ratio']*100:.1f}%) subject 中不是常数")
            if len(const_subjects) > 0:
                print(f"    常数 subject: {const_subjects[:5]}{'...' if len(const_subjects) > 5 else ''}")
    
    # 打印每个 subject 的常数特征统计
    print(f"\n各 Subject 的常数特征统计:")
    for subject_id, const_feats in sorted(subject_constant_features.items()):
        print(f"  Subject {subject_id}: {len(const_feats)} 个常数特征")
        if len(const_feats) > 0:
            print(f"    {const_feats}")
    
    print("=" * 60)
    
    return valid_features, feature_stats

# 全局特征筛选已移除 - 现在采用新策略：
# 始终使用全部6个特征，如果段中有任何常数特征则跳过该段
# 这样可以保证所有模型的特征维度完全一致 (300, 6)

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
        
        # 取出原始 X / y
        X_raw = subdf.drop(columns=["subject_id", "SPO2_win_mean"]).astype(float)
        y = subdf["SPO2_win_mean"].values.astype(float)
        if len(subdf) < 10:
            continue

        # ====== NEW: per-subject normalization ======
        if USE_NORMALIZATION:
            # 計算每個 feature 的 mean/std
            means = X_raw.mean(axis=0)
            stds = X_raw.std(axis=0)

            # 避免 std = 0 的欄位（完全沒變化）
            nonzero_mask = stds > 1e-8
            if not np.all(nonzero_mask):
                dropped_cols = X_raw.columns[~nonzero_mask].tolist()
                if len(dropped_cols) > 0:
                    print(f"  主體 {subject_id} 有常數特徵被移除: {dropped_cols}")
                X_raw = X_raw.loc[:, nonzero_mask]
                means = means[nonzero_mask]
                stds = stds[nonzero_mask]

            # 真正做 normalization
            X = (X_raw - means) / stds
        else:
            # 不使用 normalization，但仍需要检查常数特征
            stds = X_raw.std(axis=0)
            nonzero_mask = stds > 1e-8
            if not np.all(nonzero_mask):
                dropped_cols = X_raw.columns[~nonzero_mask].tolist()
                if len(dropped_cols) > 0:
                    print(f"  主體 {subject_id} 有常數特徵被移除: {dropped_cols}")
                X_raw = X_raw.loc[:, nonzero_mask]
            X = X_raw
        # ============================================

        # 根據配置選擇使用正則化或普通線性回歸
        if USE_REGULARIZATION:
            # 你目前是用 ElasticNet，如果之後想改成 Ridge：
            from sklearn.linear_model import Ridge
            lr = Ridge(alpha=ALPHA)
            # lr = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=10000, random_state=42)
        else:
            lr = LinearRegression()

        lr.fit(X, y)
        yhat = lr.predict(X)

        # 計算評估指標
        r2 = r2_score(y, yhat)
        mae = mean_absolute_error(y, yhat)
        rmse = np.sqrt(mean_squared_error(y, yhat))

        # ====== NEW: 避免 constant 警告 ======
        if np.std(yhat) < 1e-8:
            # 預測是常數，pearsonr 會丟 ConstantInputWarning
            pcc = np.nan   # 或者你想設成 0 也可以
            print(f"  主體 {subject_id} 預測為常數，PCC 設為 NaN")
        else:
            pcc, _ = pearsonr(y, yhat)
        # ====================================

        results_lr_raw.append({
            "subject_id": subject_id,
            "n_frames": len(subdf),
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "PCC": pcc
        })
        
        # 為每个 subject 生成并保存预测曲线图（如果启用）
        if save_plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y, label="True SpO2", linewidth=1.5, color='black')
            plt.plot(yhat, label="Predicted SpO2 (LR)", linewidth=1.5, color='blue', linestyle='--')
            plt.title(f"Subject {subject_id} - Backtest (Linear Regression, R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}, PCC={pcc if not np.isnan(pcc) else 0:.3f})")
            plt.xlabel("Frame index")
            plt.ylabel("SpO2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            if selected_channels is None:
                ch_suffix = "all"
            else:
                ch_suffix = "_".join(selected_channels)
            output_path = os.path.join(OUTPUT_DIR, f"vis_{subject_id}_{ch_suffix}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()

    results_lr_raw_df = pd.DataFrame(results_lr_raw).sort_values("R2", ascending=False)
    return results_lr_raw_df

def calculate_similarity(features_A, features_B):
    """
    计算两组特征之间的相似度（使用 Pearson Correlation）
    
    假设两组特征的维度完全相同（segment_length x n_features），
    将特征矩阵 flatten 后计算 Pearson correlation。
    
    Parameters:
    -----------
    features_A : DataFrame or ndarray
        第一组特征 (segment_length, n_features)
    features_B : DataFrame or ndarray
        第二组特征 (segment_length, n_features)
    
    Returns:
    --------
    similarity_score : float
        相似度分数（-1 到 1 之间，越大越相似）
        如果计算失败，返回 -1
    """
    # 转换为 numpy array
    if isinstance(features_A, pd.DataFrame):
        features_A = features_A.values
    if isinstance(features_B, pd.DataFrame):
        features_B = features_B.values
    
    # 检查维度是否相同
    if features_A.shape != features_B.shape:
        print(f"  警告: 特征维度不匹配 {features_A.shape} vs {features_B.shape}")
        return -1.0
    
    # Flatten 特征矩阵
    vec_A = features_A.flatten()
    vec_B = features_B.flatten()
    
    # 检查是否有常数向量
    if np.std(vec_A) < 1e-8 or np.std(vec_B) < 1e-8:
        return -1.0
    
    # 计算 Pearson correlation
    try:
        corr, _ = pearsonr(vec_A, vec_B)
        return corr if not np.isnan(corr) else -1.0
    except Exception as e:
        print(f"  计算 correlation 时出错: {e}")
        return -1.0

def train_model_pool(selected_channels, feat_df, segment_length=SEGMENT_LENGTH):
    """
    训练所有 subject 的模型并构建模型池（使用段切分策略）
    
    策略：
    - 始终使用全部特征（6个通道）
    - 如果段中有任何常数特征（std=0），则跳过该段
    - 这样可以保证所有模型的特征维度完全一致 (segment_length, n_features)
    
    Parameters:
    -----------
    selected_channels : list or None
        要使用的通道列表
    feat_df : DataFrame
        特征数据
    segment_length : int
        每个模型训练时使用的固定样本数（段长度）
    
    Returns:
    --------
    model_pool : list of dict
        模型池，每个元素包含:
        - 'subject_id': subject ID
        - 'segment_id': 段 ID（从 0 开始）
        - 'model_id': 唯一模型标识 (subject_id_segment_id)
        - 'model': 训练好的模型对象
        - 'feature_cols': 使用的特征列名（全部特征）
        - 'means': 归一化的均值
        - 'stds': 归一化的标准差
        - 'X_raw_segment': 原始特征矩阵 (segment_length, n_features)，用于相似度计算
    """
    print("\n" + "=" * 60)
    print(f"训练模型池（段长度={segment_length}）...")
    print("=" * 60)
    
    # 确定使用的通道
    if selected_channels is None:
        used_channels = all_channel_names
        print(f"使用所有通道: {used_channels}")
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
        print(f"使用选定通道: {used_channels}")
    
    # 选择特征列
    feature_cols = [f"{ch}_mean" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols].copy()
    
    model_pool = []
    total_segments = 0
    skipped_subjects = 0
    
    # 为每个 subject 切分并训练多个模型
    for subject_id, subdf in feat_df_selected.groupby("subject_id", sort=False):
        n_samples = len(subdf)
        print(f"\n处理 Subject {subject_id}: {n_samples} 个样本")
        
        # 检查样本数是否足够
        if n_samples < segment_length:
            print(f"  跳过: 样本数不足 {segment_length}，实际只有 {n_samples}")
            skipped_subjects += 1
            continue
        
        # 计算可以切分多少段（非重叠）
        n_segments = n_samples // segment_length
        print(f"  可切分为 {n_segments} 段")
        
        # 取出原始 X / y
        X_full = subdf.drop(columns=["subject_id", "SPO2_win_mean"]).astype(float)
        y_full = subdf["SPO2_win_mean"].values.astype(float)
        
        # 对每一段进行训练
        for seg_idx in range(n_segments):
            start_idx = seg_idx * segment_length
            end_idx = start_idx + segment_length
            
            # 提取该段的数据
            X_raw_segment = X_full.iloc[start_idx:end_idx].copy()
            y_segment = y_full[start_idx:end_idx]
            
            # 检查是否有任何常数特征（std = 0）
            stds = X_raw_segment.std(axis=0)
            has_constant_features = (stds < 1e-8).any()
            
            if has_constant_features:
                # 如果有常数特征，直接跳过该段
                constant_cols = X_raw_segment.columns[stds < 1e-8].tolist()
                print(f"  段 {seg_idx}: 跳过（有常数特征: {constant_cols}）")
                continue
            
            # 所有特征都有效，使用全部特征进行训练
            # Per-segment normalization
            if USE_NORMALIZATION:
                means = X_raw_segment.mean(axis=0)
                stds = X_raw_segment.std(axis=0)
                # Normalization
                X_normalized = (X_raw_segment - means) / stds
            else:
                # 不使用 normalization
                means = None
                stds = None
                X_normalized = X_raw_segment
            
            # 训练模型
            if USE_REGULARIZATION:
                from sklearn.linear_model import Ridge
                lr = Ridge(alpha=ALPHA)
            else:
                lr = LinearRegression()
            
            lr.fit(X_normalized, y_segment)
            
            # 保存到模型池
            # 注意：所有模型都使用全部特征，维度固定为 (segment_length, n_features)
            model_id = f"{subject_id}_seg{seg_idx}"
            model_pool.append({
                'subject_id': subject_id,
                'segment_id': seg_idx,
                'model_id': model_id,
                'model': lr,
                'feature_cols': X_raw_segment.columns.tolist(),  # 全部特征列
                'means': means.values if means is not None else None,
                'stds': stds.values if stds is not None else None,
                'X_raw_segment': X_raw_segment.values  # 保存原始特征矩阵 (segment_length, n_features) 用于相似度计算
            })
            total_segments += 1
        
        print(f"  Subject {subject_id} 完成: 训练了 {n_segments} 个模型")
    
    print(f"\n模型池构建完成:")
    print(f"  总 subject 数: {len(feat_df_selected['subject_id'].unique())}")
    print(f"  跳过 subject 数: {skipped_subjects}")
    print(f"  有效 subject 数: {len(feat_df_selected['subject_id'].unique()) - skipped_subjects}")
    print(f"  总模型数: {len(model_pool)}")
    print(f"  平均每个 subject: {len(model_pool) / max(1, len(feat_df_selected['subject_id'].unique()) - skipped_subjects):.1f} 个模型")
    print("=" * 60)
    return model_pool

def ensemble_predict_and_evaluate(model_pool, feat_df, selected_channels, top_k, segment_length=SEGMENT_LENGTH, save_plots=False):
    """
    使用 ensemble 模型池进行预测和评估（使用段切分策略）
    
    Parameters:
    -----------
    model_pool : list of dict
        训练好的模型池
    feat_df : DataFrame
        特征数据
    selected_channels : list or None
        使用的通道列表
    top_k : int
        选择 TOP_K 个最相似的模型
    segment_length : int
        段长度，应与训练时一致
    save_plots : bool
        是否保存预测图表
    
    Returns:
    --------
    results_df : DataFrame
        评估结果
    """
    print("\n" + "=" * 60)
    print(f"使用 Ensemble 模型池进行预测 (TOP_K={top_k}, 段长度={segment_length})...")
    print("=" * 60)
    
    # 确定使用的通道
    if selected_channels is None:
        used_channels = all_channel_names
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
    
    feature_cols = [f"{ch}_mean" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols].copy()
    
    results = []
    
    # 对每个 subject 进行测试
    for test_subject_id, test_subdf in feat_df_selected.groupby("subject_id", sort=False):
        n_samples = len(test_subdf)
        print(f"\n测试 Subject {test_subject_id}: {n_samples} 个样本")
        
        # 检查样本数是否足够
        if n_samples < segment_length:
            print(f"  跳过: 样本数不足 {segment_length}")
            continue
        
        # 获取测试数据的完整特征和标签
        X_test_full = test_subdf.drop(columns=["subject_id", "SPO2_win_mean"]).astype(float)
        y_test_full = test_subdf["SPO2_win_mean"].values.astype(float)
        
        # 计算可以切分多少段
        n_test_segments = n_samples // segment_length
        print(f"  可切分为 {n_test_segments} 段")
        
        # 对每一段进行预测
        all_segment_predictions = []
        all_segment_labels = []
        
        for test_seg_idx in range(n_test_segments):
            start_idx = test_seg_idx * segment_length
            end_idx = start_idx + segment_length
            
            # 提取该测试段的数据
            X_test_segment_df = X_test_full.iloc[start_idx:end_idx].copy()
            y_test_segment = y_test_full[start_idx:end_idx]
            
            # 检查测试段是否有常数特征（与训练时保持一致）
            test_stds = X_test_segment_df.std(axis=0)
            has_constant_features = (test_stds < 1e-8).any()
            
            if has_constant_features:
                # 如果有常数特征，直接跳过该测试段
                constant_cols = X_test_segment_df.columns[test_stds < 1e-8].tolist()
                print(f"  测试段 {test_seg_idx}: 跳过（有常数特征: {constant_cols}）")
                continue
            
            # 计算该测试段与模型池中所有模型的相似度
            # 注意：相似度计算使用 numpy array，预测使用 DataFrame
            X_test_segment_array = X_test_segment_df.values
            
            similarities = []
            for model_info in model_pool:
                model_subject_id = model_info['subject_id']
                model_id = model_info['model_id']
                
                # 排除来自同一个 subject 的模型
                if model_subject_id == test_subject_id:
                    continue
                
                # 获取该模型训练时使用的特征矩阵（维度固定为 segment_length x n_features）
                X_train_segment = model_info['X_raw_segment']
                
                # 计算相似度（使用 correlation，使用 numpy array）
                # 现在两个矩阵维度应该完全一致 (segment_length, n_features)
                similarity = calculate_similarity(X_train_segment, X_test_segment_array)
                similarities.append({
                    'model_info': model_info,
                    'similarity': similarity,
                    'model_id': model_id
                })
            
            if len(similarities) == 0:
                print(f"  测试段 {test_seg_idx}: 没有可用的模型")
                continue
            
            # 按相似度排序，选择 TOP_K
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_k_models = similarities[:min(top_k, len(similarities))]
            
            # 打印选择的模型
            if test_seg_idx == 0:  # 只在第一段打印详细信息
                print(f"  测试段 {test_seg_idx} 选择的 TOP_{min(top_k, len(similarities))} 模型:")
                for rank, sim_info in enumerate(top_k_models, 1):
                    print(f"    {rank}. {sim_info['model_id']}, 相似度: {sim_info['similarity']:.4f}")
            
            # 使用 TOP_K 模型进行预测
            segment_predictions = []
            for sim_info in top_k_models:
                model_info = sim_info['model_info']
                model = model_info['model']
                means = model_info['means']
                stds = model_info['stds']
                feature_cols_model = model_info['feature_cols']
                
                # 应用相同的预处理（使用 DataFrame 以匹配训练时的特征名称）
                # 确保特征顺序与训练时一致
                X_test_segment_aligned = X_test_segment_df[feature_cols_model].copy()
                
                # 根据训练时是否使用 normalization 来决定预处理方式
                if USE_NORMALIZATION and means is not None and stds is not None:
                    X_test_normalized = (X_test_segment_aligned - means) / stds
                else:
                    X_test_normalized = X_test_segment_aligned
                
                # 预测（使用 DataFrame，避免特征名称警告）
                y_pred = model.predict(X_test_normalized)
                segment_predictions.append(y_pred)
            
            # Ensemble: 取平均
            if len(segment_predictions) > 0:
                y_segment_ensemble = np.mean(segment_predictions, axis=0)
                all_segment_predictions.append(y_segment_ensemble)
                all_segment_labels.append(y_test_segment)
            else:
                print(f"  测试段 {test_seg_idx}: 警告，没有可用的模型进行预测")
        
        # 如果没有任何段的预测，跳过该 subject
        if len(all_segment_predictions) == 0:
            print(f"  跳过 Subject {test_subject_id}: 没有成功预测的段")
            continue
        
        # 合并所有段的预测和标签
        y_ensemble = np.concatenate(all_segment_predictions)
        y_test = np.concatenate(all_segment_labels)
        
        # 计算评估指标
        r2 = r2_score(y_test, y_ensemble)
        mae = mean_absolute_error(y_test, y_ensemble)
        
        # 计算 PCC
        if np.std(y_ensemble) < 1e-8:
            pcc = np.nan
            print(f"  预测为常数，PCC 设为 NaN")
        else:
            pcc, _ = pearsonr(y_test, y_ensemble)
        
        results.append({
            "subject_id": test_subject_id,
            "n_frames": len(y_test),
            "n_segments": len(all_segment_predictions),
            "n_models_used": top_k,
            "R2": r2,
            "MAE": mae,
            "PCC": pcc
        })
        
        print(f"  R² = {r2:.4f}, MAE = {mae:.4f}, PCC = {pcc if not np.isnan(pcc) else 0:.4f}")
        
        # 保存预测图表（如果启用）
        if save_plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_test, label="True SpO2", linewidth=1.5, color='black')
            plt.plot(y_ensemble, label=f"Ensemble Prediction (TOP_{top_k})", linewidth=1.5, color='red', linestyle='--')
            plt.title(f"Subject {test_subject_id} - Ensemble (TOP_K={top_k}, R²={r2:.3f}, MAE={mae:.2f}, PCC={pcc if not np.isnan(pcc) else 0:.3f})")
            plt.xlabel("Frame index")
            plt.ylabel("SpO2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(OUTPUT_DIR, f"vis_ensemble_{test_subject_id}_topk{top_k}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
    
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Ensemble 预测完成")
    print("=" * 60)
    return results_df

# 主程序逻辑
if SELECTED_CHANNELS == "ensemble":
    # 使用 ensemble model pool 进行预测
    print("\n" + "=" * 60)
    print("Ensemble Model Pool 模式")
    print("=" * 60)
    
    # 训练模型池（使用所有通道）
    model_pool = train_model_pool(None, feat_df_all, segment_length=SEGMENT_LENGTH)
    
    # 使用 ensemble 进行预测和评估
    results_ensemble_df = ensemble_predict_and_evaluate(
        model_pool, 
        feat_df_all, 
        selected_channels=None, 
        top_k=TOP_K, 
        segment_length=SEGMENT_LENGTH,
        save_plots=SAVE_PLOTS
    )
    
    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, f"results_ensemble_topk{TOP_K}_seg{SEGMENT_LENGTH}.csv")
    results_ensemble_df.to_csv(results_path, index=False)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("Ensemble 评估结果汇总:")
    print("=" * 60)
    print(f"总测试 subjects 数: {len(results_ensemble_df)}")
    print(f"平均 R²: {results_ensemble_df['R2'].mean():.4f}")
    print(f"平均 MAE: {results_ensemble_df['MAE'].mean():.4f}")
    print(f"平均 PCC: {results_ensemble_df['PCC'].mean():.4f}")
    print(f"\n结果已保存到: {results_path}")
    print("=" * 60)
    print("\n详细结果:")
    print(results_ensemble_df.to_string(index=False))

elif SELECTED_CHANNELS == "all_single":
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