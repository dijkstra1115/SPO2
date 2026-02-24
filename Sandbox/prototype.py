import numpy as np
import pandas as pd
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 tkinter 错误
import matplotlib.pyplot as plt

# Config
WIN_LEN = 60
STEP = 1
OUTPUT_DIR = "./output"  # 输出目录
SAVE_PLOTS = False  # 是否保存每个 subject 的预测图

# Normalization 参数
USE_NORMALIZATION = False  # 是否使用归一化

# 正则化参数
USE_REGULARIZATION = False  # 是否使用正则化
ALPHA = 1.0  # 正则化强度 (alpha)

# Ensemble 模型池配置
TOP_K = 5  # 从其他模型中选择 TOP_K 个最相似的模型进行 ensemble
SEGMENT_LENGTH = 900  # 每个模型训练时使用的固定样本数（段长度）

# 多 CSV 擴大模型池：可一次加入多個 .csv，合併後訓練與評估
DATA_CSV_PATHS = [
    "./data/prc-c920.csv",
    "./data/prc-i15.csv",
    "./data/prc-i15m.csv",
]

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

# getsixchannels 回傳的固定順序，用於依名稱取對應通道
CHANNEL_ORDER = ["cg", "cr", "yiq_i", "dbdr_dr", "pos_y", "chrom_x"]
all_channel_names = ["cg", "cr", "yiq_i"]


def build_features_from_df(df, source_name=None):
    """
    從單一 CSV 的 DataFrame 建特徵。若給定 source_name，會將 subject_id 前綴為 source_name_subject_id 以區分不同檔案。
    回傳 list of dict（每筆一 row）。
    """
    df = df.copy()
    df["Subject"] = df["Folder"].apply(extract_subject_id)
    rows = []
    for subject_id, g in df.groupby("Subject", sort=False):
        effective_id = f"{source_name}_{subject_id}" if source_name else subject_id
        print(f"  Processing subject {effective_id}")
        R = g["COLOR_R"].values.astype(float)
        G = g["COLOR_G"].values.astype(float)
        B = g["COLOR_B"].values.astype(float)
        SPO2 = g["SPO2"].values.astype(float)
        N = len(g)
        if N < WIN_LEN:
            continue
        rgb = np.vstack([R, G, B])
        six = getsixchannels(rgb)
        for start in range(0, N - WIN_LEN + 1, STEP):
            end = start + WIN_LEN
            y = float(SPO2[start:end].mean())
            row = {"subject_id": effective_id, "SPO2_win_mean": y}
            for name in all_channel_names:
                six_idx = CHANNEL_ORDER.index(name)
                ch_segment = six[six_idx][start:end]
                dc = ch_segment.mean()
                ac = ch_segment.std()
                val = (ac / dc) if dc >= 1e-6 else 0
                row[f"{name}_acdc"] = val
            rows.append(row)
    return rows


# 載入多個 CSV、合併特徵，擴大 model pool
print("=" * 60)
print("Loading CSV(s) and building features...")
print("=" * 60)
all_rows = []
for csv_path in DATA_CSV_PATHS:
    if not os.path.isfile(csv_path):
        print(f"Skip (file not found): {csv_path}")
        continue
    print(f"File: {csv_path}")
    df = pd.read_csv(csv_path)
    source_name = os.path.splitext(os.path.basename(csv_path))[0]
    all_rows.extend(build_features_from_df(df, source_name=source_name))

feat_df_all = pd.DataFrame(all_rows)
print(f"Feature building completed. Total samples: {len(feat_df_all)}")
print(f"Total subjects: {feat_df_all['subject_id'].nunique()}")
print("=" * 60)

def calculate_ccc(y_true, y_pred):
    """
    计算 Concordance Correlation Coefficient (CCC)
    
    Parameters:
    -----------
    y_true : ndarray
        真实值
    y_pred : ndarray
        预测值
    
    Returns:
    --------
    ccc : float
        CCC 值（-1 到 1 之间，越接近 1 越好）
        如果计算失败，返回 -1
    """
    # 检查是否有常数向量
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return -1.0
    
    # 计算均值
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # 计算方差
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # 计算协方差
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # 计算 CCC
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator < 1e-8:
        return -1.0
    
    ccc = numerator / denominator
    return ccc if not np.isnan(ccc) else -1.0


def calculate_similarity(features_model, features_test, weights_model):
    """
    计算模型与测试样本之间的 Shape Similarity 和 Range Similarity
    
    【Shape Similarity】（主要条件）
    - 指标：Pearson correlation
    - 资料：z-score 后的 feature
    - normalization 规则：
      - model 的 feature → 用 model 自己的 mean/std
      - test subject 的 feature → 用 test subject 自己的 mean/std
    - 对 6 个通道各自算 Pearson
    - 用 |w_s| 对 6 个通道的 Pearson 加权平均
    
    【Range Similarity】（辅助条件）
    - 指标：CCC（Concordance Correlation Coefficient）
    - 资料：raw feature（不做 z-score）
    - 对 6 个通道各自算 CCC
    - 同样用 |w_s| 加权平均
    
    Parameters:
    -----------
    features_model : ndarray
        模型的原始特征 (segment_length, n_features=6)
    features_test : ndarray
        测试样本的原始特征 (segment_length, n_features=6)
    weights_model : ndarray
        模型的权重 (n_features=6,)，用于加权
    
    Returns:
    --------
    shape_score : float
        Shape similarity 分数（-1 到 1 之间，越大越相似）
    range_score : float
        Range similarity 分数（-1 到 1 之间，越大越相似）
    """
    # 转换为 numpy array
    if isinstance(features_model, pd.DataFrame):
        features_model = features_model.values
    if isinstance(features_test, pd.DataFrame):
        features_test = features_test.values
    
    # 检查维度是否相同
    if features_model.shape != features_test.shape:
        print(f"  警告: 特征维度不匹配 {features_model.shape} vs {features_test.shape}")
        return -1.0, -1.0
    
    n_samples, n_features = features_model.shape
    
    # 特征数量需与权重一致（支持任意通道数，不限于 6）
    if len(weights_model) != n_features:
        print(f"  警告: 特征数 {n_features} 与权重数 {len(weights_model)} 不一致")
        return -1.0, -1.0
    
    # 计算权重的绝对值（用于加权平均）
    abs_weights = np.abs(weights_model)
    weight_sum = np.sum(abs_weights)
    
    if weight_sum < 1e-8:
        print(f"  警告: 权重总和接近 0")
        return -1.0, -1.0
    
    # 归一化权重
    normalized_weights = abs_weights / weight_sum
    
    # === 1. 计算 Shape Similarity ===
    shape_scores = []
    for i in range(n_features):
        # 对模型特征做 z-score（用模型自己的 mean/std）
        model_channel = features_model[:, i]
        model_mean = np.mean(model_channel)
        model_std = np.std(model_channel)
        
        # 对测试特征做 z-score（用测试自己的 mean/std）
        test_channel = features_test[:, i]
        test_mean = np.mean(test_channel)
        test_std = np.std(test_channel)
        
        # 检查是否有常数通道
        if model_std < 1e-8 or test_std < 1e-8:
            shape_scores.append(-1.0)
            continue
        
        # Z-score normalization
        model_channel_zscore = (model_channel - model_mean) / model_std
        test_channel_zscore = (test_channel - test_mean) / test_std
        
        # 计算 Pearson correlation
        try:
            corr, _ = pearsonr(model_channel_zscore, test_channel_zscore)
            shape_scores.append(corr if not np.isnan(corr) else -1.0)
        except Exception as e:
            shape_scores.append(-1.0)
    
    # 加权平均
    shape_score = np.sum(np.array(shape_scores) * normalized_weights)
    
    # === 2. 计算 Range Similarity ===
    range_scores = []
    for i in range(n_features):
        # 直接使用原始特征（不做 z-score）
        model_channel = features_model[:, i]
        test_channel = features_test[:, i]
        
        # 计算 CCC
        ccc = calculate_ccc(model_channel, test_channel)
        range_scores.append(ccc)
    
    # 加权平均
    range_score = np.sum(np.array(range_scores) * normalized_weights)
    
    return shape_score, range_score

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
    
    # 选择特征列（特征名为 通道_acdc）
    feature_cols = [f"{ch}_acdc" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols].copy()
    
    model_pool = []
    total_segments = 0
    skipped_subjects = 0
    filtered_by_pcc = 0  # 因 PCC <= 0.3 被过滤的模型数
    filtered_by_r2 = 0  # 因 R² <= 0.1 被过滤的模型数
    
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
            
            # 训练后立即评估：计算预测值与真实值的 PCC 和 R²
            y_pred = lr.predict(X_normalized)
            
            # 计算 R²
            train_r2 = r2_score(y_segment, y_pred)
            if np.isnan(train_r2):
                train_r2 = -1.0
            
            # 计算 PCC（Pearson correlation coefficient）
            if np.std(y_pred) < 1e-8 or np.std(y_segment) < 1e-8:
                # 如果预测值或真实值为常数，PCC 设为 -1（会被过滤）
                train_pcc = -1.0
            else:
                train_pcc, _ = pearsonr(y_segment, y_pred)
                if np.isnan(train_pcc):
                    train_pcc = -1.0
            
            # 筛选条件：同时检查 R² > 0.3 和 PCC > 0.5
            if train_r2 <= 0.3:
                print(f"  段 {seg_idx}: 跳过（训练集 R² = {train_r2:.4f} <= 0.3）")
                filtered_by_r2 += 1
                continue
            
            if train_pcc <= 0.5:
                print(f"  段 {seg_idx}: 跳过（训练集 PCC = {train_pcc:.4f} <= 0.5）")
                filtered_by_pcc += 1
                continue
            
            # 保存到模型池
            # 注意：所有模型都使用全部特征，维度固定为 (segment_length, n_features)
            model_id = f"{subject_id}_seg{seg_idx}"
            model_pool.append({
                'subject_id': subject_id,
                'segment_id': seg_idx,
                'model_id': model_id,
                'model': lr,
                'feature_cols': X_raw_segment.columns.tolist(),  # 全部特征列
                'weights': lr.coef_,  # 模型权重 (n_features,)
                'bias': lr.intercept_,  # 模型偏置
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
    print(f"  训练的总段数: {total_segments + filtered_by_pcc + filtered_by_r2}")
    print(f"  因 R² <= 0.3 被过滤: {filtered_by_r2} 个模型")
    print(f"  因 PCC <= 0.5 被过滤: {filtered_by_pcc} 个模型")
    print(f"  最终模型池大小: {len(model_pool)} 个模型")
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
    
    feature_cols = [f"{ch}_acdc" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols].copy()
    
    results = []
    
    # 用于存储所有 subject 的预测结果和真实值（用于计算整体 PCC）
    all_subjects_predictions = []
    all_subjects_labels = []
    
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
                weights = model_info['weights']
                
                # 计算相似度（返回 shape_score 和 range_score）
                # 现在两个矩阵维度应该完全一致 (segment_length, n_features)
                shape_score, range_score = calculate_similarity(X_train_segment, X_test_segment_array, weights)
                similarities.append({
                    'model_info': model_info,
                    'shape_score': shape_score,
                    'range_score': range_score,
                    'model_id': model_id
                })
            
            if len(similarities) == 0:
                print(f"  测试段 {test_seg_idx}: 没有可用的模型")
                continue
            
            # 【两层筛选】
            # 第 1 层：按 shape_score 排序，取前 M = 2*top_k
            M = 2 * top_k
            similarities.sort(key=lambda x: x['shape_score'], reverse=True)
            top_m_candidates = similarities[:min(M, len(similarities))]
            
            # 第 2 层：在这 M 个候选中，按 range_score 排序，取前 TOP_K
            top_m_candidates.sort(key=lambda x: x['range_score'], reverse=True)
            top_k_models = top_m_candidates[:min(top_k, len(top_m_candidates))]
            
            # 打印选择的模型
            if test_seg_idx == 0:  # 只在第一段打印详细信息
                print(f"  测试段 {test_seg_idx} 选择的 TOP_{min(top_k, len(top_k_models))} 模型:")
                for rank, sim_info in enumerate(top_k_models, 1):
                    print(f"    {rank}. {sim_info['model_id']}, shape: {sim_info['shape_score']:.4f}, range: {sim_info['range_score']:.4f}")
            
            # 【关键修改】所有模型在「test subject 的座标系」下预测
            # Step 1: 计算 test subject 自己的 mean 和 std
            # Step 2: 用 test subject 的 mean/std 做 normalization
            if USE_NORMALIZATION:
                test_means = X_test_segment_df.mean(axis=0)
                test_stds = X_test_segment_df.std(axis=0)
                X_test_normalized = (X_test_segment_df - test_means) / test_stds
            else:
                X_test_normalized = X_test_segment_df
            
            # 使用 TOP_K 模型进行预测
            segment_predictions = []
            for sim_info in top_k_models:
                model_info = sim_info['model_info']
                weights = model_info['weights']
                bias = model_info['bias']
                feature_cols_model = model_info['feature_cols']
                
                # 确保特征顺序与模型训练时一致
                X_test_normalized_aligned = X_test_normalized[feature_cols_model].values
                
                # Step 3: 手动计算预测值 y_hat = w^T x_norm + b
                # 所有模型都用 test subject normalized 的 features
                y_pred = np.dot(X_test_normalized_aligned, weights) + bias
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
        
        # 将当前 subject 的预测结果和真实值添加到整体列表中
        all_subjects_predictions.append(y_ensemble)
        all_subjects_labels.append(y_test)
        
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
    
    # 计算所有 subject 串联后的整体 PCC
    if len(all_subjects_predictions) > 0:
        all_predictions_concatenated = np.concatenate(all_subjects_predictions)
        all_labels_concatenated = np.concatenate(all_subjects_labels)
        
        if np.std(all_predictions_concatenated) < 1e-8:
            overall_pcc = np.nan
            print(f"\n整体预测为常数，整体 PCC 设为 NaN")
        else:
            overall_pcc, _ = pearsonr(all_labels_concatenated, all_predictions_concatenated)
            print(f"\n所有 subject 串联后的整体 PCC: {overall_pcc:.6f}")
            print(f"总样本数: {len(all_predictions_concatenated)}")
    else:
        overall_pcc = np.nan
        print("\n没有有效的预测结果，无法计算整体 PCC")
    
    print("\n" + "=" * 60)
    print("Ensemble 预测完成")
    print("=" * 60)
    
    # 将整体 PCC 添加到结果中
    results_df['overall_PCC'] = overall_pcc
    
    return results_df

# 主程序逻辑 - Ensemble Model Pool 模式
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
print(f"平均 PCC (每个 subject): {results_ensemble_df['PCC'].mean():.4f}")
if 'overall_PCC' in results_ensemble_df.columns and not pd.isna(results_ensemble_df['overall_PCC'].iloc[0]):
    print(f"整体 PCC (所有 subject 串联): {results_ensemble_df['overall_PCC'].iloc[0]:.6f}")
print(f"\n结果已保存到: {results_path}")
print("=" * 60)
print("\n详细结果:")
print(results_ensemble_df.to_string(index=False))