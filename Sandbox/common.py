"""
共用模組：設定、特徵建構、相似度計算、模型池訓練與 Ensemble 推論。
供 train.py 與 infer.py 使用。
"""
import numpy as np
import pandas as pd
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy.fft import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============== Config ==============
WIN_LEN = 60
STEP = 1
OUTPUT_DIR = "./output"

# Normalization
USE_NORMALIZATION = False
# 正則化
USE_REGULARIZATION = False
ALPHA = 1.0

# Ensemble 模型池預設
TOP_K = 5
SEGMENT_LENGTH = 900

# rPPG / HR 相關參數
FS = 30                  # 取樣率 (fps)
HR_UPDATE_SEC = 15       # HR 重新計算間隔（秒）
MIN_RPPG_FRAMES = 512   # 穩定計算 HR 所需最少 rPPG frames
BW = 0.2                # 帶通濾波帶寬 ±Hz

# 模型池過濾閾值：訓練集 R² / PCC 低於此值的段不加入模型池
MIN_TRAIN_R2 = 0.5
MIN_TRAIN_PCC = 0.5

# 模型與設定儲存目錄（train.py 會寫入此目錄，infer.py 由此載入）
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")

# 通道
CHANNEL_ORDER = ["cg", "cr", "yiq_i", "dbdr_dr", "pos_y", "chrom_x"]
all_channel_names = ["R", "G", "B"]

# RGB 均值欄位（用於 model selection 時的 RGB 相似度）
rgb_mean_col_names = ["R_mean", "G_mean", "B_mean"]
# RGB 相似度在第二階段排序中的權重（與 range_score 加權組合）
RGB_SIM_WEIGHT = 0.5


def extract_subject_id(folder):
    """從 Folder 中提取 [] 內的 id"""
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


def getPOS(r_buf, g_buf, b_buf, win_len=30):
    epsilon = 1e-8
    if type(r_buf) is not np.ndarray or type(g_buf) is not np.ndarray or type(b_buf) is not np.ndarray:
        return False, np.array([])
    if r_buf.size == 0 or g_buf.size == 0 or b_buf.size == 0:
        return False, np.array([])

    ret_POS = np.zeros((1, r_buf.size), dtype=np.float32)
    C = np.vstack((r_buf, g_buf, b_buf)).astype(np.float32)
    color_base = np.array([[2, -1, -1], [0, -1, 1]], dtype=np.float32)

    for i in range(r_buf.size - win_len + 1):
        c_tmp = C[:, i:i+win_len]
        mean_c = np.mean(c_tmp, axis=1)
        if np.allclose(mean_c, 0):
            return False, np.array([])
        c_tmp = c_tmp / (mean_c[:, None] + epsilon)
        s = color_base @ c_tmp
        std_s0 = s[0, :].std()
        std_s1 = s[1, :].std()
        if np.isclose(std_s1, 0):
            std_s1 = epsilon
        alpha_base = np.array([[1, std_s0 / std_s1]], dtype=np.float32)
        p = (alpha_base @ s).flatten()

        mean_p = np.mean(p)
        std_p = np.std(p) if not np.isclose(np.std(p), 0) else epsilon
        ret_POS[0, i:i+win_len] += ((p - mean_p) / std_p).astype(np.float32)

    return True, ret_POS


def calculate_hr_score(signal, fps=30, freq_lower=0.5, freq_upper=3.0):
    """
    计算信号的 HR Score (Signal Quality Index)
    
    参数:
    signal: 输入的 rPPG 信号数组
    fps: 信号的采样率（帧率），默认为 30 fps
    freq_lower: 心率频率下限 (Hz)，默认为 0.5 Hz (30 bpm)
    freq_upper: 心率频率上限 (Hz)，默认为 3.0 Hz (180 bpm)
    
    返回:
    hr: 估计的心率 (bpm)
    sqi: 信号质量指数 (HR Score)，范围 0-1
    spectrum: 频谱数据
    frequencies: 对应的频率数组
    """
    signal_len = len(signal)
    
    # 执行 FFT
    fft_result = fft(signal)
    
    # 只取一半的频谱（由于对称性）
    half_len = signal_len // 2
    
    # 计算频率数组
    frequencies = np.arange(half_len) * fps / signal_len
    
    # 计算幅值谱
    magnitude_spectrum = np.abs(fft_result[:half_len])
    
    # 初始化变量
    band_w = 0
    sqi = 0.0
    max_mag = 0.0
    max_freq = 0.0
    hr_i = -1
    
    # 在有效频率范围内寻找峰值
    for i in range(half_len):
        freq = frequencies[i]
        
        if freq_lower <= freq <= freq_upper:
            mag = magnitude_spectrum[i]
            
            sqi += mag
            band_w += 1
            
            if max_mag < mag:
                hr_i = i
                max_mag = mag
                max_freq = freq
        else:
            # 将范围外的频率幅值设为0（滤波）
            magnitude_spectrum[i] = 0
    
    # 计算心率
    hr = 60.0 * max_freq
    
    # 计算信号质量指数 (SQI)
    if max_mag > np.finfo(float).eps and band_w > 0:
        sqi = 1.0 - (sqi / (max_mag * band_w))
    else:
        sqi = 0.0
    
    return hr, sqi, magnitude_spectrum, frequencies


def butter_filter(sig, fs=30, cutoff=0.1, btype='low', order=3):
    nyq = 0.5 * fs
    Wn = cutoff / nyq if np.isscalar(cutoff) else [c / nyq for c in cutoff]
    sos = butter(order, Wn, btype=btype, output='sos')
    return sosfiltfilt(sos, sig)


def build_features_from_df(df, source_name=None):
    """
    從單一 CSV 的 DataFrame 建特徵。
    流程：POS → rPPG → HR（每 HR_UPDATE_SEC 秒更新）→ 帶通濾波 RGB → 滑窗 AC/DC。
    前 MIN_RPPG_FRAMES 筆因無法穩定估計 HR 而丟棄。
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

        # 至少需要 MIN_RPPG_FRAMES + WIN_LEN 才能產出特徵
        if N < MIN_RPPG_FRAMES + WIN_LEN:
            print(f"    Skipped: insufficient frames ({N} < {MIN_RPPG_FRAMES + WIN_LEN})")
            continue

        # --- Step 1: 用 POS 算 rPPG --- 
        r_norm = R / 255.0
        g_norm = G / 255.0
        b_norm = B / 255.0
        success, raw_rppg = getPOS(r_norm, g_norm, b_norm, win_len=30)
        if not success:
            print(f"    Skipped: POS algorithm failed")
            continue

        # --- Step 2: 每隔 HR_UPDATE_SEC 秒計算一次 HR（用最近 MIN_RPPG_FRAMES 的 rPPG）---
        hr_update_interval = HR_UPDATE_SEC * FS
        hr_update_frames = list(range(MIN_RPPG_FRAMES, N, hr_update_interval))
        hr_list = []
        for uf in hr_update_frames:
            rppg_start = max(0, uf - MIN_RPPG_FRAMES)
            rppg_seg = raw_rppg[0, rppg_start:uf]
            hr, _, _, _ = calculate_hr_score(rppg_seg, FS)
            hr_list.append(hr)
        print(f"    N={N}, HR updates={len(hr_list)}, HR values={[f'{h:.1f}' for h in hr_list]}")

        # --- Step 3: 逐 epoch 帶通濾波 RGB ---
        r_filtered = np.zeros(N)
        g_filtered = np.zeros(N)
        b_filtered = np.zeros(N)

        epoch_edges = hr_update_frames + [N]
        for ei in range(len(epoch_edges) - 1):
            ep_start = epoch_edges[ei]
            ep_end = epoch_edges[ei + 1]
            hr = hr_list[ei]

            lo = hr / 60.0 - BW
            hi = hr / 60.0 + BW
            if lo <= 0:
                lo = 0.05  # 防止負頻率

            try:
                r_filt = butter_filter(R, fs=FS, cutoff=[lo, hi], btype='band')
                g_filt = butter_filter(G, fs=FS, cutoff=[lo, hi], btype='band')
                b_filt = butter_filter(B, fs=FS, cutoff=[lo, hi], btype='band')
            except Exception as e:
                print(f"    Warning: bandpass failed (HR={hr:.1f}, band=[{lo:.2f},{hi:.2f}]): {e}")
                continue

            # 第一個 epoch 往前延伸 WIN_LEN-1 筆，讓第一個滑窗完整覆蓋
            fill_start = max(0, ep_start - WIN_LEN + 1) if ei == 0 else ep_start
            r_filtered[fill_start:ep_end] = r_filt[fill_start:ep_end]
            g_filtered[fill_start:ep_end] = g_filt[fill_start:ep_end]
            b_filtered[fill_start:ep_end] = b_filt[fill_start:ep_end]

        # --- Step 4: 逐 frame 滑窗計算 AC/DC ---
        # 第一個輸出對應 frame MIN_RPPG_FRAMES（window 的最後一格）
        first_start = max(0, MIN_RPPG_FRAMES - WIN_LEN + 1)
        n_rows_before = len(rows)
        for start in range(first_start, N - WIN_LEN + 1, STEP):
            end = start + WIN_LEN
            y = float(SPO2[end - 1])

            row = {"subject_id": effective_id, "SPO2_win_mean": y}
            for ch_name, filt_arr, orig_arr in [("R", r_filtered, R),
                                                  ("G", g_filtered, G),
                                                  ("B", b_filtered, B)]:
                ac = np.std(filt_arr[start:end])
                dc = np.mean(orig_arr[start:end])
                row[f"{ch_name}_acdc"] = (ac / dc) if dc >= 1e-6 else 0.0
                row[f"{ch_name}_mean"] = float(orig_arr[end - 1])  # 存該幀的原始 RGB 值，段級再取平均
            rows.append(row)

        n_new = len(rows) - n_rows_before
        print(f"    Generated {n_new} rows (first output at frame {MIN_RPPG_FRAMES}, {MIN_RPPG_FRAMES} frames skipped)")
    return rows


def load_features_from_csv_paths(csv_paths, verbose=True):
    """
    從多個 CSV 路徑載入並合併特徵，回傳單一 DataFrame。
    """
    all_rows = []
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            if verbose:
                print(f"Skip (file not found): {csv_path}")
            continue
        if verbose:
            print(f"File: {csv_path}")
        df = pd.read_csv(csv_path)
        source_name = os.path.splitext(os.path.basename(csv_path))[0]
        all_rows.extend(build_features_from_df(df, source_name=source_name))
    feat_df = pd.DataFrame(all_rows)
    if verbose and len(feat_df) > 0:
        print(f"Feature building completed. Total samples: {len(feat_df)}")
        print(f"Total subjects: {feat_df['subject_id'].nunique()}")
    return feat_df


def calculate_ccc(y_true, y_pred):
    """計算 Concordance Correlation Coefficient (CCC)。"""
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return -1.0
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator < 1e-8:
        return -1.0
    ccc = numerator / denominator
    return ccc if not np.isnan(ccc) else -1.0


def calculate_similarity(features_model, features_test, weights_model):
    """
    計算模型與測試樣本之間的 Shape Similarity 和 Range Similarity。
    回傳 (shape_score, range_score)。
    """
    if isinstance(features_model, pd.DataFrame):
        features_model = features_model.values
    if isinstance(features_test, pd.DataFrame):
        features_test = features_test.values
    if features_model.shape != features_test.shape:
        print(f"  警告: 特徵維度不匹配 {features_model.shape} vs {features_test.shape}")
        return -1.0, -1.0
    n_samples, n_features = features_model.shape
    if len(weights_model) != n_features:
        print(f"  警告: 特徵數 {n_features} 與權重數 {len(weights_model)} 不一致")
        return -1.0, -1.0
    abs_weights = np.abs(weights_model)
    weight_sum = np.sum(abs_weights)
    if weight_sum < 1e-8:
        return -1.0, -1.0
    normalized_weights = abs_weights / weight_sum

    shape_scores = []
    for i in range(n_features):
        model_channel = features_model[:, i]
        model_mean, model_std = np.mean(model_channel), np.std(model_channel)
        test_channel = features_test[:, i]
        test_mean, test_std = np.mean(test_channel), np.std(test_channel)
        if model_std < 1e-8 or test_std < 1e-8:
            shape_scores.append(-1.0)
            continue
        model_z = (model_channel - model_mean) / model_std
        test_z = (test_channel - test_mean) / test_std
        try:
            corr, _ = pearsonr(model_z, test_z)
            shape_scores.append(corr if not np.isnan(corr) else -1.0)
        except Exception:
            shape_scores.append(-1.0)
    shape_score = np.sum(np.array(shape_scores) * normalized_weights)

    range_scores = []
    for i in range(n_features):
        ccc = calculate_ccc(features_model[:, i], features_test[:, i])
        range_scores.append(ccc)
    range_score = np.sum(np.array(range_scores) * normalized_weights)
    return shape_score, range_score


def calculate_rgb_similarity(rgb_mean_a, rgb_mean_b):
    """
    計算兩個段的 RGB 均值向量的餘弦相似度。
    rgb_mean_a, rgb_mean_b: shape (3,)，分別為 [mean_R, mean_G, mean_B]。
    回傳 cosine similarity ∈ [0, 1]（因為 RGB 值恆正）。
    """
    a = np.asarray(rgb_mean_a, dtype=np.float64)
    b = np.asarray(rgb_mean_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(cos_sim, 0.0, 1.0))


def train_model_pool(selected_channels, feat_df, segment_length=SEGMENT_LENGTH,
                     use_normalization=USE_NORMALIZATION, use_regularization=USE_REGULARIZATION, alpha=ALPHA,
                     min_train_r2=MIN_TRAIN_R2, min_train_pcc=MIN_TRAIN_PCC):
    """
    訓練所有 subject 的模型並建構模型池（使用段切分策略）。
    回傳 model_pool: list of dict（每項含 subject_id, segment_id, model_id, model, feature_cols, weights, bias, means, stds, X_raw_segment）。
    """
    print("\n" + "=" * 60)
    print(f"訓練模型池（段長度={segment_length}）...")
    print("=" * 60)
    if selected_channels is None:
        used_channels = all_channel_names
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
    feature_cols = [f"{ch}_acdc" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols + rgb_mean_col_names].copy()

    model_pool = []
    total_segments = 0
    skipped_subjects = 0
    filtered_by_pcc = 0
    filtered_by_r2 = 0

    for subject_id, subdf in feat_df_selected.groupby("subject_id", sort=False):
        n_samples = len(subdf)
        print(f"\n處理 Subject {subject_id}: {n_samples} 個樣本")
        if n_samples < segment_length:
            print(f"  跳過: 樣本數不足 {segment_length}，實際只有 {n_samples}")
            skipped_subjects += 1
            continue
        n_segments = n_samples // segment_length
        print(f"  可切分為 {n_segments} 段")
        X_full = subdf[feature_cols].astype(float)
        rgb_mean_full = subdf[rgb_mean_col_names].astype(float)
        y_full = subdf["SPO2_win_mean"].values.astype(float)

        for seg_idx in range(n_segments):
            start_idx = seg_idx * segment_length
            end_idx = start_idx + segment_length
            X_raw_segment = X_full.iloc[start_idx:end_idx].copy()
            y_segment = y_full[start_idx:end_idx]
            # 段級 RGB 均值向量（用於 model selection）
            rgb_mean_segment = rgb_mean_full.iloc[start_idx:end_idx].mean(axis=0).values
            stds = X_raw_segment.std(axis=0)
            if (stds < 1e-8).any():
                constant_cols = X_raw_segment.columns[stds < 1e-8].tolist()
                print(f"  段 {seg_idx}: 跳過（有常數特徵: {constant_cols}）")
                continue

            if use_normalization:
                means = X_raw_segment.mean(axis=0)
                stds = X_raw_segment.std(axis=0)
                X_normalized = (X_raw_segment - means) / stds
            else:
                means = None
                stds = None
                X_normalized = X_raw_segment

            if use_regularization:
                from sklearn.linear_model import Ridge
                lr = Ridge(alpha=alpha)
            else:
                lr = LinearRegression()
            lr.fit(X_normalized, y_segment)
            y_pred = lr.predict(X_normalized)
            train_r2 = r2_score(y_segment, y_pred)
            if np.isnan(train_r2):
                train_r2 = -1.0
            if np.std(y_pred) < 1e-8 or np.std(y_segment) < 1e-8:
                train_pcc = -1.0
            else:
                train_pcc, _ = pearsonr(y_segment, y_pred)
                train_pcc = train_pcc if not np.isnan(train_pcc) else -1.0

            if train_r2 <= min_train_r2:
                print(f"  段 {seg_idx}: 跳過（訓練集 R² = {train_r2:.4f} <= {min_train_r2}）")
                filtered_by_r2 += 1
                continue
            if train_pcc <= min_train_pcc:
                print(f"  段 {seg_idx}: 跳過（訓練集 PCC = {train_pcc:.4f} <= {min_train_pcc}）")
                filtered_by_pcc += 1
                continue

            model_id = f"{subject_id}_seg{seg_idx}"
            model_pool.append({
                'subject_id': subject_id,
                'segment_id': seg_idx,
                'model_id': model_id,
                'model': lr,
                'feature_cols': X_raw_segment.columns.tolist(),
                'weights': lr.coef_,
                'bias': lr.intercept_,
                'means': means.values if means is not None else None,
                'stds': stds.values if stds is not None else None,
                'X_raw_segment': X_raw_segment.values,
                'rgb_mean_segment': rgb_mean_segment,  # 段級 RGB 均值 (3,)
            })
            total_segments += 1
        print(f"  Subject {subject_id} 完成: 訓練了 {n_segments} 個模型")

    print(f"\n模型池建構完成:")
    print(f"  最終模型池大小: {len(model_pool)} 個模型")
    print("=" * 60)
    return model_pool


def ensemble_predict_and_evaluate(model_pool, feat_df, selected_channels, top_k,
                                  segment_length=SEGMENT_LENGTH,
                                  use_normalization=USE_NORMALIZATION,
                                  save_plots=False, output_dir=OUTPUT_DIR):
    """
    使用 ensemble 模型池進行預測與評估。回傳 (results_df, overall_pcc)。
    """
    print("\n" + "=" * 60)
    print(f"使用 Ensemble 模型池進行預測 (TOP_K={top_k}, 段長度={segment_length})...")
    print("=" * 60)
    if selected_channels is None:
        used_channels = all_channel_names
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
    feature_cols = [f"{ch}_acdc" for ch in used_channels]
    feat_df_selected = feat_df[["subject_id", "SPO2_win_mean"] + feature_cols + rgb_mean_col_names].copy()

    results = []
    all_subjects_predictions = []
    all_subjects_labels = []

    for test_subject_id, test_subdf in feat_df_selected.groupby("subject_id", sort=False):
        n_samples = len(test_subdf)
        print(f"\n測試 Subject {test_subject_id}: {n_samples} 個樣本")
        if n_samples < segment_length:
            print(f"  跳過: 樣本數不足 {segment_length}")
            continue
        X_test_full = test_subdf[feature_cols].astype(float)
        rgb_mean_test_full = test_subdf[rgb_mean_col_names].astype(float)
        y_test_full = test_subdf["SPO2_win_mean"].values.astype(float)
        n_test_segments = n_samples // segment_length
        print(f"  可切分為 {n_test_segments} 段")

        all_segment_predictions = []
        all_segment_labels = []

        for test_seg_idx in range(n_test_segments):
            start_idx = test_seg_idx * segment_length
            end_idx = start_idx + segment_length
            X_test_segment_df = X_test_full.iloc[start_idx:end_idx].copy()
            y_test_segment = y_test_full[start_idx:end_idx]
            test_stds = X_test_segment_df.std(axis=0)
            if (test_stds < 1e-8).any():
                constant_cols = X_test_segment_df.columns[test_stds < 1e-8].tolist()
                print(f"  測試段 {test_seg_idx}: 跳過（有常數特徵: {constant_cols}）")
                continue
            X_test_segment_array = X_test_segment_df.values
            # 測試段的 RGB 均值向量
            rgb_mean_test_segment = rgb_mean_test_full.iloc[start_idx:end_idx].mean(axis=0).values
            similarities = []
            for model_info in model_pool:
                if model_info['subject_id'] == test_subject_id:
                    continue
                X_train_segment = model_info['X_raw_segment']
                weights = model_info['weights']
                shape_score, range_score = calculate_similarity(X_train_segment, X_test_segment_array, weights)
                rgb_score = calculate_rgb_similarity(model_info['rgb_mean_segment'], rgb_mean_test_segment)
                similarities.append({
                    'model_info': model_info,
                    'shape_score': shape_score,
                    'range_score': range_score,
                    'rgb_score': rgb_score,
                    'model_id': model_info['model_id']
                })
            if len(similarities) == 0:
                print(f"  測試段 {test_seg_idx}: 沒有可用模型")
                continue

            M = 2 * top_k
            similarities.sort(key=lambda x: x['shape_score'], reverse=True)
            top_m_candidates = similarities[:min(M, len(similarities))]
            # 第二階段：range_score + RGB_SIM_WEIGHT * rgb_score 組合排序
            top_m_candidates.sort(key=lambda x: x['range_score'] + RGB_SIM_WEIGHT * x['rgb_score'], reverse=True)
            top_k_models = top_m_candidates[:min(top_k, len(top_m_candidates))]
            if test_seg_idx == 0:
                print(f"  測試段 {test_seg_idx} 選擇的 TOP_{min(top_k, len(top_k_models))} 模型:")
                for rank, sim_info in enumerate(top_k_models, 1):
                    print(f"    {rank}. {sim_info['model_id']}, shape: {sim_info['shape_score']:.4f}, range: {sim_info['range_score']:.4f}, rgb: {sim_info['rgb_score']:.4f}")

            if use_normalization:
                test_means = X_test_segment_df.mean(axis=0)
                test_stds = X_test_segment_df.std(axis=0)
                X_test_normalized = (X_test_segment_df - test_means) / test_stds
            else:
                X_test_normalized = X_test_segment_df

            segment_predictions = []
            for sim_info in top_k_models:
                model_info = sim_info['model_info']
                weights = model_info['weights']
                bias = model_info['bias']
                feature_cols_model = model_info['feature_cols']
                X_test_normalized_aligned = X_test_normalized[feature_cols_model].values
                y_pred = np.dot(X_test_normalized_aligned, weights) + bias
                segment_predictions.append(y_pred)
            if len(segment_predictions) > 0:
                y_segment_ensemble = np.mean(segment_predictions, axis=0)
                all_segment_predictions.append(y_segment_ensemble)
                all_segment_labels.append(y_test_segment)

        if len(all_segment_predictions) == 0:
            print(f"  跳過 Subject {test_subject_id}: 沒有成功預測的段")
            continue

        y_ensemble = np.concatenate(all_segment_predictions)
        y_test = np.concatenate(all_segment_labels)
        r2 = r2_score(y_test, y_ensemble)
        mae = mean_absolute_error(y_test, y_ensemble)
        pcc = np.nan if np.std(y_ensemble) < 1e-8 else pearsonr(y_test, y_ensemble)[0]
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
        all_subjects_predictions.append(y_ensemble)
        all_subjects_labels.append(y_test)

        if save_plots:
            plot_dir = os.path.join(output_dir, "sub_plot")
            os.makedirs(plot_dir, exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.plot(y_test, label="True SpO2", linewidth=1.5, color='black')
            plt.plot(y_ensemble, label=f"Ensemble Prediction (TOP_{top_k})", linewidth=1.5, color='red', linestyle='--')
            plt.title(f"Subject {test_subject_id} - Ensemble (TOP_K={top_k}, R²={r2:.3f}, MAE={mae:.2f}, PCC={pcc if not np.isnan(pcc) else 0:.3f})")
            plt.xlabel("Frame index")
            plt.ylabel("SpO2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            output_path = os.path.join(plot_dir, f"vis_ensemble_{test_subject_id}_topk{top_k}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()

    results_df = pd.DataFrame(results)
    if len(all_subjects_predictions) > 0:
        all_pred = np.concatenate(all_subjects_predictions)
        all_labels = np.concatenate(all_subjects_labels)
        overall_pcc = np.nan if np.std(all_pred) < 1e-8 else pearsonr(all_labels, all_pred)[0]
        if not np.isnan(overall_pcc):
            print(f"\n所有 subject 串聯後的整體 PCC: {overall_pcc:.6f}")
            print(f"總樣本數: {len(all_pred)}")
    else:
        overall_pcc = np.nan
    results_df['overall_PCC'] = overall_pcc
    print("\n" + "=" * 60)
    print("Ensemble 預測完成")
    print("=" * 60)
    return results_df
