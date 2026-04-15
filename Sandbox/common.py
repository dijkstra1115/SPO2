"""
共用模組：設定、特徵建構、相似度計算、模型池訓練與 Ensemble 推論。
供 train.py 與 infer.py 使用。
"""
import numpy as np
import pandas as pd
import re
import os
import math
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
USE_NORMALIZATION = True
# 正則化
USE_REGULARIZATION = True
ALPHA = 1.0

# Ensemble 模型池預設
TOP_K = 25
SEGMENT_LENGTH = 900

# rPPG / HR 相關參數
FS = 30                  # 取樣率 (fps)
HR_UPDATE_SEC = 15       # HR 重新計算間隔（秒）
MIN_RPPG_FRAMES = 512   # 穩定計算 HR 所需最少 rPPG frames
BW = 0.2                # 帶通濾波帶寬 ±Hz

# 模型池過濾閾值：訓練集 R2 / PCC 低於此值的段不加入模型池
MIN_TRAIN_R2 = 0.2
MIN_TRAIN_PCC = 0.2

# Subject 級 SpO2 跨幅篩選：僅保留「該 subject 所有 segment 串起來後」SpO2 最大最小值差值 >= 此值的 subject
MIN_SPO2_RANGE = 10

# 模型與設定儲存目錄（train.py 會寫入此目錄，infer.py 由此載入）
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")

# 通道
CHANNEL_ORDER = ["cg", "cr", "yiq_i", "dbdr_dr", "pos_y", "chrom_x"]
all_channel_names = ["R", "G", "B", "RoR_RG", "RoR_RB"]

# RGB 均值欄位（用於 model selection 時的 RGB 相似度）
rgb_mean_col_names = ["R_mean", "G_mean", "B_mean"]
pool_base_feature_cols = [
    "R_acdc", "G_acdc", "B_acdc",
    "RoR_RG_acdc", "RoR_RB_acdc",
]
pool_extended_feature_cols = pool_base_feature_cols + [
    "POS_Y_acdc", "CHROM_X_acdc",
    "R_acdc_long", "G_acdc_long", "B_acdc_long",
    "RoR_RG_acdc_long", "RoR_RB_acdc_long",
    "delta_R_acdc", "delta_G_acdc", "delta_B_acdc",
    "acdc_ratio_long_short_R", "acdc_ratio_long_short_G", "acdc_ratio_long_short_B",
    "sqi",
]
USE_EXTENDED_POOL_FEATURES = True
# Additional derived features for global model only
derived_acdc_col_names = ["POS_Y_acdc", "CHROM_X_acdc",
                          "R_acdc_long", "G_acdc_long", "B_acdc_long",
                          "RoR_RG_acdc_long", "RoR_RB_acdc_long",
                          "delta_R_acdc", "delta_G_acdc", "delta_B_acdc",
                          "acdc_ratio_long_short_R", "acdc_ratio_long_short_G", "acdc_ratio_long_short_B",
                          "sqi"]
# RGB 相似度在第二階段排序中的權重（與 range_score 加權組合）
RGB_SIM_WEIGHT = 0.5


def get_pool_feature_cols(selected_channels=None):
    if USE_EXTENDED_POOL_FEATURES and selected_channels is None:
        return list(pool_extended_feature_cols)

    if selected_channels is None:
        used_channels = all_channel_names
    else:
        used_channels = [ch for ch in selected_channels if ch in all_channel_names]
    return [f"{ch}_acdc" for ch in used_channels]


def dedupe_preserve_order(columns):
    return list(dict.fromkeys(columns))


def resolve_column_name(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        match = lowered.get(str(name).strip().lower())
        if match is not None:
            return match
    raise KeyError(f"Missing required column. Tried: {candidates}")


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
    df = df.copy()
    df["Subject"] = df["Folder"].apply(extract_subject_id)
    spo2_col = resolve_column_name(df, ["SPO2", "SPo2", "SpO2", "spo2", "SpO2 Score", "SPO2 Score"])
    rows = []
    
    # 【修改 1】：改為依照 Folder 分組處理，避免訊號濾波跨越片段
    for folder_name, g in df.groupby("Folder", sort=False):
        subject_id = g["Subject"].iloc[0]
        effective_id = f"{source_name}_{subject_id}" if source_name else subject_id
        # Dataset group for cross-device leakage prevention (e.g., prc-c920 -> prc)
        dataset_group = source_name.split('-')[0] if source_name else ""
        subject_group = f"{dataset_group}_{subject_id}" if dataset_group else subject_id
        print(f"  Processing folder {folder_name} (Subject {effective_id}, group {subject_group})")
        
        R = g["COLOR_R"].values.astype(float)
        G = g["COLOR_G"].values.astype(float)
        B = g["COLOR_B"].values.astype(float)
        SPO2 = g[spo2_col].values.astype(float)
        N = len(g)

        if N < MIN_RPPG_FRAMES + WIN_LEN:
            print(f"    Skipped: insufficient frames in folder ({N})")
            continue

        r_norm = R / 255.0
        g_norm = G / 255.0
        b_norm = B / 255.0
        success, raw_rppg = getPOS(r_norm, g_norm, b_norm, win_len=30)
        if not success:
            print(f"    Skipped: POS algorithm failed")
            continue

        hr_update_interval = HR_UPDATE_SEC * FS
        hr_update_frames = list(range(MIN_RPPG_FRAMES, N, hr_update_interval))
        hr_list = []
        for uf in hr_update_frames:
            rppg_start = max(0, uf - MIN_RPPG_FRAMES)
            rppg_seg = raw_rppg[0, rppg_start:uf]
            hr, _, _, _ = calculate_hr_score(rppg_seg, FS)
            hr_list.append(hr)

        r_filtered = np.zeros(N)
        g_filtered = np.zeros(N)
        b_filtered = np.zeros(N)
        # Derived channels
        pos_y_raw = -2 * R + G + B
        chrom_x_raw = 3 * R - 2 * G
        pos_y_filtered = np.zeros(N)
        chrom_x_filtered = np.zeros(N)

        epoch_edges = hr_update_frames + [N]
        for ei in range(len(epoch_edges) - 1):
            ep_start = epoch_edges[ei]
            ep_end = epoch_edges[ei + 1]
            hr = hr_list[ei]
            lo = max(0.05, hr / 60.0 - BW)
            hi = hr / 60.0 + BW

            try:
                r_filt = butter_filter(R, fs=FS, cutoff=[lo, hi], btype='band')
                g_filt = butter_filter(G, fs=FS, cutoff=[lo, hi], btype='band')
                b_filt = butter_filter(B, fs=FS, cutoff=[lo, hi], btype='band')
                pos_y_filt = butter_filter(pos_y_raw, fs=FS, cutoff=[lo, hi], btype='band')
                chrom_x_filt = butter_filter(chrom_x_raw, fs=FS, cutoff=[lo, hi], btype='band')
            except Exception as e:
                continue

            fill_start = max(0, ep_start - WIN_LEN + 1) if ei == 0 else ep_start
            r_filtered[fill_start:ep_end] = r_filt[fill_start:ep_end]
            g_filtered[fill_start:ep_end] = g_filt[fill_start:ep_end]
            b_filtered[fill_start:ep_end] = b_filt[fill_start:ep_end]
            pos_y_filtered[fill_start:ep_end] = pos_y_filt[fill_start:ep_end]
            chrom_x_filtered[fill_start:ep_end] = chrom_x_filt[fill_start:ep_end]

        first_start = max(0, MIN_RPPG_FRAMES - WIN_LEN + 1)
        for start in range(first_start, N - WIN_LEN + 1, STEP):
            end = start + WIN_LEN
            y = float(SPO2[end - 1])

            # 【修改 2】：將 folder_name 也存入特徵中
            row = {
                "subject_id": effective_id,
                "subject_group": subject_group,
                "dataset_name": source_name or "unknown",
                "folder_name": folder_name,
                "SPO2_win_mean": y,
            }
            for ch_name, filt_arr, orig_arr in [("R", r_filtered, R),
                                                  ("G", g_filtered, G),
                                                  ("B", b_filtered, B)]:
                ac = np.std(filt_arr[start:end])
                dc = np.mean(orig_arr[start:end])
                row[f"{ch_name}_acdc"] = (ac / dc) if dc >= 1e-6 else 0.0
                row[f"{ch_name}_mean"] = float(orig_arr[end - 1])
            # Ratio of Ratios features (classic SpO2 estimation principle)
            r_acdc = row["R_acdc"]
            g_acdc = row["G_acdc"]
            b_acdc = row["B_acdc"]
            row["RoR_RG_acdc"] = (r_acdc / g_acdc) if abs(g_acdc) > 1e-8 else 0.0
            row["RoR_RB_acdc"] = (r_acdc / b_acdc) if abs(b_acdc) > 1e-8 else 0.0
            # Derived channel features: POS_Y and CHROM_X
            for ch_name, filt_arr, orig_arr in [("POS_Y", pos_y_filtered, pos_y_raw),
                                                  ("CHROM_X", chrom_x_filtered, chrom_x_raw)]:
                ac = np.std(filt_arr[start:end])
                dc = abs(np.mean(orig_arr[start:end]))
                row[f"{ch_name}_acdc"] = (ac / dc) if dc >= 1e-6 else 0.0
            # Long-window AC/DC features (180 frames = 6s, for GBR model stability)
            long_win = 180
            long_start = max(0, end - long_win)
            for ch_name, filt_arr, orig_arr in [("R", r_filtered, R),
                                                  ("G", g_filtered, G),
                                                  ("B", b_filtered, B)]:
                ac_long = np.std(filt_arr[long_start:end])
                dc_long = np.mean(orig_arr[long_start:end])
                row[f"{ch_name}_acdc_long"] = (ac_long / dc_long) if dc_long >= 1e-6 else 0.0
            r_acdc_long = row["R_acdc_long"]
            g_acdc_long = row["G_acdc_long"]
            b_acdc_long = row["B_acdc_long"]
            row["RoR_RG_acdc_long"] = (r_acdc_long / g_acdc_long) if abs(g_acdc_long) > 1e-8 else 0.0
            row["RoR_RB_acdc_long"] = (r_acdc_long / b_acdc_long) if abs(b_acdc_long) > 1e-8 else 0.0
            # Temporal derivative features: difference from previous window
            if start > 0:
                prev_start = start - STEP
                prev_end = prev_start + WIN_LEN
                for ch_name, filt_arr, orig_arr in [("R", r_filtered, R),
                                                      ("G", g_filtered, G),
                                                      ("B", b_filtered, B)]:
                    prev_ac = np.std(filt_arr[prev_start:prev_end])
                    prev_dc = np.mean(orig_arr[prev_start:prev_end])
                    prev_acdc = (prev_ac / prev_dc) if prev_dc >= 1e-6 else 0.0
                    row[f"delta_{ch_name}_acdc"] = row[f"{ch_name}_acdc"] - prev_acdc
            else:
                row["delta_R_acdc"] = 0.0
                row["delta_G_acdc"] = 0.0
                row["delta_B_acdc"] = 0.0
            # SQI from rPPG signal in this window
            rppg_win = raw_rppg[0, start:end]
            if len(rppg_win) >= 30:
                _, sqi_val, _, _ = calculate_hr_score(rppg_win, FS)
                row["sqi"] = sqi_val
            else:
                row["sqi"] = 0.0
            # Multi-scale contrast: ratio of long-window to short-window AC/DC
            for ch_name in ["R", "G", "B"]:
                short_val = row[f"{ch_name}_acdc"]
                long_val = row[f"{ch_name}_acdc_long"]
                row[f"acdc_ratio_long_short_{ch_name}"] = (long_val / short_val) if abs(short_val) > 1e-8 else 1.0
            rows.append(row)
            
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


def filter_feat_df_by_spo2_range(feat_df, min_range=MIN_SPO2_RANGE, segment_length=None, verbose=True):
    """
    依「同一 subject_id 下會被用來訓練/評估的 SpO2 跨幅」篩選 subject。
    - 若提供 segment_length：只考慮「該 folder 樣本數 >= segment_length」的 rows 來算跨幅，
      與後續 train/eval 實際使用的資料一致，避免通過篩選的 subject 在圖上只剩窄範圍。
    - 僅保留跨幅 >= min_range 的 subject。
    回傳篩選後的 DataFrame（不修改原 DataFrame）。
    """
    if "subject_id" not in feat_df.columns or "SPO2_win_mean" not in feat_df.columns:
        return feat_df
    if "folder_name" not in feat_df.columns and segment_length is not None:
        segment_length = None  # 無法依 folder 過濾時退化成用全部 rows

    if segment_length is not None:
        # 只考慮「會被使用的」folder：該 folder 的 row 數 >= segment_length
        folder_counts = feat_df.groupby("folder_name").size()
        valid_folders = folder_counts.index[folder_counts >= segment_length].tolist()
        df_for_range = feat_df[feat_df["folder_name"].isin(valid_folders)]
    else:
        df_for_range = feat_df

    agg = df_for_range.groupby("subject_id")["SPO2_win_mean"].agg(["min", "max"])
    agg["spo2_range"] = agg["max"] - agg["min"]
    valid_subject_ids = agg.index[agg["spo2_range"] >= min_range].tolist()
    dropped = agg.index[agg["spo2_range"] < min_range].tolist()
    filtered = feat_df[feat_df["subject_id"].isin(valid_subject_ids)].copy()
    if verbose:
        print(f"SpO2 跨幅篩選 (min_range={min_range}, 僅計入長度>={segment_length or 'N/A'} 的 folder): 保留 {len(valid_subject_ids)} 個 subject，排除 {len(dropped)} 個 subject")
        if dropped:
            for sid in dropped:
                r = agg.loc[sid, "spo2_range"]
                print(f"  排除 subject_id={sid} (SpO2 跨幅={r:.2f})")
    return filtered


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


def infer_dataset_name(subject_id, default="unknown"):
    if not subject_id or "_" not in str(subject_id):
        return default
    return str(subject_id).rsplit("_", 1)[0]


def evaluate_single_model_on_dataset(model_info, dataset_df, feature_cols, segment_length,
                                     use_normalization=USE_NORMALIZATION):
    predictions = []
    labels = []
    subject_rows = []

    for test_subject_id, test_subdf in dataset_df.groupby("subject_id", sort=False):
        test_subject_group = test_subdf["subject_group"].iloc[0]
        test_exclusion_key = test_subject_group if EXCLUDE_SAME_SUBJECT_GROUP else test_subject_id
        model_exclusion_key = (
            model_info.get("subject_group", model_info["subject_id"])
            if EXCLUDE_SAME_SUBJECT_GROUP else
            model_info["subject_id"]
        )
        if test_exclusion_key == model_exclusion_key:
            continue

        subject_segment_predictions = []
        subject_segment_labels = []

        for _, folder_df in test_subdf.groupby("folder_name", sort=False):
            n_samples = len(folder_df)
            if n_samples < segment_length:
                continue

            n_segments = n_samples // segment_length
            X_test_full = folder_df[feature_cols].astype(float)
            y_test_full = folder_df["SPO2_win_mean"].values.astype(float)

            for seg_idx in range(n_segments):
                start_idx = seg_idx * segment_length
                end_idx = start_idx + segment_length
                X_test_segment_df = X_test_full.iloc[start_idx:end_idx].copy()
                y_test_segment = y_test_full[start_idx:end_idx]

                test_stds = X_test_segment_df.std(axis=0)
                if (test_stds < 1e-8).any():
                    continue

                if use_normalization:
                    test_means = X_test_segment_df.mean(axis=0)
                    X_test_input = (X_test_segment_df - test_means) / test_stds
                else:
                    X_test_input = X_test_segment_df

                y_pred = np.dot(X_test_input[model_info["feature_cols"]].values, model_info["weights"]) + model_info["bias"]
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    continue

                subject_segment_predictions.append(y_pred)
                subject_segment_labels.append(y_test_segment)

        if len(subject_segment_predictions) == 0:
            continue

        y_pred_full = np.concatenate(subject_segment_predictions)
        y_true_full = np.concatenate(subject_segment_labels)
        subject_range = float(np.max(y_true_full) - np.min(y_true_full))
        if subject_range < MIN_SPO2_RANGE:
            continue

        r2 = r2_score(y_true_full, y_pred_full)
        mae = mean_absolute_error(y_true_full, y_pred_full)
        pcc = np.nan if np.std(y_pred_full) < 1e-8 else pearsonr(y_true_full, y_pred_full)[0]
        subject_rows.append({
            "subject_id": test_subject_id,
            "R2": r2,
            "MAE": mae,
            "PCC": pcc,
            "n_frames": len(y_true_full),
        })
        predictions.append(y_pred_full)
        labels.append(y_true_full)

    if len(subject_rows) == 0:
        return {
            "n_subjects": 0,
            "mean_R2": np.nan,
            "mean_MAE": np.nan,
            "mean_PCC": np.nan,
            "overall_PCC": np.nan,
        }

    subject_df = pd.DataFrame(subject_rows)
    all_pred = np.concatenate(predictions)
    all_true = np.concatenate(labels)
    overall_pcc = np.nan if np.std(all_pred) < 1e-8 else pearsonr(all_true, all_pred)[0]
    return {
        "n_subjects": int(len(subject_df)),
        "mean_R2": float(subject_df["R2"].mean()),
        "mean_MAE": float(subject_df["MAE"].mean()),
        "mean_PCC": float(subject_df["PCC"].mean()),
        "overall_PCC": float(overall_pcc) if not np.isnan(overall_pcc) else np.nan,
    }


def evaluate_model_pool_by_dataset(model_pool, feat_df, selected_channels=None,
                                   segment_length=SEGMENT_LENGTH,
                                   use_normalization=USE_NORMALIZATION):
    feature_cols = get_pool_feature_cols(selected_channels)
    req_cols = ["subject_id", "subject_group", "dataset_name", "folder_name", "SPO2_win_mean"] + feature_cols
    feat_df_selected = feat_df[req_cols].copy()

    dataset_names = []
    if "dataset_name" in feat_df_selected.columns:
        dataset_names = sorted(feat_df_selected["dataset_name"].dropna().unique().tolist())
    if not dataset_names:
        dataset_names = sorted({infer_dataset_name(sid) for sid in feat_df_selected["subject_id"].unique()})
        feat_df_selected["dataset_name"] = feat_df_selected["subject_id"].map(infer_dataset_name)

    evaluation_rows = []
    for model_info in model_pool:
        for dataset_name in dataset_names:
            dataset_df = feat_df_selected[feat_df_selected["dataset_name"] == dataset_name]
            metrics = evaluate_single_model_on_dataset(
                model_info=model_info,
                dataset_df=dataset_df,
                feature_cols=feature_cols,
                segment_length=segment_length,
                use_normalization=use_normalization,
            )
            evaluation_rows.append({
                "model_id": model_info["model_id"],
                "dataset_name": dataset_name,
                "train_subject_id": model_info["subject_id"],
                "train_dataset_name": model_info.get("dataset_name", infer_dataset_name(model_info["subject_id"])),
                "segment_id": model_info["segment_id"],
                "train_R2": model_info.get("train_r2", np.nan),
                "train_PCC": model_info.get("train_pcc", np.nan),
                **metrics,
            })

    return pd.DataFrame(evaluation_rows)


def select_model_pool(raw_model_pool, feat_df, selected_channels=None,
                      segment_length=SEGMENT_LENGTH,
                      use_normalization=USE_NORMALIZATION,
                      min_dataset_r2=0.1,
                      min_pool_size=5,
                      max_pool_size=None):
    if len(raw_model_pool) == 0:
        empty_df = pd.DataFrame()
        return raw_model_pool, empty_df, empty_df

    eval_df = evaluate_model_pool_by_dataset(
        model_pool=raw_model_pool,
        feat_df=feat_df,
        selected_channels=selected_channels,
        segment_length=segment_length,
        use_normalization=use_normalization,
    )
    if len(eval_df) == 0:
        empty_df = pd.DataFrame()
        return raw_model_pool, empty_df, empty_df

    valid_eval_df = eval_df[eval_df["n_subjects"] > 0].copy()
    if len(valid_eval_df) == 0:
        summary_df = eval_df.copy()
        summary_df["coverage_count"] = 0
        summary_df["model_score"] = np.nan
        return raw_model_pool, eval_df, summary_df

    summary_df = (
        valid_eval_df.groupby("model_id", as_index=False)
        .agg(
            coverage_count=("mean_R2", lambda s: int(np.sum(np.nan_to_num(s, nan=-1.0) >= min_dataset_r2))),
            mean_dataset_R2=("mean_R2", "mean"),
            best_dataset_R2=("mean_R2", "max"),
            mean_dataset_PCC=("mean_PCC", "mean"),
            mean_dataset_MAE=("mean_MAE", "mean"),
            datasets_evaluated=("dataset_name", "nunique"),
            train_subject_id=("train_subject_id", "first"),
            train_dataset_name=("train_dataset_name", "first"),
            segment_id=("segment_id", "first"),
            train_R2=("train_R2", "first"),
            train_PCC=("train_PCC", "first"),
        )
    )
    summary_df["model_score"] = (
        0.7 * summary_df["mean_dataset_R2"].fillna(-1.0)
        + 0.2 * summary_df["best_dataset_R2"].fillna(-1.0)
        + 0.1 * summary_df["mean_dataset_PCC"].fillna(-1.0)
    )
    summary_df = summary_df.sort_values(
        ["coverage_count", "model_score", "best_dataset_R2", "train_PCC"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    max_pool_size = max_pool_size or len(summary_df)
    max_pool_size = max(min_pool_size, min(max_pool_size, len(summary_df)))

    selected_ids = []
    selected_id_set = set()

    dataset_seed_df = valid_eval_df.sort_values(
        ["dataset_name", "mean_R2", "mean_PCC", "train_PCC"],
        ascending=[True, False, False, False],
    )
    for dataset_name, dataset_rows in dataset_seed_df.groupby("dataset_name", sort=False):
        preferred = dataset_rows[dataset_rows["mean_R2"] >= min_dataset_r2]
        candidate_rows = preferred if len(preferred) > 0 else dataset_rows
        best_model_id = candidate_rows.iloc[0]["model_id"]
        if best_model_id not in selected_id_set:
            selected_ids.append(best_model_id)
            selected_id_set.add(best_model_id)
        if len(selected_ids) >= max_pool_size:
            break

    for _, row in summary_df.iterrows():
        model_id = row["model_id"]
        if model_id in selected_id_set:
            continue
        selected_ids.append(model_id)
        selected_id_set.add(model_id)
        if len(selected_ids) >= max_pool_size:
            break

    if len(selected_ids) < min_pool_size:
        for _, row in summary_df.iterrows():
            model_id = row["model_id"]
            if model_id in selected_id_set:
                continue
            selected_ids.append(model_id)
            selected_id_set.add(model_id)
            if len(selected_ids) >= min_pool_size:
                break

    selected_pool = [model_info for model_info in raw_model_pool if model_info["model_id"] in selected_id_set]
    summary_df["selected"] = summary_df["model_id"].isin(selected_id_set)
    return selected_pool, eval_df, summary_df


def train_model_pool(selected_channels, feat_df, segment_length=SEGMENT_LENGTH,
                     use_normalization=USE_NORMALIZATION, use_regularization=USE_REGULARIZATION, alpha=ALPHA,
                     min_train_r2=MIN_TRAIN_R2, min_train_pcc=MIN_TRAIN_PCC):
    """
    訓練所有 subject 的模型並建構模型池（使用 Folder 獨立切分策略）。
    回傳 model_pool: list of dict（每項含 subject_id, folder_name, segment_id, model_id, model, feature_cols, weights, bias, means, stds, X_raw_segment）。
    """
    print("\n" + "=" * 60)
    print(f"訓練模型池（段長度={segment_length}，依 Folder 獨立切割）...")
    print("=" * 60)
    
    feature_cols = get_pool_feature_cols(selected_channels)
    
    # 確保特徵中包含 folder_name
    req_cols = ["subject_id", "subject_group", "dataset_name", "folder_name", "SPO2_win_mean"] + feature_cols + rgb_mean_col_names
    if "folder_name" not in feat_df.columns:
        raise ValueError("特徵 DataFrame 缺少 'folder_name' 欄位，請確認 build_features_from_df 是否已更新！")
        
    feat_df_selected = feat_df[req_cols].copy()

    model_pool = []
    total_segments = 0
    skipped_folders = 0
    filtered_by_pcc = 0
    filtered_by_r2 = 0

    # 第一層：依照 subject_id 分組
    for subject_id, subdf in feat_df_selected.groupby("subject_id", sort=False):
        n_samples_sub = len(subdf)
        print(f"\n處理 Subject {subject_id}: 總共 {n_samples_sub} 個樣本")
        
        # 第二層：依照 folder_name (片段) 分組，確保 Segment 不跨 Folder
        for folder_name, folder_df in subdf.groupby("folder_name", sort=False):
            n_samples = len(folder_df)
            if n_samples < segment_length:
                print(f"  跳過 Folder {folder_name}: 樣本數不足 {segment_length} (僅 {n_samples})")
                skipped_folders += 1
                continue
                
            n_segments = n_samples // segment_length
            print(f"  Folder {folder_name}: {n_samples} 樣本，可切為 {n_segments} 段")
            
            X_full = folder_df[feature_cols].astype(float)
            rgb_mean_full = folder_df[rgb_mean_col_names].astype(float)
            y_full = folder_df["SPO2_win_mean"].values.astype(float)

            for seg_idx in range(n_segments):
                start_idx = seg_idx * segment_length
                end_idx = start_idx + segment_length
                
                X_raw_segment = X_full.iloc[start_idx:end_idx].copy()
                y_segment = y_full[start_idx:end_idx]
                rgb_mean_segment = rgb_mean_full.iloc[start_idx:end_idx].mean(axis=0).values
                
                stds = X_raw_segment.std(axis=0)
                if (stds < 1e-8).any():
                    constant_cols = X_raw_segment.columns[stds < 1e-8].tolist()
                    print(f"    段 {seg_idx}: 跳過（有常數特徵: {constant_cols}）")
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
                    print(f"    段 {seg_idx}: 跳過（訓練集 R2 = {train_r2:.4f} <= {min_train_r2}）")
                    filtered_by_r2 += 1
                    continue
                if train_pcc <= min_train_pcc:
                    print(f"    段 {seg_idx}: 跳過（訓練集 PCC = {train_pcc:.4f} <= {min_train_pcc}）")
                    filtered_by_pcc += 1
                    continue

                # 修改 model_id，加入 folder_name 確保唯一性與可讀性
                model_id = f"{subject_id}_{folder_name}_seg{seg_idx}"
                subject_group = subdf["subject_group"].iloc[0] if "subject_group" in subdf.columns else subject_id
                model_pool.append({
                    'subject_id': subject_id,
                    'subject_group': subject_group,
                    'folder_name': folder_name,
                    'segment_id': seg_idx,
                    'model_id': model_id,
                    'dataset_name': folder_df["dataset_name"].iloc[0] if "dataset_name" in folder_df.columns else infer_dataset_name(subject_id),
                    'model': lr,
                    'feature_cols': X_raw_segment.columns.tolist(),
                    'weights': lr.coef_,
                    'bias': lr.intercept_,
                    'train_r2': float(train_r2),
                    'train_pcc': float(train_pcc),
                    'means': means.values if means is not None else None,
                    'stds': stds.values if stds is not None else None,
                    'X_raw_segment': X_raw_segment.values,
                    'rgb_mean_segment': rgb_mean_segment,
                    'spo2_mean': float(np.mean(y_segment)),
                    'spo2_std': float(np.std(y_segment)),
                })
                total_segments += 1
        print(f"  Subject {subject_id} 完成")

    print(f"\n模型池建構完成:")
    print(f"  最終模型池大小: {len(model_pool)} 個模型")
    print(f"  過濾統計: R2不足({filtered_by_r2}), PCC不足({filtered_by_pcc})")
    print("=" * 60)
    return model_pool


GLOBAL_MODEL_BLEND = 0.35  # Blend weight for global LOSO model (0 = ensemble only, 1 = global only)
ROLLING_MEDIAN_WINDOW = 1801  # Set <= 1 to disable subject-level rolling median smoothing
EMA_SPAN = 901  # Set <= 1 to disable subject-level EMA smoothing
EXCLUDE_SAME_SUBJECT_GROUP = True  # When False, only exact same subject_id is excluded from the pool/global model


def ensemble_predict_and_evaluate(model_pool, feat_df, selected_channels, top_k,
                                  segment_length=SEGMENT_LENGTH,
                                  use_normalization=USE_NORMALIZATION,
                                  save_plots=False, output_dir=OUTPUT_DIR,
                                  return_frame_predictions=False):
    """
    使用 ensemble 模型池進行預測與評估。
    邏輯更新：
    1. Folder 總長度不足 segment_length (900) 的直接捨棄。
    2. 第一段 900 幀正常挑選模型。
    3. 結尾不足 900 的殘餘部分，直接沿用該 Folder 最後一次選出的模型。
    """
    print("\n" + "=" * 60)
    print(f"使用 Ensemble 模型池進行預測 (TOP_K={top_k}, 段長度={segment_length})...")
    print("=" * 60)
    
    feature_cols = get_pool_feature_cols(selected_channels)
    req_cols = dedupe_preserve_order(
        ["subject_id", "subject_group", "folder_name", "SPO2_win_mean"] +
        feature_cols + rgb_mean_col_names + derived_acdc_col_names
    )
    feat_df_selected = feat_df[req_cols].copy()

    results = []
    all_subjects_predictions = []
    all_subjects_labels = []
    frame_prediction_rows = []

    for test_subject_id, test_subdf in feat_df_selected.groupby("subject_id", sort=False):
        test_subject_group = test_subdf["subject_group"].iloc[0]
        print(f"\n測試 Subject {test_subject_id} (group {test_subject_group})")
        test_exclusion_key = test_subject_group if EXCLUDE_SAME_SUBJECT_GROUP else test_subject_id

        # Train a global LOSO model for blending
        # Exclude all subjects in the same group (same person, different devices)
        global_model = None
        global_feature_cols = None
        if GLOBAL_MODEL_BLEND > 0:
            from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
            train_exclusion_col = "subject_group" if EXCLUDE_SAME_SUBJECT_GROUP else "subject_id"
            train_df = feat_df_selected[feat_df_selected[train_exclusion_col] != test_exclusion_key]
            # Use feature cols, RGB means, and derived channel features for global model
            global_feature_cols = dedupe_preserve_order(
                feature_cols + [c for c in rgb_mean_col_names if c not in feature_cols] + derived_acdc_col_names
            )
            X_global_train = train_df[global_feature_cols].values.astype(float)
            y_global_train = train_df["SPO2_win_mean"].values.astype(float)
            if len(X_global_train) > 10:
                global_model = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=3, learning_rate=0.03,
                    min_samples_leaf=100, l2_regularization=2.0,
                    random_state=42
                )
                global_model.fit(X_global_train, y_global_train)

        subject_segment_predictions = []
        subject_segment_labels = []

        for folder_name, folder_df in test_subdf.groupby("folder_name", sort=False):
            n_samples = len(folder_df)
            
            # 【修改點】：如果 Folder 總長度不到 900，直接捨棄，不進行特例處理
            if n_samples < segment_length:
                print(f"  Folder {folder_name}: 長度不足 {segment_length}，已捨棄。")
                continue
                
            n_test_segments = n_samples // segment_length
            total_segments = math.ceil(n_samples / segment_length) 
            
            X_test_full = folder_df[feature_cols].astype(float)
            rgb_mean_test_full = folder_df[rgb_mean_col_names].astype(float)
            y_test_full = folder_df["SPO2_win_mean"].values.astype(float)
            
            last_top_k_models = None 

            for test_seg_idx in range(total_segments):
                start_idx = test_seg_idx * segment_length
                end_idx = min(start_idx + segment_length, n_samples) 
                
                X_test_segment_df = X_test_full.iloc[start_idx:end_idx].copy()
                y_test_segment = y_test_full[start_idx:end_idx]
                
                test_stds = X_test_segment_df.std(axis=0)
                if (test_stds < 1e-8).any():
                    continue
                
                # 只有在完整的 900 幀段落才重新挑選模型
                if test_seg_idx < n_test_segments:
                    X_test_segment_array = X_test_segment_df.values
                    rgb_mean_test_segment = rgb_mean_test_full.iloc[start_idx:end_idx].mean(axis=0).values
                    
                    similarities = []
                    for model_info in model_pool:
                        # Exclude all models from same person (same group), not just same device
                        model_exclusion_key = (
                            model_info.get('subject_group', model_info['subject_id'])
                            if EXCLUDE_SAME_SUBJECT_GROUP else
                            model_info['subject_id']
                        )
                        if model_exclusion_key == test_exclusion_key:
                            continue

                        shape_score, range_score = calculate_similarity(model_info['X_raw_segment'], X_test_segment_array, model_info['weights'])
                        rgb_score = calculate_rgb_similarity(model_info['rgb_mean_segment'], rgb_mean_test_segment)

                        similarities.append({
                            'model_info': model_info,
                            'shape_score': shape_score,
                            'range_score': range_score,
                            'rgb_score': rgb_score,
                            'model_id': model_info['model_id']
                        })

                    if len(similarities) == 0:
                        continue

                    # Single-stage selection: combine all similarity metrics
                    for s in similarities:
                        s['combined'] = 0.3 * s['shape_score'] + 0.7 * s['range_score']
                    similarities.sort(key=lambda x: x['combined'], reverse=True)
                    top_k_models = similarities[:min(top_k, len(similarities))]
                    
                    last_top_k_models = top_k_models 
                    
                else:
                    # 殘餘段落：沿用最後一次成功的模型組合
                    top_k_models = last_top_k_models
                    if top_k_models is None: # 防護邏輯：理論上上面已經過濾長度，此處不應發生
                        continue

                # 預測運算
                if use_normalization:
                    test_means, test_stds = X_test_segment_df.mean(axis=0), X_test_segment_df.std(axis=0)
                    X_test_input = (X_test_segment_df - test_means) / test_stds
                else:
                    X_test_input = X_test_segment_df

                segment_preds = []
                sim_weights = []
                train_spo2_stats = []  # (mean, std) per model
                for sim_info in top_k_models:
                    m = sim_info['model_info']
                    y_p = np.dot(X_test_input[m['feature_cols']].values, m['weights']) + m['bias']
                    if np.any(np.isnan(y_p)) or np.any(np.isinf(y_p)):
                        continue
                    segment_preds.append(y_p)
                    w = max(sim_info['combined'], 0.01)
                    sim_weights.append(w)
                    train_spo2_stats.append((m.get('spo2_mean', 95.0), m.get('spo2_std', 3.0)))

                if len(segment_preds) > 0:
                    preds_arr = np.array(segment_preds)
                    weights_arr = np.array(sim_weights)
                    train_means_arr = np.array([s[0] for s in train_spo2_stats])
                    train_stds_arr = np.array([s[1] for s in train_spo2_stats])
                    # Consensus filtering
                    med_pred = np.median(preds_arr, axis=0)
                    deviations = np.mean(np.abs(preds_arr - med_pred), axis=1)
                    dev_threshold = np.median(deviations) * 1.5
                    keep_mask = deviations <= dev_threshold
                    if np.sum(keep_mask) >= 3:
                        preds_arr = preds_arr[keep_mask]
                        weights_arr = weights_arr[keep_mask]
                        train_means_arr = train_means_arr[keep_mask]
                        train_stds_arr = train_stds_arr[keep_mask]
                    # Weighted average
                    weights_arr = weights_arr / weights_arr.sum()
                    avg_pred = np.average(preds_arr, axis=0, weights=weights_arr)
                    # Blend with global LOSO model
                    if global_model is not None and GLOBAL_MODEL_BLEND > 0:
                        X_global_test = folder_df[global_feature_cols].iloc[start_idx:end_idx].astype(float)
                        global_pred = global_model.predict(X_global_test.values)
                        avg_pred = (1 - GLOBAL_MODEL_BLEND) * avg_pred + GLOBAL_MODEL_BLEND * global_pred
                    avg_pred = np.clip(avg_pred, 70, 100)
                    subject_segment_predictions.append(avg_pred)
                    subject_segment_labels.append(y_test_segment)

        # 彙整該 Subject 結果
        if len(subject_segment_predictions) == 0:
            continue

        y_ensemble = np.concatenate(subject_segment_predictions)
        y_test = np.concatenate(subject_segment_labels)

        # Rolling median smoothing to reduce prediction noise
        smooth_window = int(ROLLING_MEDIAN_WINDOW)
        if smooth_window > 1 and smooth_window % 2 == 0:
            smooth_window += 1
        if smooth_window > 1 and len(y_ensemble) >= smooth_window:
            y_smoothed = pd.Series(y_ensemble).rolling(window=smooth_window, center=True, min_periods=1).median().values
            y_ensemble = y_smoothed

        # Second-pass EMA smoothing for additional noise reduction
        ema_span = int(EMA_SPAN)
        if ema_span > 1 and len(y_ensemble) >= ema_span:
            y_ema = pd.Series(y_ensemble).ewm(span=ema_span, adjust=False).mean().values
            y_ensemble = y_ema

        # 防護：實際用於評估的資料跨幅若仍不足，跳過該 subject（不列入結果、不繪圖）
        subject_range = float(np.max(y_test) - np.min(y_test))
        if subject_range < MIN_SPO2_RANGE:
            print(f"  跳過 Subject {test_subject_id}: 實際用於評估的 SpO2 跨幅={subject_range:.2f} < {MIN_SPO2_RANGE}，不列入結果與繪圖")
            continue

        r2 = r2_score(y_test, y_ensemble)
        mae = mean_absolute_error(y_test, y_ensemble)
        pcc = np.nan if np.std(y_ensemble) < 1e-8 else pearsonr(y_test, y_ensemble)[0]
        
        results.append({"subject_id": test_subject_id, "n_frames": len(y_test), "R2": r2, "MAE": mae, "PCC": pcc})
        all_subjects_predictions.append(y_ensemble)
        all_subjects_labels.append(y_test)
        if return_frame_predictions:
            frame_prediction_rows.extend(
                {
                    "subject_id": test_subject_id,
                    "subject_group": test_subject_group,
                    "frame_idx": frame_idx,
                    "y_true": float(y_true_val),
                    "y_pred": float(y_pred_val),
                }
                for frame_idx, (y_true_val, y_pred_val) in enumerate(zip(y_test, y_ensemble))
            )

        if save_plots:
            plot_dir = os.path.join(output_dir, "sub_plot")
            os.makedirs(plot_dir, exist_ok=True)
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, label="True SpO2", linewidth=1.5, color='black')
            plt.plot(y_ensemble, label=f"Ensemble Prediction", linewidth=1.5, color='red', linestyle='--')
            
            # 畫出每個段落(包含沿用的殘餘段落)的邊界，方便查看
            current_pos = 0
            for seg_pred in subject_segment_predictions:
                current_pos += len(seg_pred)
                plt.axvline(x=current_pos, color='blue', linestyle=':', alpha=0.3)
                
            plt.title(f"Subject {test_subject_id} - Ensemble (TOP_K={top_k}, R2={r2:.3f}, MAE={mae:.2f}, PCC={pcc if not np.isnan(pcc) else 0:.3f})")
            plt.xlabel("Cumulative Frame index (Segment Boundaries marked by dotted lines)")
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
            print(f"總預測樣本數: {len(all_pred)}")
    else:
        overall_pcc = np.nan
        
    results_df['overall_PCC'] = overall_pcc
    print("\n" + "=" * 60)
    print("Ensemble 預測完成")
    print("=" * 60)
    if return_frame_predictions:
        return results_df, pd.DataFrame(frame_prediction_rows)
    return results_df
