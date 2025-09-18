# train_loso_tcn.py
# Cross-subject LOSO with a simple TCN (Temporal Convolutional Network)
# No filtering; uses six-channel transform provided by the user.

import argparse
import os
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def getsixchannels(rgb: np.ndarray) -> List[np.ndarray]:
    """rgb: (3, N) array -> returns list of 6 arrays (length N each)."""
    # rgb rows: [R, G, B]
    yiq_i =  0.595716 * rgb[0, :] - 0.274453 * rgb[1, :] - 0.321263 * rgb[2, :]
    ycgcr_cg = 128.0 -  81.085 * rgb[0, :] / 255.0 + 112.000 * rgb[1, :] / 255.0 - 30.915 * rgb[2, :] / 255.0
    ycgcr_cr = 128.0 + 112.000 * rgb[0, :] / 255.0 -  93.786 * rgb[1, :] / 255.0 - 18.214 * rgb[2, :] / 255.0
    ydbdr_dr = -1.333 * rgb[0, :] + 1.116 * rgb[1, :] + 0.217 * rgb[2, :]
    pos_y = -2*rgb[0, :] + rgb[1, :] + rgb[2, :]
    chrom_x =   3*rgb[0, :] - 2*rgb[1, :]
    return [ycgcr_cg, ycgcr_cr, yiq_i, ydbdr_dr, pos_y, chrom_x]

def get_invariant_channels(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (3, N) 原始 0..255
    return: (K, N) 追加的強度/白平衡更不敏感特徵
    """
    eps = 1e-8
    R = rgb[0].astype(np.float32)
    G = rgb[1].astype(np.float32)
    B = rgb[2].astype(np.float32)

    S = R + G + B + eps
    # A) 強度尺度不變
    r = R / S
    g = G / S
    # hue（用幾何定義，比 HSV 更平滑）
    hue = np.arctan2(np.sqrt(3.0) * (B - R), 2.0*G - R - B)

    # NDI / ratio family
    rg_nd = (R - G) / (R + G + eps)
    gb_nd = (G - B) / (G + B + eps)
    rb_nd = (R - B) / (R + B + eps)
    log_rg = np.log((R + eps) / (G + eps))
    log_gb = np.log((G + eps) / (B + eps))

    # C) AC/DC（以短窗或整段，先給全段粗略版；若要窗內版本，放到切窗後做）
    R_mu, G_mu, B_mu = R.mean(), G.mean(), B.mean()
    acdc_R = (R - R_mu) / (R_mu + eps)
    acdc_G = (G - G_mu) / (G_mu + eps)
    acdc_B = (B - B_mu) / (B_mu + eps)

    # D) 一階差分（對 rg/hue 做就好）
    dr = np.diff(r, prepend=r[0])
    dg = np.diff(g, prepend=g[0])
    dh = np.diff(hue, prepend=hue[0])

    feats = np.vstack([
        r, g, hue, rg_nd, gb_nd, rb_nd, log_rg, log_gb,
        # acdc_R, acdc_G, acdc_B,
        # dr, dg, dh
    ])
    return feats.astype(np.float32)

def getCCT(rgb_: np.ndarray) -> np.ndarray:
    """
    rgb_: (3, N) in [0..255]
    回傳 shape: (1, N) 的 CCT；若分母為 0 則以極小值保護。
    公式：McCamy approximation
    """
    eps = 1e-8
    rgb = rgb_.astype(np.float32) / 255.0
    R, G, B = rgb[0], rgb[1], rgb[2]

    # sRGB D65 -> XYZ（線性）
    X = 0.4124*R + 0.3576*G + 0.1805*B
    Y = 0.2126*R + 0.7152*G + 0.0722*B
    Z = 0.0193*R + 0.1192*G + 0.9505*B

    denom = np.maximum(X + Y + Z, eps)
    x = X / denom
    y = Y / denom

    n = (x - 0.3320) / np.maximum(0.1858 - y, eps)
    CCT = 437*(n**3) + 3601*(n**2) + 6861*n + 5517
    return CCT.reshape(1, -1).astype(np.float32)

def build_windows(
    X_all: np.ndarray, spo2: np.ndarray, seq_len: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_all: (C, N), spo2: (N,)
    Return X: (num_windows, C, seq_len), y: (num_windows,)
    Label = mean SpO2 within the window
    """
    C, N = X_all.shape[0], X_all.shape[1]
    xs, ys = [], []
    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        xs.append(X_all[:, start:end])
        # ys.append(float(spo2[start:end].mean()))
        # 使用最後一個值
        ys.append(float(spo2[end - 1]))
    if len(xs) == 0:
        return np.zeros((0, C, seq_len), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y

def apply_segment_norm_numpy(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    Xn = X.copy()
    mu = Xn.mean(axis=2, keepdims=True)
    std = Xn.std(axis=2, keepdims=True) + 1e-8
    Xn = (Xn - mu) / std
    return Xn

def flatten_windows(X: np.ndarray) -> np.ndarray:
    """(N, C, T) -> (N, C*T) 用於 sklearn 模型。"""
    if X.ndim != 3:
        return X
    N, C, T = X.shape
    return X.reshape(N, C * T)

def compute_sample_weights_from_labels(y: np.ndarray, bin_edges: np.ndarray = None, gamma: float = 1.0) -> np.ndarray:
    """
    給定 y（SpO2 標籤），依分佈計算每個樣本的權重。
    預設用 75~100 的整數 bin；權重 = (1 / bin_count) ** gamma，再做 min-max 或均值正規化到 ~1 的量級。
    """
    if bin_edges is None:
        bin_edges = np.arange(74.5, 100.5 + 1e-6, 1.0)  # 75,76,...,100 的邊界

    # 將 y 以整數 SpO2 分箱（或你也可以改成 np.round(y)）
    y_int = np.clip(np.round(y).astype(int), 75, 100)
    hist_counts = np.bincount(y_int - 75, minlength=26).astype(np.float64)  # 75..100 共 26 格

    # 避免除以 0：對空 bin 給一個很小的 count
    hist_counts = np.where(hist_counts > 0, hist_counts, 1.0)

    inv_freq = 1.0 / hist_counts  # 反頻率
    inv_freq = inv_freq ** max(1.0, gamma)

    # 將每個樣本對應到自己的 bin 權重
    weights = inv_freq[y_int - 75]

    # 正規化到平均為 1（穩定訓練）
    weights = weights / (weights.mean() + 1e-8)
    return weights.astype(np.float32)

def welford_cumulative_stats(ch6: np.ndarray, eps: float = 1e-8):
    """
    因果累積（Welford）估計每個 t 的 mu_t, std_t。
    ch6: (C, N) -> returns mu (C,N), std (C,N)
    """
    C, N = ch6.shape
    # 使用 float64 提高精度，避免溢出
    mu = np.zeros((C, N), dtype=np.float64)
    M2 = np.zeros((C, N), dtype=np.float64)
    mu[:, 0] = ch6[:, 0].astype(np.float64)
    M2[:, 0] = 0.0
    
    for t in range(1, N):
        x = ch6[:, t].astype(np.float64)
        delta = x - mu[:, t-1]
        mu[:, t] = mu[:, t-1] + delta / (t + 1)
        delta2 = x - mu[:, t]
        M2[:, t] = M2[:, t-1] + delta * delta2
    
    var = np.zeros_like(M2)
    for t in range(N):
        denom = max(1, t)
        var[:, t] = M2[:, t] / denom
    std = np.sqrt(np.maximum(var, 0.0)) + eps
    
    # 轉回 float32 以保持與其他代碼的兼容性
    return mu.astype(np.float32), std.astype(np.float32)

def ema_stats_over_time(ch6: np.ndarray, alpha: float = 0.05, eps: float = 1e-8):
    """
    因果 EMA 估 mu_t, std_t（以 EMA 近似變異）。
    ch6: (C, N) -> returns mu (C,N), std (C,N)
    """
    C, N = ch6.shape
    mu = np.zeros((C, N), dtype=np.float32)
    v  = np.zeros((C, N), dtype=np.float32)
    mu[:, 0] = ch6[:, 0]
    v[:, 0]  = 0.0
    for t in range(1, N):
        x = ch6[:, t]
        mu[:, t] = alpha * x + (1 - alpha) * mu[:, t-1]
        diff = x - mu[:, t]
        v[:, t]  = alpha * (diff * diff) + (1 - alpha) * v[:, t-1]
    std = np.sqrt(v) + eps
    return mu, std

def build_windows_stride_frozen(
    X_all: np.ndarray, spo2: np.ndarray, seq_len: int, stride: int,
    mu_time: np.ndarray, std_time: np.ndarray, warmup_frames: int = 0
):
    C, N = X_all.shape
    xs, ys = [], []
    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        t_ref = end - 1
        if t_ref < warmup_frames:
            continue
        mu_w  = mu_time[:, t_ref][:, None]   # (C,1)
        std_w = std_time[:, t_ref][:, None]  # (C,1)
        x_win = X_all[:, start:end].astype(np.float32).copy()
        x_win = (x_win - mu_w) / (std_w + 1e-8)
        xs.append(x_win)
        # ys.append(float(spo2[start:end].mean()))
        # 使用最後一個值
        ys.append(float(spo2[end - 1]))
    if not xs:
        return np.zeros((0, C, seq_len), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.float32)

# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, segment_normalization: bool = False, weights: np.ndarray = None):
        """
        X: (N, C, T), y: (N,)
        segment_normalization=True 時，對每個樣本做 (x - mean_c)/std_c，按通道 c 各自計算。
        weights: (N,) 或 None；若提供，訓練時會用於加權 loss。
        """
        self.X = torch.from_numpy(X)        # 直接先轉 tensor，提高效率
        self.y = torch.from_numpy(y)
        self.segment_normalization = segment_normalization

        if weights is None:
            self.w = torch.ones(len(y), dtype=torch.float32)
        else:
            assert len(weights) == len(y), "weights 長度需與 y 相同"
            self.w = torch.from_numpy(weights.astype(np.float32))

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.segment_normalization:
            x = x.clone()
            mu  = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-8)
            x = (x - mu) / std
        return x, self.y[idx], self.w[idx]

# -----------------------------
# TCN building blocks
# -----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        # remove extra padding on the right
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNRegressor(nn.Module):
    def __init__(self, n_channels=6, hidden=64, levels=3, kernel_size=5, dropout=0.1):
        super().__init__()
        layers = []
        in_c = n_channels
        for i in range(levels):
            out_c = hidden
            dilation = 2 ** i
            layers += [TemporalBlock(in_c, out_c, kernel_size, dilation, dropout)]
            in_c = out_c
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)  # global average pooling over time
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: (B, C=6, T)
        h = self.tcn(x)
        h = self.gap(h)     # (B, hidden, 1)
        y = self.head(h)    # (B, 1)
        return y.squeeze(1)

# -----------------------------
# Training / Eval
# -----------------------------
def train_one(model, loader, optim, device):
    model.train()
    crit = nn.MSELoss(reduction='none')
    total = 0.0
    n = 0
    for xb, yb, wb in loader:  # ← 接 weights
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        optim.zero_grad()
        pred = model(xb)
        loss_elem = crit(pred, yb)          # (B,)
        loss = (loss_elem * wb).mean()      # 加權平均
        loss.backward()
        optim.step()

        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(1, n)

@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    if len(preds) == 0:
        return math.nan, math.nan, math.nan
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    rmse = math.sqrt(((y_pred - y_true) ** 2).mean())
    mae = np.abs(y_pred - y_true).mean()
    r2 = r2_score(y_true, y_pred)
    return r2, mae, rmse, y_pred

# -----------------------------
# Data preparation for LOSO
# -----------------------------
def prepare_subject_windows(
    df: pd.DataFrame, fps: int, seq_sec: int, stride_sec: int,
    use_pos: bool = False, pos_win: int = 30,
    args: argparse.Namespace = None,
    train_normalization: str = None,  # 新增：訓練時的標準化方式
    test_normalization: str = None    # 新增：測試時的標準化方式
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    回傳: subject -> (X, y)
    X shape: (num_windows, C, T)，其中 C=6 或 7（含 POS）
    y shape: (num_windows,)
    """
    seq_len = fps * seq_sec
    stride = max(1, fps * stride_sec)
    result = {}

    for folder, g in df.groupby("Folder", sort=False):
        R = g["COLOR_R"].values.astype(np.float32)
        G = g["COLOR_G"].values.astype(np.float32)
        B = g["COLOR_B"].values.astype(np.float32)
        SPO2 = g["SPO2"].values.astype(np.float32)
        N = len(g)
        if N < seq_len:
            continue

        rgb = np.vstack([R, G, B])
        chs = []

        # 6 通道
        # six_list = getsixchannels(rgb)
        # ch6 = np.stack(six_list, axis=0).astype(np.float32)  # (6,N)
        # chs.append(ch6)

        # 新增不變特徵
        inv = get_invariant_channels(rgb)   # (K, N)
        chs.append(inv)

        # optional POS
        if use_pos:
            ok, pos = getPOS(R.copy(), G.copy(), B.copy(), win_len=pos_win)
            if ok and pos.size == N:
                chs.append(pos.reshape(1, -1).astype(np.float32))
        # optional CCT
        if getattr(args, "use_cct", False):
            cct = getCCT(rgb)  # (1,N)
            chs.append(cct.astype(np.float32))

        X_all = np.concatenate(chs, axis=0)  # (C, N)  # <= 這裡的 C 可能是 6/7/8

        # 選 normalization 模式
        norm_mode = train_normalization or test_normalization  # 你原先的判斷
        mu_time, std_time = None, None

        if norm_mode == "subject":
            mu = X_all.mean(axis=1, keepdims=True)
            std = X_all.std(axis=1, keepdims=True) + 1e-8
            X_all = (X_all - mu) / std

        elif norm_mode == "cumulative":
            mu_time, std_time = welford_cumulative_stats(X_all)  # <== 改成對 X_all
        elif norm_mode == "ema":
            alpha = args.ema_alpha if args is not None else 0.05
            mu_time, std_time = ema_stats_over_time(X_all, alpha=alpha)
        # elif norm_mode == "segment": 切窗後再做（在 Dataset / numpy 端）

        # 切窗
        if norm_mode in ["cumulative", "ema"] and mu_time is not None:
            warmup_frames = max(0, int((args.warmup_sec if args is not None else 10) * fps))
            Xw, yw = build_windows_stride_frozen(
                X_all, SPO2, seq_len=seq_len, stride=stride,
                mu_time=mu_time, std_time=std_time, warmup_frames=warmup_frames
            )
        else:
            Xw, yw = build_windows(X_all, SPO2, seq_len=seq_len, stride=stride)

        if len(Xw):
            result[folder] = (Xw, yw)

    return result

# -----------------------------
# Fixed-mode utilities
# -----------------------------

def evaluate_test_with_trained_model(model, train_info: Dict[str, any], test_path: str, args, device, test_normalization: str = None, save_predictions: bool = False) -> Dict[str, float]:
    """使用已訓練的模型評估測試集"""
    expected_cols = {"Folder","COLOR_R","COLOR_G","COLOR_B","SPO2"}
    df_te = pd.read_csv(test_path)
    if not expected_cols.issubset(df_te.columns):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    subj_te = prepare_subject_windows(
        df_te, fps=args.fps, seq_sec=args.seq_sec, stride_sec=args.stride_sec,
        use_pos=args.use_pos, pos_win=args.pos_win,
        args=args, test_normalization=test_normalization
    )

    if len(subj_te) == 0:
        raise ValueError("No test subjects produced windows. Check seq_sec/stride_sec.")
    X_test = np.concatenate([v[0] for v in subj_te.values()], axis=0)
    y_test = np.concatenate([v[1] for v in subj_te.values()], axis=0)

    result_row = {
        "test_csv": os.path.basename(test_path),
        "model": args.model,
    }

    if args.model == "tcn":
        te_ds = SeqDataset(X_test, y_test, segment_normalization=True if args.test_normalization == "segment" else False)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        r2_te, mae_te, rmse_te, y_pred_te = eval_one(model, te_loader, device)
        print(f"[TEST][TCN][{os.path.basename(test_path)}] "
              f"R2={r2_te:.3f} | MAE={mae_te:.2f} | RMSE={rmse_te:.2f}")
        result_row.update({"R2": float(r2_te), "MAE": float(mae_te)})
        
        # 如果需要保存預測結果，需要重新計算預測值
        if save_predictions:
            result_row["y_true"] = y_test
            result_row["y_pred"] = y_pred_te
        
        return result_row
    else:
        X_test_np = X_test.copy()
        if args.test_normalization == "segment":
            X_test_np = apply_segment_norm_numpy(X_test_np)
        Xte = flatten_windows(X_test_np)
        yte = y_test

        y_pred_te = model.predict(Xte)
        rmse_te = math.sqrt(((y_pred_te - yte) ** 2).mean())
        mae_te = np.abs(y_pred_te - yte).mean()
        r2_te = r2_score(yte, y_pred_te)
        print(f"[TEST][{args.model.upper()}][{os.path.basename(test_path)}] "
              f"R2={r2_te:.3f} | MAE={mae_te:.2f} | RMSE={rmse_te:.2f}")
        result_row.update({"R2": float(r2_te), "MAE": float(mae_te)})
        
        # 保存預測結果
        if save_predictions:
            result_row["y_true"] = yte
            result_row["y_pred"] = y_pred_te
        
        return result_row

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # ---- 原有參數（保留） ----
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seq_sec", type=int, default=30, help="sequence length in seconds")
    parser.add_argument("--stride_sec", type=int, default=2, help="stride in seconds")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3, help="number of TCN residual blocks")
    parser.add_argument("--kernel", type=int, default=5, help="TCN kernel size")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val split from training subjects' windows")
    parser.add_argument("--ema_alpha", type=float, default=0.05,
                        help="EMA 模式下的 α，越大越快追隨變化 (0<α≤1)")
    parser.add_argument("--warmup_sec", type=int, default=10,
                        help="warm-up 時長（秒）；在此之前不產生輸出視窗")
    # 新增：分別控制訓練和測試的標準化方式
    parser.add_argument("--train_normalization", type=str, default=None,
                        choices=["none", "subject", "segment", "cumulative", "ema"],
                        help="訓練時的標準化方式：none | subject | segment | cumulative | ema")
    parser.add_argument("--test_normalization", type=str, default=None,
                        choices=["none", "subject", "segment", "cumulative", "ema"],
                        help="測試時的標準化方式：none | subject | segment | cumulative | ema")
    parser.add_argument("--use_pos", action="store_true",
                        help="Append POS as an extra input channel (no HR prediction).")
    parser.add_argument("--pos_win", type=int, default=30,
                        help="Window length (in samples) for POS overlap-add (default 30 @ 30fps = 1s).")
    parser.add_argument("--use_cct", action="store_true",
                        help="Append CCT (correlated color temperature) as an extra input channel.")
    # ---- 新增參數 ----
    parser.add_argument("--model", type=str, default="tcn",
                        choices=["tcn", "rf", "svr"],
                        help="選擇模型: tcn | rf (RandomForestRegressor) | svr (Support Vector Regressor)")
    parser.add_argument("--rf_estimators", type=int, default=10, help="RandomForest 樹數")
    parser.add_argument("--rf_max_depth", type=int, default=None, help="RandomForest 最大深度")
    parser.add_argument("--svr_kernel", type=str, default="rbf", help="SVR kernel")
    parser.add_argument("--svr_C", type=float, default=1.0, help="SVR C 參數")
    parser.add_argument("--svr_gamma", type=str, default="scale", help="SVR gamma: scale | auto 或數值")
    parser.add_argument("--train_csv", nargs='+', type=str, default=None,
                        help="One or more train CSV paths when --mode fixed")
    parser.add_argument("--test_csv", nargs='+', type=str, default=None,
                        help="One or more test CSV paths when --mode fixed")
    parser.add_argument("--use_label_weights", action="store_true",
                        help="啟用依 SpO2 分佈的加權 loss（訓練時）")
    parser.add_argument("--weight_gamma", type=float, default=1.0,
                        help="權重指數，>1 會更強化稀有區間；=1 為單純的反頻率")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction results for visualization")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Training Device: {args.device}")

    # --------- FIXED: 優化版本 - 相同訓練集只訓練一次 ---------
    if not args.train_csv or not args.test_csv:
        raise ValueError("When --mode fixed, please provide --train_csv and --test_csv.")

    # argparse 使用 nargs='+' 時會是 list；保險處理單一路徑字串
    train_list = args.train_csv if isinstance(args.train_csv, list) else [args.train_csv]
    test_list  = args.test_csv  if isinstance(args.test_csv, list)  else [args.test_csv]

    combo_rows = []
    
    # 合併所有訓練資料
    print(f"\n[COMBINING] 合併 {len(train_list)} 個訓練檔案...")
    all_subj_tr = {}
    
    for tr_path in train_list:
        print(f"[LOADING] {os.path.basename(tr_path)}")
        expected_cols = {"Folder","COLOR_R","COLOR_G","COLOR_B","SPO2"}
        df_tr = pd.read_csv(tr_path)
        if not expected_cols.issubset(df_tr.columns):
            raise ValueError(f"CSV must contain columns: {expected_cols}")
        
        subj_tr = prepare_subject_windows(
            df_tr, fps=args.fps, seq_sec=args.seq_sec, stride_sec=args.stride_sec,
            use_pos=args.use_pos, pos_win=args.pos_win,
            args=args, train_normalization=args.train_normalization
        )
        
        # 合併到總的訓練資料中，使用檔案名作為前綴避免受試者ID衝突
        file_prefix = os.path.splitext(os.path.basename(tr_path))[0]
        for subj_id, (X, y) in subj_tr.items():
            new_subj_id = f"{file_prefix}_{subj_id}"
            all_subj_tr[new_subj_id] = (X, y)
    
    if len(all_subj_tr) == 0:
        raise ValueError("No train subjects produced windows. Check seq_sec/stride_sec.")
    
    print(f"[COMBINED] 總共合併了 {len(all_subj_tr)} 個受試者的資料")
    
    # 使用合併後的資料訓練單一模型
    print(f"\n[TRAINING] 使用合併資料訓練單一模型...")
    X_train = np.concatenate([v[0] for v in all_subj_tr.values()], axis=0)
    y_train = np.concatenate([v[1] for v in all_subj_tr.values()], axis=0)

    n_train = len(X_train)
    idx = np.arange(n_train); np.random.shuffle(idx)
    split = int(n_train * (1.0 - args.val_ratio))
    tr_idx, va_idx = idx[:split], idx[split:]

    # ---- 產生訓練集 sample weight（只用於訓練，不影響驗證） ----
    train_weights = None
    if args.use_label_weights:
        train_weights_all = compute_sample_weights_from_labels(y_train, gamma=args.weight_gamma)
        train_weights = train_weights_all[tr_idx]  # 只取訓練子集

    train_info = {
        "X_train": X_train,
        "y_train": y_train,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "n_channels": X_train.shape[1]
    }

    if args.model == "tcn":
        tr_ds = SeqDataset(
            X_train[tr_idx], y_train[tr_idx],
            segment_normalization=True if args.train_normalization == "segment" else False,
            weights=train_weights
        )
        va_ds = SeqDataset(
            X_train[va_idx], y_train[va_idx],
            segment_normalization=True if args.train_normalization == "segment" else False,
            weights=None  # 驗證集不加權
        )

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = TCNRegressor(
            n_channels=train_info["n_channels"], hidden=args.hidden, levels=args.levels,
            kernel_size=args.kernel, dropout=args.dropout
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_va = float("inf"); best_state = None; patience, bad = args.patience, 0
        for ep in range(1, args.epochs + 1):
            tr_loss = train_one(model, tr_loader, optim, device)
            r2_va, mae_va, rmse_va, y_pred_va = eval_one(model, va_loader, device)
            if mae_va < best_va:
                best_va = mae_va
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(f"[TRAIN][TCN][合併資料] "
                  f"Epoch {ep:03d} | train MSE:{tr_loss:.4f} | val R2:{r2_va:.3f} MAE:{mae_va:.2f} RMSE:{rmse_va:.2f}")
            if bad >= patience:
                print("[TRAIN][TCN] Early stopping."); break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        print(f"[TRAINED] 模型訓練完成，最終 R2: {r2_va:.4f}, MAE: {mae_va:.4f}")
    else:
        # sklearn 模型
        X_train_np = X_train.copy()
        if args.train_normalization == "segment":
            X_train_np = apply_segment_norm_numpy(X_train_np)
        Xtr = flatten_windows(X_train_np[tr_idx])
        ytr = y_train[tr_idx]
        Xva = flatten_windows(X_train_np[va_idx])
        yva = y_train[va_idx]

        sample_weight = None
        if args.use_label_weights:
            train_weights_all = compute_sample_weights_from_labels(y_train, gamma=args.weight_gamma)
            sample_weight = train_weights_all[tr_idx]

        if args.model == "rf":
            model = RandomForestRegressor(n_estimators=args.rf_estimators,
                                          max_depth=args.rf_max_depth,
                                          n_jobs=-1,
                                          random_state=args.seed)
        elif args.model == "svr":
            gamma_val = args.svr_gamma
            try:
                gamma_val = float(gamma_val)
            except:
                gamma_val = "scale" if gamma_val == "scale" else "auto"
            model = SVR(kernel="rbf", gamma=gamma_val, C=args.svr_C)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        model.fit(Xtr, ytr, sample_weight=sample_weight)
        y_pred_va = model.predict(Xva)
        r2_va = r2_score(yva, y_pred_va)
        mae_va = np.mean(np.abs(yva - y_pred_va))
        rmse_va = np.sqrt(np.mean((yva - y_pred_va) ** 2))
        
        print(f"[TRAINED] 模型訓練完成，R2: {r2_va:.4f}, MAE: {mae_va:.4f}")
    
    # 使用單一模型測試所有測試資料
    for te_path in test_list:
        print(f"[TESTING] 合併模型 -> {os.path.basename(te_path)}")
        row = evaluate_test_with_trained_model(model, train_info, te_path, args, device, args.test_normalization, args.save_predictions)
        row["train_csv"] = "combined_all"  # 標記為合併訓練
        combo_rows.append(row)

    if len(combo_rows):
        df = pd.DataFrame(combo_rows)
        
        # 分別產出 R2 和 MAE 的 CSV
        r2_pivot = df.pivot(index='train_csv', columns='test_csv', values='R2')
        mae_pivot = df.pivot(index='train_csv', columns='test_csv', values='MAE')
        
        # 重置 index 名稱
        r2_pivot.index.name = 'train_file'
        mae_pivot.index.name = 'train_file'
        
        # 儲存 CSV 檔案
        r2_pivot.to_csv("fixed_combo_results_R2.csv")
        mae_pivot.to_csv("fixed_combo_results_MAE.csv")
        
        print(f"\nSaved: fixed_combo_results_R2.csv")
        print(f"Saved: fixed_combo_results_MAE.csv")
        
        # 保存預測結果（如果啟用）
        if args.save_predictions:
            predictions_data = []
            for _, row in df.iterrows():
                if 'y_true' in row and 'y_pred' in row:
                    y_true = row['y_true']
                    y_pred = row['y_pred']
                    for i in range(len(y_true)):
                        predictions_data.append({
                            'train_file': row['train_csv'],
                            'test_file': row['test_csv'],
                            'model': row['model'],
                            'true_label': y_true[i],
                            'prediction': y_pred[i]
                        })
            
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df.to_csv('predictions_results.csv', index=False)
                print(f"Saved: predictions_results.csv")
        

        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import TwoSlopeNorm
        
        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 處理欄位名稱：移除 .csv 後綴
        r2_pivot_clean = r2_pivot.copy()
        mae_pivot_clean = mae_pivot.copy()
        
        # 移除 columns 的 .csv 後綴
        r2_pivot_clean.columns = [col.replace('.csv', '') for col in r2_pivot_clean.columns]
        mae_pivot_clean.columns = [col.replace('.csv', '') for col in mae_pivot_clean.columns]
        
        # 移除 index 的 .csv 後綴
        r2_pivot_clean.index = [idx.replace('.csv', '') for idx in r2_pivot_clean.index]
        mae_pivot_clean.index = [idx.replace('.csv', '') for idx in mae_pivot_clean.index]
        
        # 創建子圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        r2_vals = r2_pivot_clean.values.astype(float)

        # 忽略 NaN 取 min/max
        vmin = np.nanmin(r2_vals)
        vmax = np.nanmax(r2_vals)

        if not (vmin < 0 < vmax):
            vmin = min(vmin, -1e-6)
            vmax = max(vmax,  1e-6)

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        # ---- 自訂 colorbar 刻度 ----
        neg_ticks = np.arange(np.floor(vmin/10)*10, 0, 10)
        pos_step = 0.05 if vmax <= 1 else vmax/4
        pos_ticks = np.arange(0, vmax + 1e-9, pos_step) if vmax > 0 else np.array([0.0])
        ticks = np.unique(np.concatenate([neg_ticks, [0.0], pos_ticks]))

        def format_colorbar_tick(value):
            return f"{value:.2f}" if -1 < value < 1 else f"{int(value)}"

        sns.heatmap(
            r2_pivot_clean, annot=True, fmt='.2f',
            cmap='RdYlGn', norm=norm, mask=np.isnan(r2_vals),
            ax=ax1,
            cbar_kws={'label': 'R²', 'ticks': ticks, 'format': matplotlib.ticker.FuncFormatter(lambda x, pos: format_colorbar_tick(x))},
            annot_kws={'fontsize': 8}
        )
        ax1.set_title('R² Heatmap')
        ax1.set_xlabel('Test')
        ax1.set_ylabel('Train')
        # 旋轉 x 軸標籤 45 度
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        
        # MAE 熱圖 - 修復顯示問題
        sns.heatmap(mae_pivot_clean, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                    center=10.0, ax=ax2, cbar_kws={'label': 'MAE'},
                    annot_kws={'fontsize': 8})
        ax2.set_title('MAE Heatmap')
        ax2.set_xlabel('Test')
        ax2.set_ylabel('Train')
        # 旋轉 x 軸標籤 45 度
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig('fixed_combo_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: fixed_combo_heatmap.png")


if __name__ == "__main__":
    main()
