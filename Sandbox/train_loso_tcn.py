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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
        ys.append(float(spo2[start:end].mean()))
    if len(xs) == 0:
        return np.zeros((0, C, seq_len), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y

def apply_segment_norm_numpy(X: np.ndarray) -> np.ndarray:
    """
    對 numpy 格式的 (N, C, T) 做每個樣本、每通道的 instance normalization。
    只對前六個特徵進行標準化，第七個特徵（POS）保持不變。
    """
    if X.size == 0:
        return X
    
    # 複製原始數據
    X_norm = X.copy()
    
    # 只對前六個通道進行標準化
    channels_to_norm = min(6, X.shape[1])  # 確保不超過實際通道數
    
    if channels_to_norm > 0:
        mu = X[:, :channels_to_norm, :].mean(axis=2, keepdims=True)
        std = X[:, :channels_to_norm, :].std(axis=2, keepdims=True) + 1e-8
        X_norm[:, :channels_to_norm, :] = (X[:, :channels_to_norm, :] - mu) / std
    
    return X_norm

def flatten_windows(X: np.ndarray) -> np.ndarray:
    """(N, C, T) -> (N, C*T) 用於 sklearn 模型。"""
    if X.ndim != 3:
        return X
    N, C, T = X.shape
    return X.reshape(N, C * T)

def streaming_zscore_cumulative(ch6: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    對 (C=6, N) 做「累積」z-score：
    對每個通道 c，t 時刻用 [0..t] 的均值與標準差做標準化。
    Welford 線上演算法，避免數值不穩。
    """
    C, N = ch6.shape
    out = np.empty_like(ch6, dtype=np.float32)

    for c in range(C):
        mean = 0.0
        M2 = 0.0
        for t in range(N):
            x = float(ch6[c, t])
            # 更新統計量（t 計數從 1 開始）
            n = t + 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2
            var = M2 / max(1, n - 1)  # 無偏估計；n=1 時先給極小方差
            std = math.sqrt(var) if var > 0 else 0.0
            out[c, t] = (x - mean) / (std + eps)
    return out

def streaming_zscore_ema(ch6: np.ndarray, alpha: float = 0.05, eps: float = 1e-8) -> np.ndarray:
    """
    對 (C=6, N) 做 EMA z-score：
    mu_t = α x_t + (1-α) mu_{t-1}
    var_t 用 EMA 近似：v_t = α (x_t - mu_t)^2 + (1-α) v_{t-1}
    """
    C, N = ch6.shape
    out = np.empty_like(ch6, dtype=np.float32)

    for c in range(C):
        mu = float(ch6[c, 0])
        v  = 0.0
        out[c, 0] = 0.0
        for t in range(1, N):
            x = float(ch6[c, t])
            mu = alpha * x + (1.0 - alpha) * mu
            diff = x - mu
            v   = alpha * (diff * diff) + (1.0 - alpha) * v
            std = math.sqrt(v)
            out[c, t] = diff / (std + eps)
    return out

# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, segment_normalization: bool = False):
        """
        X: (N, C, T), y: (N,)
        segment_normalization=True 時，對每個樣本做 (x - mean_c)/std_c，按通道 c 各自計算。
        """
        self.X = torch.from_numpy(X)        # 直接先轉 tensor，提高效率
        self.y = torch.from_numpy(y)
        self.segment_normalization = segment_normalization

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.segment_normalization:
            x = x.clone()  # ★ 先複製，避免動到底層儲存
            channels_to_norm = min(6, x.shape[0])
            if channels_to_norm > 0:
                mu  = x[:channels_to_norm].mean(dim=-1, keepdim=True)
                std = x[:channels_to_norm].std(dim=-1, keepdim=True).clamp_min(1e-8)
                x[:channels_to_norm] = (x[:channels_to_norm] - mu) / std
        return x, self.y[idx]

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
    crit = nn.MSELoss()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad()
        pred = model(xb)
        loss = crit(pred, yb)
        loss.backward()
        optim.step()
        total += loss.item() * len(xb)
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
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
    return r2, mae, rmse

# -----------------------------
# Data preparation for LOSO
# -----------------------------
def prepare_subject_windows(
    df: pd.DataFrame, fps: int, seq_sec: int, stride_sec: int,
    zscore_within_subject: bool = True,
    use_pos: bool = False, pos_win: int = 30,
    args: argparse.Namespace = None
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

        # 6 通道轉換（無濾波）
        rgb = np.vstack([R, G, B])
        six_list = getsixchannels(rgb)
        ch6 = np.stack(six_list, axis=0).astype(np.float32)  # (6, N)

        # 受試者內 z-score
        if zscore_within_subject and args.online_norm == "none":
            mu = ch6.mean(axis=1, keepdims=True)
            std = ch6.std(axis=1, keepdims=True) + 1e-8
            ch6 = (ch6 - mu) / std

        # ★ 新增：線上正規化（只處理前 6 通道）
        if args.online_norm != "none":
            if args.online_norm == "cumulative":
                ch6 = streaming_zscore_cumulative(ch6)
            elif args.online_norm == "ema":
                ch6 = streaming_zscore_ema(ch6, alpha=args.ema_alpha)

        # 產生 POS 通道（可選）
        if use_pos:
            ok, pos = getPOS(R.copy(), G.copy(), B.copy(), win_len=pos_win)
            if ok and pos.size == N:
                pos_ch = pos.reshape(1, -1).astype(np.float32)  # (1, N)
                p_mu = pos_ch.mean(axis=1, keepdims=True)
                pos_ch = pos_ch - p_mu
                X_all = np.concatenate([ch6, pos_ch], axis=0)  # (7, N)
            else:
                # 失敗就只用 6 通道
                X_all = ch6
        else:
            X_all = ch6

        # 切視窗
        Xw, yw = build_windows(X_all, SPO2, seq_len=seq_len, stride=stride)
        if len(Xw):
            result[folder] = (Xw, yw)

    return result

def concat_except(subject_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], exclude_key: str):
    Xs, ys = [], []
    for k, (X, y) in subject_dict.items():
        if k == exclude_key: 
            continue
        Xs.append(X); ys.append(y)
    if len(Xs) == 0:
        return np.zeros((0,)), np.zeros((0,))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

# -----------------------------
# Fixed-mode utilities
# -----------------------------
def train_model_once(train_path: str, args, device) -> Tuple[any, Dict[str, any]]:
    """訓練一次模型，返回模型和訓練資料資訊"""
    expected_cols = {"Folder","COLOR_R","COLOR_G","COLOR_B","SPO2"}
    df_tr = pd.read_csv(train_path)
    if not expected_cols.issubset(df_tr.columns):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    subj_tr = prepare_subject_windows(
        df_tr, fps=args.fps, seq_sec=args.seq_sec, stride_sec=args.stride_sec,
        zscore_within_subject=args.subject_normalization,
        use_pos=args.use_pos, pos_win=args.pos_win,
        args=args
    )

    if len(subj_tr) == 0:
        raise ValueError("No train subjects produced windows. Check seq_sec/stride_sec.")
    X_train = np.concatenate([v[0] for v in subj_tr.values()], axis=0)
    y_train = np.concatenate([v[1] for v in subj_tr.values()], axis=0)

    n_train = len(X_train)
    idx = np.arange(n_train); np.random.shuffle(idx)
    split = int(n_train * (1.0 - args.val_ratio))
    tr_idx, va_idx = idx[:split], idx[split:]

    train_info = {
        "X_train": X_train,
        "y_train": y_train,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "n_channels": X_train.shape[1]
    }

    if args.model == "tcn":
        tr_ds = SeqDataset(X_train[tr_idx], y_train[tr_idx], segment_normalization=args.segment_normalization)
        va_ds = SeqDataset(X_train[va_idx], y_train[va_idx], segment_normalization=args.segment_normalization)

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
            r2_va, mae_va, rmse_va = eval_one(model, va_loader, device)
            if rmse_va < best_va:
                best_va = rmse_va
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(f"[TRAIN][TCN][{os.path.basename(train_path)}] "
                  f"Epoch {ep:03d} | train MSE:{tr_loss:.4f} | val R2:{r2_va:.3f} MAE:{mae_va:.2f} RMSE:{rmse_va:.2f}")
            if bad >= patience:
                print("[TRAIN][TCN] Early stopping."); break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        return model, train_info
    else:
        # sklearn 模型
        X_train_np = X_train.copy()
        if args.segment_normalization:
            X_train_np = apply_segment_norm_numpy(X_train_np)
        Xtr = flatten_windows(X_train_np[tr_idx])
        ytr = y_train[tr_idx]
        Xva = flatten_windows(X_train_np[va_idx])
        yva = y_train[va_idx]

        if args.model == "rf":
            model = RandomForestRegressor(n_estimators=args.rf_estimators,
                                          max_depth=args.rf_max_depth,
                                          n_jobs=-1,
                                          random_state=args.seed)
        elif args.model == "svr":
            gamma_val = args.svr_gamma
            try:
                gamma_val = float(args.svr_gamma)
            except Exception:
                pass
            model = SVR(kernel=args.svr_kernel, C=args.svr_C, gamma=gamma_val)
        else:
            raise ValueError("Unknown sklearn model")

        model.fit(Xtr, ytr)
        y_pred_va = model.predict(Xva)
        rmse_va = math.sqrt(((y_pred_va - yva) ** 2).mean())
        mae_va = np.abs(y_pred_va - yva).mean()
        r2_va = r2_score(yva, y_pred_va)
        print(f"[TRAIN][{args.model.upper()}][{os.path.basename(train_path)}] "
              f"val R2:{r2_va:.3f} MAE:{mae_va:.2f} RMSE:{rmse_va:.2f}")
        
        return model, train_info

def evaluate_test_with_trained_model(model, train_info: Dict[str, any], test_path: str, args, device) -> Dict[str, float]:
    """使用已訓練的模型評估測試集"""
    expected_cols = {"Folder","COLOR_R","COLOR_G","COLOR_B","SPO2"}
    df_te = pd.read_csv(test_path)
    if not expected_cols.issubset(df_te.columns):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    subj_te = prepare_subject_windows(
        df_te, fps=args.fps, seq_sec=args.seq_sec, stride_sec=args.stride_sec,
        zscore_within_subject=args.subject_normalization,
        use_pos=args.use_pos, pos_win=args.pos_win,
        args=args
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
        te_ds = SeqDataset(X_test, y_test, segment_normalization=args.segment_normalization)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        r2_te, mae_te, rmse_te = eval_one(model, te_loader, device)
        print(f"[TEST][TCN][{os.path.basename(test_path)}] "
              f"R2={r2_te:.3f} | MAE={mae_te:.2f} | RMSE={rmse_te:.2f}")
        result_row.update({"R2": float(r2_te), "MAE": float(mae_te)})
        return result_row
    else:
        X_test_np = X_test.copy()
        if args.segment_normalization:
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
        return result_row

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # ---- 原有參數（保留） ----
    parser.add_argument("--csv", type=str, help="Path to data.csv (LOSO mode)")
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
    parser.add_argument("--subject_normalization", action="store_true", help="enable per-subject z-score normalization")
    parser.add_argument("--segment_normalization", action="store_true",
                        help="Apply per-window, per-channel instance normalization on inputs.")
    parser.add_argument("--online_norm", type=str, default="none",
                        choices=["none", "cumulative", "ema"],
                        help="即時正規化模式：none | cumulative(累積) | ema(指數)")
    parser.add_argument("--ema_alpha", type=float, default=0.05,
                        help="EMA 模式下的 α，越大越快追隨變化 (0<α≤1)")
    parser.add_argument("--use_pos", action="store_true",
                        help="Append POS as an extra input channel (no HR prediction).")
    parser.add_argument("--pos_win", type=int, default=30,
                        help="Window length (in samples) for POS overlap-add (default 30 @ 30fps = 1s).")
    # ---- 新增參數 ----
    parser.add_argument("--model", type=str, default="tcn",
                        choices=["tcn", "rf", "svr"],
                        help="選擇模型: tcn | rf (RandomForestRegressor) | svr (Support Vector Regressor)")
    parser.add_argument("--rf_estimators", type=int, default=10, help="RandomForest 樹數")
    parser.add_argument("--rf_max_depth", type=int, default=None, help="RandomForest 最大深度")
    parser.add_argument("--svr_kernel", type=str, default="rbf", help="SVR kernel")
    parser.add_argument("--svr_C", type=float, default=1.0, help="SVR C 參數")
    parser.add_argument("--svr_gamma", type=str, default="scale", help="SVR gamma: scale | auto 或數值")
    parser.add_argument("--mode", type=str, default="loso",
                        choices=["loso", "fixed"],
                        help="Evaluation mode: 'loso' (leave-one-subject-out) or 'fixed' (train/test CSV).")
    parser.add_argument("--train_csv", nargs='+', type=str, default=None,
                        help="One or more train CSV paths when --mode fixed")
    parser.add_argument("--test_csv", nargs='+', type=str, default=None,
                        help="One or more test CSV paths when --mode fixed")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Training Device: {args.device}")

    expected_cols = {"Folder","COLOR_R","COLOR_G","COLOR_B","SPO2"}

    if args.mode == "fixed":
        # --------- FIXED: 優化版本 - 相同訓練集只訓練一次 ---------
        if not args.train_csv or not args.test_csv:
            raise ValueError("When --mode fixed, please provide --train_csv and --test_csv.")

        # argparse 使用 nargs='+' 時會是 list；保險處理單一路徑字串
        train_list = args.train_csv if isinstance(args.train_csv, list) else [args.train_csv]
        test_list  = args.test_csv  if isinstance(args.test_csv, list)  else [args.test_csv]

        combo_rows = []
        trained_models = {}  # 快取已訓練的模型
        
        for tr_path in train_list:
            if tr_path not in trained_models:
                print(f"\n[TRAINING] {os.path.basename(tr_path)}")
                model, train_info = train_model_once(tr_path, args, device)
                trained_models[tr_path] = (model, train_info)
            else:
                print(f"\n[REUSING] {os.path.basename(tr_path)}")
                model, train_info = trained_models[tr_path]
            
            for te_path in test_list:
                # 跳過相同的訓練和測試檔案（避免資料洩漏）
                if tr_path == te_path:
                    print(f"[SKIP] {os.path.basename(tr_path)} -> {os.path.basename(te_path)} (相同檔案，跳過)")
                    continue
                
                print(f"[TESTING] {os.path.basename(tr_path)} -> {os.path.basename(te_path)}")
                row = evaluate_test_with_trained_model(model, train_info, te_path, args, device)
                row["train_csv"] = os.path.basename(tr_path)  # 補上訓練檔案名稱
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
            
            # 生成熱圖
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
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
                
                # R2 熱圖
                sns.heatmap(r2_pivot_clean, annot=True, fmt='.2f', cmap='RdYlGn', 
                           center=0.0, ax=ax1, cbar_kws={'label': 'R2 Score'},
                           annot_kws={'fontsize': 8})
                ax1.set_title('R2 Score Heatmap')
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
                
            except ImportError:
                print("注意：無法生成熱圖，請安裝 matplotlib 和 seaborn")
                print("pip install matplotlib seaborn")
        return

    # --------- LOSO（維持你原本流程，僅小調整） ---------
    if not args.csv:
        raise ValueError("When --mode loso, please provide --csv.")
    df = pd.read_csv(args.csv)
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    subj_data = prepare_subject_windows(
        df, fps=args.fps, seq_sec=args.seq_sec, stride_sec=args.stride_sec,
        zscore_within_subject=args.subject_normalization,
        use_pos=args.use_pos, pos_win=args.pos_win,
        args=args
    )
    subjects = list(subj_data.keys())
    print(f"[Info] Subjects with enough data: {len(subjects)} -> {subjects[:5]}{'...' if len(subjects)>5 else ''}")

    all_rows = []
    for test_subj in subjects:
        X_train_np, y_train_np = concat_except(subj_data, exclude_key=test_subj)
        X_test_np,  y_test_np  = subj_data[test_subj]
        if len(X_test_np) == 0 or len(X_train_np) == 0:
            continue

        n_train = len(X_train_np)
        idx = np.arange(n_train); np.random.shuffle(idx)
        split = int(n_train * (1.0 - args.val_ratio))
        tr_idx, va_idx = idx[:split], idx[split:]

        if args.model == "tcn":
            tr_ds = SeqDataset(X_train_np[tr_idx], y_train_np[tr_idx], segment_normalization=args.segment_normalization)
            va_ds = SeqDataset(X_train_np[va_idx], y_train_np[va_idx], segment_normalization=args.segment_normalization)
            te_ds = SeqDataset(X_test_np,        y_test_np,        segment_normalization=args.segment_normalization)

            tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
            va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            n_channels = X_train_np.shape[1]
            model = TCNRegressor(
                n_channels=n_channels, hidden=args.hidden, levels=args.levels,
                kernel_size=args.kernel, dropout=args.dropout
            ).to(device)

            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_va = float("inf"); best_state = None; patience, bad = 10, 0
            for ep in range(1, args.epochs + 1):
                tr_loss = train_one(model, tr_loader, optim, device)
                r2_va, mae_va, rmse_va = eval_one(model, va_loader, device)
                if rmse_va < best_va:
                    best_va = rmse_va
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                print(f"[{test_subj}][TCN] Epoch {ep:03d} | train MSE:{tr_loss:.4f} | val R2:{r2_va:.3f} MAE:{mae_va:.2f} RMSE:{rmse_va:.2f}")
                if bad >= patience:
                    print(f"[{test_subj}][TCN] Early stopping."); break

            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            r2_te, mae_te, rmse_te = eval_one(model, te_loader, device)
            row = {"test_subject": test_subj, "n_test_windows": len(X_test_np),
                   "R2": float(r2_te), "MAE": float(mae_te), "RMSE": float(rmse_te)}
            all_rows.append(row)
            print(f"[TEST {test_subj}][TCN] R2={row['R2']:.3f} | MAE={row['MAE']:.2f} | RMSE={row['RMSE']:.2f}")
        else:
            Xtr_full = X_train_np.copy()
            Xte_full = X_test_np.copy()
            if args.segment_normalization:
                Xtr_full = apply_segment_norm_numpy(Xtr_full)
                Xte_full = apply_segment_norm_numpy(Xte_full)
            Xtr = flatten_windows(Xtr_full[tr_idx])
            ytr = y_train_np[tr_idx]
            Xva = flatten_windows(Xtr_full[va_idx])
            yva = y_train_np[va_idx]
            Xte = flatten_windows(Xte_full)
            yte = y_test_np

            if args.model == "rf":
                model = RandomForestRegressor(n_estimators=args.rf_estimators,
                                              max_depth=args.rf_max_depth,
                                              n_jobs=-1,
                                              random_state=args.seed)
            elif args.model == "svr":
                gamma_val = args.svr_gamma
                try:
                    gamma_val = float(args.svr_gamma)
                except Exception:
                    pass
                model = SVR(kernel=args.svr_kernel, C=args.svr_C, gamma=gamma_val)
            else:
                raise ValueError("Unknown sklearn model")

            model.fit(Xtr, ytr)
            y_pred_va = model.predict(Xva)
            rmse_va = math.sqrt(((y_pred_va - yva) ** 2).mean())
            mae_va = np.abs(y_pred_va - yva).mean()
            r2_va = r2_score(yva, y_pred_va)
            print(f"[{test_subj}][{args.model.upper()}] val R2:{r2_va:.3f} MAE:{mae_va:.2f} RMSE:{rmse_va:.2f}")

            y_pred_te = model.predict(Xte)
            rmse_te = math.sqrt(((y_pred_te - yte) ** 2).mean())
            mae_te = np.abs(y_pred_te - yte).mean()
            r2_te = r2_score(yte, y_pred_te)
            row = {"test_subject": test_subj, "n_test_windows": len(X_test_np),
                   "R2": float(r2_te), "MAE": float(mae_te), "RMSE": float(rmse_te)}
            all_rows.append(row)
            print(f"[TEST {test_subj}][{args.model.upper()}] R2={row['R2']:.3f} | MAE={row['MAE']:.2f} | RMSE={row['RMSE']:.2f}")

    if len(all_rows):
        out = pd.DataFrame(all_rows).sort_values("R2", ascending=False)
        print("\n=== LOSO Summary ===")
        print(out.to_string(index=False))
        print("\nAverages:")
        print(f"mean R2  : {out['R2'].mean():.3f}")
        print(f"mean MAE : {out['MAE'].mean():.2f}")
        print(f"mean RMSE: {out['RMSE'].mean():.2f}")
        out.to_csv("loso_tcn_results.csv", index=False)
        print("\nSaved: loso_tcn_results.csv")
    else:
        print("No results produced (check data/window settings).")

if __name__ == "__main__":
    main()
