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

def per_subject_zscore(ch6: np.ndarray) -> np.ndarray:
    """
    ch6: (6, N) six-channel series for ONE subject.
    z-score each channel within-subject to remove baseline/scale differences.
    """
    mu = ch6.mean(axis=1, keepdims=True)
    std = ch6.std(axis=1, keepdims=True) + 1e-8
    return (ch6 - mu) / std

def build_windows(
    X_all: np.ndarray, spo2: np.ndarray, seq_len: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_all: (7, N), spo2: (N,)
    Return X: (num_windows, 7, seq_len), y: (num_windows,)
    Label = mean SpO2 within the window (可依需求改成最後一點)
    """
    N = X_all.shape[1]
    xs, ys = [], []
    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        xs.append(X_all[:, start:end])
        ys.append(float(spo2[start:end].mean()))
    if len(xs) == 0:
        return np.zeros((0, 7, seq_len), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y

# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, win_instancenorm: bool = False):
        """
        X: (N, C, T), y: (N,)
        win_instancenorm=True 時，對每個樣本做 (x - mean_c)/std_c，按通道 c 各自計算。
        """
        self.X = torch.from_numpy(X)        # 直接先轉 tensor，提高效率
        self.y = torch.from_numpy(y)
        self.win_instancenorm = win_instancenorm

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]    # shape: (C, T) - torch.float32
        if self.win_instancenorm:
            # 每個通道各自做零均值、單位變異，避免分母為 0
            mu = x.mean(dim=-1, keepdim=True)                 # (C, 1)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-8) # (C, 1)
            x = (x - mu) / std
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
    use_pos: bool = False, pos_win: int = 30
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

        # 受試者內 z-score（建議保留；若你要關閉可用 --no_zscore）
        if zscore_within_subject:
            mu = ch6.mean(axis=1, keepdims=True)
            std = ch6.std(axis=1, keepdims=True) + 1e-8
            ch6 = (ch6 - mu) / std

        # 產生 POS 通道（可選）
        if use_pos:
            ok, pos = getPOS(R.copy(), G.copy(), B.copy(), win_len=pos_win)
            if ok and pos.size == N:
                pos_ch = pos.reshape(1, -1).astype(np.float32)  # (1, N)
                # POS 也做受試者內 z-score，讓尺度對齊
                if zscore_within_subject:
                    p_mu = pos_ch.mean(axis=1, keepdims=True)
                    p_std = pos_ch.std(axis=1, keepdims=True) + 1e-8
                    pos_ch = (pos_ch - p_mu) / p_std
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
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to data.csv")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seq_sec", type=int, default=30, help="sequence length in seconds")
    parser.add_argument("--stride_sec", type=int, default=2, help="stride in seconds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3, help="number of TCN residual blocks")
    parser.add_argument("--kernel", type=int, default=5, help="TCN kernel size")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val split from training subjects' windows")
    parser.add_argument("--no_zscore", action="store_true", help="disable per-subject z-score normalization")
    parser.add_argument("--win_instancenorm", action="store_true",
                        help="Apply per-window, per-channel instance normalization on inputs.")
    parser.add_argument("--use_pos", action="store_true",
                        help="Append POS as an extra input channel (no HR prediction).")
    parser.add_argument("--pos_win", type=int, default=30,
                        help="Window length (in samples) for POS overlap-add (default 30 @ 30fps = 1s).")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load
    df = pd.read_csv(args.csv)
    # Basic sanity check
    expected_cols = {"Folder","COLOR_R","COLOR_G","COLOR_B","SPO2"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    # Prepare windows per subject
    subj_data = prepare_subject_windows(
        df, fps=args.fps, seq_sec=args.seq_sec, stride_sec=args.stride_sec,
        zscore_within_subject=(not args.no_zscore),
        use_pos=args.use_pos, pos_win=args.pos_win
    )
    subjects = list(subj_data.keys())
    print(f"Training Device: {args.device}")
    print(f"[Info] Subjects with enough data: {len(subjects)} -> {subjects[:5]}{'...' if len(subjects)>5 else ''}")

    # LOSO
    device = torch.device(args.device)
    all_rows = []
    for test_subj in subjects:
        X_train_np, y_train_np = concat_except(subj_data, exclude_key=test_subj)
        X_test_np, y_test_np = subj_data[test_subj]

        if len(X_test_np) == 0 or len(X_train_np) == 0:
            continue

        # Train/Val split (random over windows from training subjects)
        n_train = len(X_train_np)
        idx = np.arange(n_train)
        np.random.shuffle(idx)
        split = int(n_train * (1.0 - args.val_ratio))
        tr_idx, va_idx = idx[:split], idx[split:]

        Xtr = torch.from_numpy(X_train_np[tr_idx])  # (N,6,T)
        ytr = torch.from_numpy(y_train_np[tr_idx])
        Xva = torch.from_numpy(X_train_np[va_idx])
        yva = torch.from_numpy(y_train_np[va_idx])
        Xte = torch.from_numpy(X_test_np)
        yte = torch.from_numpy(y_test_np)

        tr_ds = SeqDataset(Xtr.numpy(), ytr.numpy(), win_instancenorm=args.win_instancenorm)
        va_ds = SeqDataset(Xva.numpy(), yva.numpy(), win_instancenorm=args.win_instancenorm)
        te_ds = SeqDataset(Xte.numpy(), yte.numpy(), win_instancenorm=args.win_instancenorm)


        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = TCNRegressor(
            n_channels=7 if args.use_pos else 6, hidden=args.hidden, levels=args.levels,
            kernel_size=args.kernel, dropout=args.dropout
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_va = float("inf")
        best_state = None
        patience, bad = 10, 0  # early stopping

        for ep in range(1, args.epochs + 1):
            tr_loss = train_one(model, tr_loader, optim, device)
            r2_va, mae_va, rmse_va = eval_one(model, va_loader, device)
            if rmse_va < best_va:
                best_va = rmse_va
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(f"[{test_subj}] Epoch {ep:03d} | train MSE:{tr_loss:.4f} | val R2:{r2_va:.3f} MAE:{mae_va:.2f} RMSE:{rmse_va:.2f}")
            if bad >= patience:
                print(f"[{test_subj}] Early stopping.")
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        r2_te, mae_te, rmse_te = eval_one(model, te_loader, device)

        row = {
            "test_subject": test_subj,
            "n_test_windows": len(X_test_np),
            "R2": float(r2_te),
            "MAE": float(mae_te),
            "RMSE": float(rmse_te),
        }
        all_rows.append(row)
        print(f"[TEST {test_subj}] R2={row['R2']:.3f} | MAE={row['MAE']:.2f} | RMSE={row['RMSE']:.2f}")

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
