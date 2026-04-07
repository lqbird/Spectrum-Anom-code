"""
Multi-type Baselines on SpectrumDet Features (with feature-wise Transformer + our tail module)
----------------------------------------------------------------------------------------------
依赖:
    pip install numpy pandas scikit-learn torch matplotlib scipy

数据假设:
    - CSV 第 0 列: 时间索引 (不作为特征)
    - 中间列: 特征
    - 最后一列: 标签 y ∈ {0,1} (0=正常, 1=异常)
    - 行顺序即时间顺序

训练:
    - 前 30% 时间步作为无监督训练集 (仅用特征, 不用标签)
    - 后 70% 作为测试集 (用标签评估)

方法:
    - 传统一类检测: OCSVM, IsoForest, LOF, KNN_Dist
    - 深度序列自编码器 (PyTorch):
        - CNN_AE (时间序列滑窗)
        - LSTM_AE (时间序列滑窗)
        - Transformer_AE (特征维 self-attention, 逐样本重构)

阈值:
    - baseline: 固定 test quantile + 简单时间平滑
    - ours: log1p + 因果滚动稳健 z + POT-GPD + stable alarm
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import genpareto

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================
# 全局配置
# =============================

CSV_PATH = "E:/博博work/VScode/Transformer_based_change/正异常特征结果5.14/abnormal_timeseries_csv/ab1/feature_levelS_FM_result_set_retime_new.csv"

TRAIN_FRAC = 0.3                  # 前 30% 作为训练
TEST_ANOMALY_Q = 0.95             # baseline: 使用 test scores 的 95% 分位数作为阈值

IOU_THRESHOLD = 0.1               # 事件匹配 IoU 阈值
TIME_PER_STEP = 1.0               # 每个时间步的时间长度

RANDOM_SEED = 42

# 稳健 z-score 滚动窗口
ROBUST_WINDOW = 144

# 我们模块里的 POT 超参
TAIL_FRAC_START = 0.98
TAIL_FRAC_MIN   = 0.90
TAIL_FRAC_STEP  = 0.01
MIN_EXCEED      = 50
RISK            = 1e-3
WINSOR_Q        = 98.5
FALLBACK_Q      = 98.0

# stable alarm 超参
N_CONSECUTIVE     = 2
HYSTERESIS_WINDOW = 3

# 序列 AE 超参 (CNN/LSTM)
WINDOW_SIZE   = 144
WINDOW_STRIDE = 6
EPOCHS_AE     = 40
BATCH_SIZE_AE = 128
LR_AE         = 1e-3
CNN_HIDDEN_CHANNELS = 32
LSTM_HIDDEN_DIM     = 64

# 特征维 Transformer AE 超参
TRANS_MODEL_DIM   = 32
TRANS_NUM_HEADS   = 4
TRANS_NUM_LAYERS  = 2
EPOCHS_TRANS      = 20
BATCH_SIZE_TRANS  = 32
LR_TRANS          = 1e-3


# =============================
# 事件工具
# =============================

def extract_events(labels):
    labels = np.asarray(labels).astype(int)
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_event:
            in_event = True
            start = i
        elif v == 0 and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(labels) - 1))
    return events


def match_events(true_events, pred_events, iou_threshold=0.1):
    nT = len(true_events)
    nP = len(pred_events)
    if nT == 0 or nP == 0:
        return []

    iou_mat = np.zeros((nT, nP), dtype=np.float32)
    for i, (ts, te) in enumerate(true_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(te, pe) - max(ts, ps) + 1)
            if inter > 0:
                union = (te - ts + 1) + (pe - ps + 1) - inter
                iou_mat[i, j] = inter / union

    matches = []
    iou_copy = iou_mat.copy()
    while True:
        idx = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
        i, j = idx
        best_iou = iou_copy[i, j]
        if best_iou < iou_threshold:
            break
        matches.append((i, j, best_iou))
        iou_copy[i, :] = 0.0
        iou_copy[:, j] = 0.0

    return matches


def enforce_min_consecutive(y, min_len=2):
    y = np.asarray(y).astype(int).copy()
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j + 1 < n and y[j + 1] == 1:
                j += 1
            run_len = j - i + 1
            if run_len < min_len:
                y[i:j + 1] = 0
            i = j + 1
        else:
            i += 1
    return y


def close_small_gaps(y, max_gap=2):
    y = np.asarray(y).astype(int).copy()
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 0:
            j = i
            while j + 1 < n and y[j + 1] == 0:
                j += 1
            gap_len = j - i + 1
            left_one = (i - 1 >= 0 and y[i - 1] == 1)
            right_one = (j + 1 < n and y[j + 1] == 1)
            if left_one and right_one and gap_len <= max_gap:
                y[i:j + 1] = 1
            i = j + 1
        else:
            i += 1
    return y


def compute_event_metrics(y_true, y_pred, time_per_step=1.0, iou_threshold=0.1):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    true_events = extract_events(y_true)
    pred_events = extract_events(y_pred)

    matches = match_events(true_events, pred_events, iou_threshold=iou_threshold)

    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(true_events) - tp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    mean_iou = float(np.mean([m[2] for m in matches])) if tp > 0 else 0.0

    delays = []
    for true_idx, pred_idx, _ in matches:
        ts, _ = true_events[true_idx]
        ps, _ = pred_events[pred_idx]
        delay_steps = abs(ps - ts)
        delays.append(delay_steps * time_per_step)
    ttd = float(np.mean(delays)) if len(delays) > 0 else None

    return {
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": f1,
        "mean_iou": mean_iou,
        "ttd": ttd,
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
        "n_matched": tp,
    }


def compute_event_hit_rate(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    true_events = extract_events(y_true)
    if len(true_events) == 0:
        return 0.0

    hits = 0
    for (s, e) in true_events:
        if (y_pred[s:e + 1] == 1).any():
            hits += 1
    return hits / len(true_events)


def plot_prediction_timeline(y_true, y_pred, method_name, save_prefix=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    T = len(y_true)
    t = np.arange(T)

    plt.figure(figsize=(14, 3))
    plt.fill_between(
        t, 1, 2,
        where=(y_true == 1),
        alpha=0.5,
        label="True anomaly"
    )
    plt.fill_between(
        t, 0, 1,
        where=(y_pred == 1),
        alpha=0.5,
        label="Predicted anomaly"
    )

    plt.yticks([0.5, 1.5], ["Pred", "True"])
    plt.ylim(-0.1, 2.1)
    plt.xlabel("Test time index")
    plt.title(f"Anomaly segments: {method_name}")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_prefix is not None:
        out_path = f"{save_prefix}_{method_name}.png"
        plt.savefig(out_path, dpi=150)
        print(f"[Plot] Saved: {out_path}")

    plt.close()


def plot_scores_and_labels(scores, y_true, y_pred, thr, method_name, save_prefix=None, ylabel="Anomaly score"):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    T = len(scores)
    t = np.arange(T)

    plt.figure(figsize=(14, 4))
    plt.plot(t, scores, label="Anomaly score")
    plt.axhline(thr, linestyle="--", label=f"Threshold={thr:.3f}")

    idx_true = np.where(y_true == 1)[0]
    if len(idx_true) > 0:
        plt.scatter(idx_true, scores[idx_true],
                    marker="x", s=50, label="True anomaly", zorder=3)

    idx_pred = np.where(y_pred == 1)[0]
    if len(idx_pred) > 0:
        plt.scatter(idx_pred, scores[idx_pred],
                    marker="o", facecolors="none", edgecolors="r",
                    s=40, label="Pred anomaly", zorder=3)

    plt.xlabel("Test time index")
    plt.ylabel(ylabel)
    plt.title(f"Scores & threshold: {method_name}")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_prefix is not None:
        out_path = f"{save_prefix}_{method_name}.png"
        plt.savefig(out_path, dpi=150)
        print(f"[Plot] Saved: {out_path}")

    plt.close()


def plot_roc_pr_curves(y_true, scores, method_name, save_prefix="roc_pr"):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        fpr, tpr, roc_auc = None, None, np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        precision, recall, pr_auc = None, None, np.nan

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if fpr is not None:
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC: {method_name}")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "ROC not defined", ha="center", va="center")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: {method_name}")

    plt.subplot(1, 2, 2)
    if precision is not None:
        plt.plot(recall, precision, label=f"AP={pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR: {method_name}")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "PR not defined", ha="center", va="center")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR: {method_name}")

    plt.tight_layout()
    out_path = f"{save_prefix}_{method_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Plot] Saved: {out_path}")
    plt.close()

    return roc_auc, pr_auc


# =============================
# 我们的 robust-z + POT + stable alarm 模块
# =============================

def robust_z_causal(values, window):
    values = np.asarray(values, dtype=np.float64)
    z = np.zeros_like(values, dtype=np.float64)
    eps = 1e-9
    n = len(values)
    for t in range(n):
        a = max(0, t - window + 1)
        w = values[a:t + 1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        z[t] = (values[t] - med) / (1.4826 * mad + eps)
    return z.astype(np.float32)


def pot_threshold_auto(scores,
                       tail_start=TAIL_FRAC_START,
                       tail_min=TAIL_FRAC_MIN,
                       step=TAIL_FRAC_STEP,
                       min_exceed=MIN_EXCEED,
                       risk=RISK,
                       winsor_q=WINSOR_Q,
                       fallback_q=FALLBACK_Q):
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return 0.0

    upper = np.percentile(s, winsor_q)
    s_fit = np.clip(s, None, upper)

    thr = None
    tail = tail_start
    while tail >= tail_min:
        u = np.quantile(s_fit, tail)
        exceed = s_fit[s_fit > u] - u
        if exceed.size >= min_exceed:
            try:
                xi, loc, scale = genpareto.fit(exceed, floc=0)
                p_exceed = exceed.size / s_fit.size
                if p_exceed > 0 and risk < p_exceed:
                    F_target = 1.0 - risk / p_exceed
                    F_target = max(1e-9, min(1 - 1e-9, F_target))
                    thr = float(u + genpareto.ppf(F_target, xi, loc=0, scale=scale))
                else:
                    thr = float(u)
            except Exception:
                thr = None
            if thr is not None:
                break
        tail -= step

    if thr is None:
        thr = float(np.percentile(s_fit, fallback_q))
    return thr


def stable_alarm_from_raw(raw_pred, n_consecutive=N_CONSECUTIVE, hyst_window=HYSTERESIS_WINDOW):
    raw_pred = np.asarray(raw_pred).astype(int)
    stable = np.zeros_like(raw_pred)
    i = 0
    n = len(raw_pred)
    while i < n - n_consecutive + 1:
        if np.all(raw_pred[i:i + n_consecutive] == 1):
            start = i
            i += n_consecutive
            while i + hyst_window <= n:
                if np.any(raw_pred[i:i + hyst_window] == 1):
                    i += hyst_window
                else:
                    break
            stable[start:i] = 1
        else:
            i += 1
    return stable


# =============================
# 深度 AE 模型
# =============================

class CNNAE(nn.Module):
    def __init__(self, feat_dim, hidden_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, feat_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)      # [B, D, L]
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.transpose(1, 2)  # [B, L, D]
        return out


class LSTMAE(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feat_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h)
        return out


class TransformerFeatAE(nn.Module):
    """
    特征维 Transformer 自编码器:
        - 输入: [B, D]
        - 在特征维 D 上做 self-attention
        - 输出: [B, D]
    """
    def __init__(self, feat_dim, model_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, x, src_mask=None):
        # x: [B, D]
        x = x.unsqueeze(-1)            # [B, D, 1]
        x = self.embedding(x)          # [B, D, M]
        x = x.transpose(0, 1)          # [D, B, M]
        if src_mask is not None:
            h = self.encoder(x, mask=src_mask)
        else:
            h = self.encoder(x)
        h = h.transpose(0, 1)          # [B, D, M]
        out = self.decoder(h).squeeze(-1)  # [B, D]
        return out


# =============================
# 序列 AE 工具 (CNN/LSTM)
# =============================

def build_windows(X, window_size, stride):
    T = len(X)
    windows = []
    indices = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(X[start:end])
        indices.append((start, end))
    if not windows:
        return np.empty((0, window_size, X.shape[1]), dtype=X.dtype), []
    X_win = np.stack(windows, axis=0)
    return X_win, indices


def train_seq_ae(model, X_win, device="cpu",
                 epochs=40, batch_size=128, lr=1e-3,
                 name="AE"):
    dataset = TensorDataset(torch.from_numpy(X_win.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"[{name}] Epoch {ep + 1}/{epochs}, train MSE: {avg_loss:.6f}")

    return model


def seq_ae_point_scores(model, X_win, indices, X_full,
                        device="cpu", batch_size=256):
    model.eval()
    T, D = X_full.shape
    sum_err = np.zeros((T, D), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, len(X_win), batch_size):
            xb_np = X_win[i:i + batch_size].astype(np.float32)
            xb = torch.from_numpy(xb_np).to(device)
            recon = model(xb).cpu().numpy()

            for b_idx, rec in enumerate(recon):
                start, end = indices[i + b_idx]
                L = end - start
                err = (X_full[start:end] - rec[:L]) ** 2
                sum_err[start:end] += err
                cnt[start:end] += 1.0

    cnt = np.maximum(cnt, 1.0)[:, None]
    mse = (sum_err / cnt).mean(axis=1)
    return mse


# =============================
# 特征维 Transformer 训练 & 打分
# =============================

def train_feat_ae(model, X_train, device="cpu",
                  epochs=20, batch_size=32, lr=1e-3,
                  name="TF-AE"):
    dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"[{name}] Epoch {ep + 1}/{epochs}, train MSE: {avg_loss:.6f}")

    return model


def feat_ae_scores(model, X, device="cpu", batch_size=256):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb_np = X[i:i + batch_size].astype(np.float32)
            xb = torch.from_numpy(xb_np).to(device)
            recon = model(xb).cpu().numpy()
            err = ((xb_np - recon) ** 2).mean(axis=1)
            errs.append(err)
    return np.concatenate(errs, axis=0)


# =============================
# 主流程
# =============================

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ---------- 1. 读数据 ----------
    df = pd.read_csv(CSV_PATH)
    data = df.values
    # 关键改动: 丢弃第 0 列时间索引, 与你自己的方法保持一致
    X_all = data[:, 1:-1]
    y_all = data[:, -1].astype(int)

    n_total = len(X_all)
    split_idx = int(n_total * TRAIN_FRAC)

    X_train_raw = X_all[:split_idx]
    X_test_raw  = X_all[split_idx:]
    y_train     = y_all[:split_idx]
    y_test      = y_all[split_idx:]

    print(f"Total samples: {n_total}, train: {len(X_train_raw)}, test: {len(X_test_raw)}")
    print("Label stats (all) :", np.bincount(y_all))
    print("Label stats (test):", np.bincount(y_test))

    # ---------- 2. 标准化 ----------
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    methods_results = []

    # 评估函数在 main 内部闭包, 直接访问 y_test & methods_results
    def evaluate_baseline(name, scores_test):
        nonlocal methods_results
        scores_test = np.asarray(scores_test, dtype=float)

        thr = float(np.quantile(scores_test, TEST_ANOMALY_Q))

        y_raw = (scores_test >= thr).astype(int)
        anomaly_rate_raw = y_raw.mean()

        y_tmp = enforce_min_consecutive(y_raw, min_len=2)
        y_tmp = close_small_gaps(y_tmp, max_gap=2)
        if y_tmp.sum() == 0 and y_raw.sum() > 0:
            print(f"[Info] {name}: all anomalies removed by smoothing+gap-closing, fallback to raw mask.")
            y_pred = y_raw
        else:
            y_pred = y_tmp

        anomaly_rate_final = y_pred.mean()

        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        ev = compute_event_metrics(
            y_test, y_pred,
            time_per_step=TIME_PER_STEP,
            iou_threshold=IOU_THRESHOLD
        )
        hit_rate = compute_event_hit_rate(y_test, y_pred)

        roc_auc, pr_auc = plot_roc_pr_curves(
            y_true=y_test,
            scores=scores_test,
            method_name=name,
            save_prefix="roc_pr_baseline"
        )

        res = {
            "name": name,
            "threshold": thr,
            "point_precision": p,
            "point_recall": r,
            "point_f1": f1,
            "event_precision": ev["event_precision"],
            "event_recall": ev["event_recall"],
            "event_f1": ev["event_f1"],
            "mean_iou": ev["mean_iou"],
            "ttd": ev["ttd"],
            "n_true_events": ev["n_true_events"],
            "n_pred_events": ev["n_pred_events"],
            "n_matched": ev["n_matched"],
            "event_hit_rate": hit_rate,
            "anomaly_rate_raw": anomaly_rate_raw,
            "anomaly_rate": anomaly_rate_final,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        methods_results.append(res)

        print(f"\n=== {name} (baseline quantile) ===")
        print(f"Threshold (Q={TEST_ANOMALY_Q}): {thr:.6f}")
        print(f"Anomaly rate (raw):   {anomaly_rate_raw:.4f}")
        print(f"Anomaly rate (final): {anomaly_rate_final:.4f}")
        print(f"Point-level:  P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        print(f"Event-level:  P={ev['event_precision']:.4f}, "
              f"R={ev['event_recall']:.4f}, F1={ev['event_f1']:.4f}")
        print(f"Mean IoU (matched): {ev['mean_iou']:.4f}")
        print(f"TTD (avg): {ev['ttd']}")
        print(f"Event hit rate: {hit_rate:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        plot_prediction_timeline(
            y_true=y_test,
            y_pred=y_pred,
            method_name=name,
            save_prefix="anomaly_segments_baseline"
        )
        plot_scores_and_labels(
            scores=scores_test,
            y_true=y_test,
            y_pred=y_pred,
            thr=thr,
            method_name=name,
            save_prefix="scores_baseline",
            ylabel="Anomaly score"
        )

    def evaluate_ours(name, scores_train, scores_test_raw):
        nonlocal methods_results
        scores_train = np.asarray(scores_train, dtype=float)
        scores_test_raw = np.asarray(scores_test_raw, dtype=float)

        all_scores = np.concatenate([scores_train, scores_test_raw])
        all_log = np.log1p(all_scores)
        all_z = robust_z_causal(all_log, ROBUST_WINDOW)

        z_train = all_z[:len(scores_train)]
        z_test  = all_z[len(scores_train):]

        thr = pot_threshold_auto(z_train)
        print(f"[Info] {name}_ours: POT threshold on train robust-z: {thr:.6f}")

        y_raw = (z_test > thr).astype(int)
        anomaly_rate_raw = y_raw.mean()

        y_pred = stable_alarm_from_raw(y_raw, n_consecutive=N_CONSECUTIVE,
                                       hyst_window=HYSTERESIS_WINDOW)
        anomaly_rate_final = y_pred.mean()

        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        ev = compute_event_metrics(
            y_test, y_pred,
            time_per_step=TIME_PER_STEP,
            iou_threshold=IOU_THRESHOLD
        )
        hit_rate = compute_event_hit_rate(y_test, y_pred)

        roc_auc, pr_auc = plot_roc_pr_curves(
            y_true=y_test,
            scores=z_test,
            method_name=f"{name}_ours",
            save_prefix="roc_pr_ours"
        )

        res = {
            "name": f"{name}_ours",
            "threshold": thr,
            "point_precision": p,
            "point_recall": r,
            "point_f1": f1,
            "event_precision": ev["event_precision"],
            "event_recall": ev["event_recall"],
            "event_f1": ev["event_f1"],
            "mean_iou": ev["mean_iou"],
            "ttd": ev["ttd"],
            "n_true_events": ev["n_true_events"],
            "n_pred_events": ev["n_pred_events"],
            "n_matched": ev["n_matched"],
            "event_hit_rate": hit_rate,
            "anomaly_rate_raw": anomaly_rate_raw,
            "anomaly_rate": anomaly_rate_final,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        methods_results.append(res)

        print(f"\n=== {name}_ours (log1p + robust-z + POT + stable alarm) ===")
        print(f"POT threshold: {thr:.6f}")
        print(f"Anomaly rate (raw):   {anomaly_rate_raw:.4f}")
        print(f"Anomaly rate (final): {anomaly_rate_final:.4f}")
        print(f"Point-level:  P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        print(f"Event-level:  P={ev['event_precision']:.4f}, "
              f"R={ev['event_recall']:.4f}, F1={ev['event_f1']:.4f}")
        print(f"Mean IoU (matched): {ev['mean_iou']:.4f}")
        print(f"TTD (avg): {ev['ttd']}")
        print(f"Event hit rate: {hit_rate:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        plot_prediction_timeline(
            y_true=y_test,
            y_pred=y_pred,
            method_name=f"{name}_ours",
            save_prefix="anomaly_segments_ours"
        )
        plot_scores_and_labels(
            scores=z_test,
            y_true=y_test,
            y_pred=y_pred,
            thr=thr,
            method_name=f"{name}_ours",
            save_prefix="scores_ours",
            ylabel="Robust-z score"
        )

    # ---------- 3. 传统一类检测 ----------
    print("\n[Train] One-Class SVM ...")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_train)
    scores_test = -ocsvm.decision_function(X_test)
    evaluate_baseline("OCSVM", scores_test)

    print("\n[Train] Isolation Forest ...")
    isof = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    isof.fit(X_train)
    scores_test = -isof.score_samples(X_test)
    evaluate_baseline("IsoForest", scores_test)

    print("\n[Train] LOF (novelty=True) ...")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        novelty=True,
        contamination="auto",
        n_jobs=-1
    )
    lof.fit(X_train)
    scores_test = -lof.score_samples(X_test)
    evaluate_baseline("LOF", scores_test)

    print("\n[Train] KNN Distance ...")
    k = 10
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_train)
    dist_test, _ = nn.kneighbors(X_test, n_neighbors=k)
    scores_test = dist_test.mean(axis=1)
    evaluate_baseline("KNN_Dist", scores_test)

    # ---------- 4. 深度 AE ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Info] Using device: {device}")

    # 4.1 CNN/LSTM 基于滑窗
    X_train_win, train_indices = build_windows(X_train, WINDOW_SIZE, WINDOW_STRIDE)
    X_test_win, test_indices   = build_windows(X_test,  WINDOW_SIZE, WINDOW_STRIDE)

    print(f"Train windows: {len(X_train_win)}, Test windows: {len(X_test_win)}, "
          f"window_size={WINDOW_SIZE}, stride={WINDOW_STRIDE}")

    if len(X_train_win) > 0 and len(X_test_win) > 0:
        feat_dim = X_train.shape[1]

        print("\n[Train] CNN-AE ...")
        cnn_model = CNNAE(feat_dim, hidden_channels=CNN_HIDDEN_CHANNELS)
        cnn_model = train_seq_ae(
            cnn_model, X_train_win, device=device,
            epochs=EPOCHS_AE, batch_size=BATCH_SIZE_AE,
            lr=LR_AE, name="CNN-AE"
        )
        scores_train = seq_ae_point_scores(
            cnn_model, X_train_win, train_indices, X_train,
            device=device, batch_size=256
        )
        scores_test = seq_ae_point_scores(
            cnn_model, X_test_win, test_indices, X_test,
            device=device, batch_size=256
        )
        evaluate_baseline("CNN_AE", scores_test)
        evaluate_ours("CNN_AE", scores_train, scores_test)

        print("\n[Train] LSTM-AE ...")
        lstm_model = LSTMAE(feat_dim, hidden_dim=LSTM_HIDDEN_DIM, num_layers=1)
        lstm_model = train_seq_ae(
            lstm_model, X_train_win, device=device,
            epochs=EPOCHS_AE, batch_size=BATCH_SIZE_AE,
            lr=LR_AE, name="LSTM-AE"
        )
        scores_train = seq_ae_point_scores(
            lstm_model, X_train_win, train_indices, X_train,
            device=device, batch_size=256
        )
        scores_test = seq_ae_point_scores(
            lstm_model, X_test_win, test_indices, X_test,
            device=device, batch_size=256
        )
        evaluate_baseline("LSTM_AE", scores_test)
        evaluate_ours("LSTM_AE", scores_train, scores_test)
    else:
        print("\n[Warning] 序列长度不足以构造滑窗, CNN/LSTM AE 未运行。")

    # 4.2 特征维 Transformer-AE (与你自己的方法同构)
    print("\n[Train] Transformer-AE (feature-wise) ...")
    feat_dim = X_train.shape[1]
    trans_model = TransformerFeatAE(
        feat_dim,
        model_dim=TRANS_MODEL_DIM,
        num_heads=TRANS_NUM_HEADS,
        num_layers=TRANS_NUM_LAYERS
    )
    trans_model = train_feat_ae(
        trans_model, X_train,
        device=device,
        epochs=EPOCHS_TRANS,
        batch_size=BATCH_SIZE_TRANS,
        lr=LR_TRANS,
        name="Transformer-AE"
    )
    scores_train = feat_ae_scores(trans_model, X_train, device=device, batch_size=256)
    scores_test  = feat_ae_scores(trans_model, X_test,  device=device, batch_size=256)
    evaluate_baseline("Transformer_AE", scores_test)
    evaluate_ours("Transformer_AE", scores_train, scores_test)

    # ---------- 5. 汇总 & 可视化 ----------
    res_df = pd.DataFrame(methods_results)
    print("\n===== Summary (baseline quantile + our module on CNN/LSTM/Transformer) =====")
    print(res_df[[
        "name",
        "threshold",
        "point_precision", "point_recall", "point_f1",
        "event_precision", "event_recall", "event_f1",
        "mean_iou", "ttd",
        "event_hit_rate",
        "anomaly_rate",
        "roc_auc", "pr_auc",
    ]])

    res_df.to_csv("baseline_results_with_ours_transformer_feat.csv", index=False)
    print("\nSaved results to baseline_results_with_ours_transformer_feat.csv")

    plt.figure(figsize=(10, 4))
    x = np.arange(len(res_df))
    width = 0.35
    plt.bar(x - width/2, res_df["point_f1"], width=width, label="Point F1")
    plt.bar(x + width/2, res_df["event_f1"], width=width, label="Event F1")
    plt.xticks(x, res_df["name"], rotation=45, ha="right")
    plt.ylabel("F1 score")
    plt.title("Baseline vs Ours (F1 comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_ours_f1_bar.png", dpi=150)
    print("Saved F1 bar plot to baseline_ours_f1_bar.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    x = np.arange(len(res_df))
    width = 0.35
    plt.bar(x - width/2, res_df["roc_auc"], width=width, label="ROC-AUC")
    plt.bar(x + width/2, res_df["pr_auc"], width=width, label="PR-AUC")
    plt.xticks(x, res_df["name"], rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.title("Baseline vs Ours (AUC comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_ours_auc_bar.png", dpi=150)
    print("Saved AUC bar plot to baseline_ours_auc_bar.png")
    plt.close()


if __name__ == "__main__":
    main()
