import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from eval_tools import (
    fz0_per_step,
    exact_var_factor,
    exact_es_factor,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
)
from eval_tools import plot_var_es_diagnostics

# ============================
# Shared config (fair compare)
# ============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ALPHA = 0.01
TRAIN_FRAC = 0.85
CONTEXT_LEN = 64  # unified with SRNN
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3


# ============================
# Model definitions
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


class BasicVaRTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim=32,
        num_heads=2,
        num_layers=1,
        dropout=0.2,
        paper_exact=True,  # keep the flag if you like; we'll always use the coherent map below
    ):
        super().__init__()
        self.paper_exact = paper_exact

        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        # simple linear head to 2 logits (a,b)
        self.output_layer = nn.Linear(model_dim, 2)
        with torch.no_grad():
            # small negative bias nudges v,e into sensible region from the start
            self.output_layer.bias[:] = torch.tensor(
                [-0.5, 0.5]
            )  # any small values are fine

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        h = self.transformer(x)  # [B, T, D]
        h_last = h[:, -1, :]  # last time step

        raw = self.output_layer(h_last)  # [B, 2] -> (a,b)
        a, b = raw[:, 0], raw[:, 1]

        # Coherent mapping: v < 0, e < v
        v = -F.softplus(a)
        e = v - F.softplus(b)
        return torch.stack([v, e], dim=1)  # [B, 2]


class FZ0Loss(nn.Module):
    def __init__(self, alpha=ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        v, e = y_pred[:, 0], y_pred[:, 1]  # outputs from coherent head
        ind = (y_true <= v).float()
        # Coherent head guarantees e < v < 0, so -e > 0; no clamp needed.
        term1 = -(ind * (v - y_true)) / (self.alpha * e)
        term2 = (v / e) + torch.log(-e)
        return (term1 + term2).mean()


# ============================
# Data utils
# ============================
def create_sequences_with_overlap(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = CONTEXT_LEN,
    overlap: float = 0.3,
    add_noise: bool = True,
    noise_std: float = 0.01,
):
    step = max(1, int(seq_len * (1 - overlap)))
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_len, step):
        window = X[i : i + seq_len]
        if add_noise:
            window = window + np.random.normal(0, noise_std, window.shape)
        X_seq.append(window)
        y_seq.append(y[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ============================
# Train / Eval
# ============================
def train_with_overlapping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    alpha=ALPHA,
    seq_len: int = CONTEXT_LEN,
    paper_exact: bool = True,
) -> BasicVaRTransformer:
    X_seq, y_seq = create_sequences_with_overlap(
        X_train, y_train, seq_len=seq_len, overlap=0.98, add_noise=False
    )
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.float32)

    split = int(0.8 * len(X_t))
    train_ds = TensorDataset(X_t[:split], y_t[:split])
    val_ds = TensorDataset(X_t[split:], y_t[split:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BasicVaRTransformer(
        input_dim=input_dim,
        model_dim=32,
        num_heads=2,
        num_layers=1,
        dropout=0.2,
        paper_exact=paper_exact,
    )
    loss_fn = FZ0Loss(alpha=alpha)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = StepLR(optim, step_size=20, gamma=0.9)

    best_val, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(yb, pred)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optim.step()
                tr += loss.item()
        tr /= max(1, len(train_loader))

        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(yb, pred)
                if torch.isfinite(loss):
                    va += loss.item()
        va /= max(1, len(val_loader))

        if va < best_val:
            best_val, best_state, bad = va, model.state_dict(), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_with_expanding_window(
    model: BasicVaRTransformer, X_all, y_all, start_idx: int, seq_len: int = CONTEXT_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    var_list, es_list = [], []
    for t in range(start_idx, len(X_all)):
        left = max(0, t - seq_len)
        window = X_all[left:t]
        if len(window) < seq_len:
            continue
        x_t = torch.tensor(window.reshape(1, seq_len, -1), dtype=torch.float32)
        with torch.no_grad():
            pred = model(x_t).cpu().numpy()[0]
        var_list.append(pred[0])
        es_list.append(pred[1])
    return np.array(var_list), np.array(es_list)


# ============================
# Pipeline
# ============================
def pipeline(
    csv_path="data/merged_data_with_realised_volatility.csv",
    alpha=ALPHA,
    feature_parity=True,
    calibrate=False,
    run_tag=None,
    out_dir="saved_models",
    fig_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)
    if run_tag:
        fig_dir = os.path.join(fig_dir, run_tag)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Unified target/feature parity with SRNN
    df["log_ret"] = np.log(df["close"]).diff()
    df["target_return"] = df["log_ret"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    split = int(TRAIN_FRAC * n)
    mu_train = df.loc[: split - 1, "log_ret"].mean()
    df["x_cov"] = (df["log_ret"] - mu_train) ** 2

    if feature_parity:
        X = df[["x_cov"]].values
    else:
        X = df[["log_ret", "x_cov"]].values

    y = df["target_return"].values.astype(np.float32)

    # Standardize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X[:split])
    Xtr = scaler.transform(X[:split]).astype(np.float32)
    Xte = scaler.transform(X[split:]).astype(np.float32)

    # Train
    model = train_with_overlapping(
        Xtr,
        y[:split],
        input_dim=X.shape[1],
        alpha=alpha,
        seq_len=CONTEXT_LEN,
        paper_exact=True,
    )

    # Evaluate (raw)
    v_pred, e_pred = evaluate_with_expanding_window(
        model, np.vstack([Xtr, Xte]), y, start_idx=split, seq_len=CONTEXT_LEN
    )
    y_aligned = y[split : split + len(v_pred)]

    if calibrate:
        # exact-factor calibration on train sequences (optional)
        X_cal, y_cal = create_sequences_with_overlap(
            Xtr, y[:split], seq_len=CONTEXT_LEN, overlap=0.3, add_noise=False
        )
        with torch.no_grad():
            cal_pred = model(torch.tensor(X_cal, dtype=torch.float32)).cpu().numpy()
        v_tr, e_tr = cal_pred[:, 0], cal_pred[:, 1]
        c_v = exact_var_factor(y_cal, v_tr, alpha)
        c_e = exact_es_factor(y_cal, v_tr * c_v, e_tr, alpha)

        v_eval = v_pred * c_v
        e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)
        base = f"transformer_{run_tag+'_ ' if run_tag else ''}calibrated".replace(
            " ", ""
        )
        title = "Transformer (paper-exact, parity, calibrated)"
    else:
        v_eval, e_eval = v_pred, e_pred
        base = f"transformer_{run_tag+'_ ' if run_tag else ''}raw".replace(" ", "")
        title = "Transformer (paper-exact, parity, raw)"

    hits = (y_aligned <= v_eval).astype(int)
    LR_pof, p_pof, _, _ = kupiec_pof(hits, alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    LR_cc, p_cc = christoffersen_cc(hits, alpha)
    fz0 = fz0_per_step(y_aligned, v_eval, e_eval, alpha)

    print("=" * 60)
    print(title + (f"  [{run_tag}]" if run_tag else ""))
    print("=" * 60)
    print(f"Hit rate: {hits.mean():.4f} (Target {alpha:.4f})")
    print(f"Kupiec: LR={LR_pof:.4f}, p={p_pof:.4f}")
    print(f"Christoffersen IND: LR={LR_ind:.4f}, p={p_ind:.4f}")
    print(f"Christoffersen CC : LR={LR_cc:.4f}, p={p_cc:.4f}")
    print(f"Avg FZ0: {fz0.mean():.6f}")

    # Save arrays/metrics with tagged names
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=hits,
    )

    plot_var_es_diagnostics(
        y_true=y_aligned,
        var_pred=v_eval,
        es_pred=e_eval,
        alpha=alpha,
        title=title,
        out_dir=fig_dir,
        fname_prefix=base,
    )

    metrics = dict(
        model=title,
        alpha=float(alpha),
        hit_rate=float(hits.mean()),
        kupiec_LR=float(LR_pof),
        kupiec_p=float(p_pof),
        ind_LR=float(LR_ind),
        ind_p=float(p_ind),
        cc_LR=float(LR_cc),
        cc_p=float(p_cc),
        avg_fz0=float(fz0.mean()),
        tag=run_tag or "",
    )
    with open(os.path.join(out_dir, f"{base}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics, (v_eval, e_eval, y_aligned, fz0)


if __name__ == "__main__":
    pipeline()
