# srnn_ve3_repo_style.py
# ------------------------------------------------------------
# Repo-style SRNN-VE-3 (as per "VaR-and-peace" README):
#  • RNN (hidden=10), hybrid head: concat[h_t, sqrt_pos(h_t)] -> Linear -> (a,b)
#  • Coherent outputs: v=-softplus(a), e=v-softplus(b)  => e < v < 0
#  • FZ0 loss + optional penalties toward GARCH "true" VaR/ES if present
#  • Stateful sequence training over ordered chunks, carry hidden across chunks
#  • Optional exact calibration on train (your eval_tools)
# ------------------------------------------------------------

import os
import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_tools import (
    fz0_per_step,
    exact_var_factor,
    exact_es_factor,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    plot_var_es_diagnostics,
)

# ============================
# Config
# ============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ALPHA = 0.01
TRAIN_FRAC = 0.85

# Stateful sequence training (repo-style)
SEQ_LEN = 128  # try {64,128,256}
BATCH_SIZE = 16  # try {8,16,32}

MAX_EPOCHS = 200
PATIENCE = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 0.5
DROPOUT_P = 0.2
HIDDEN = 10  # VE-3 per README: hidden size ~10  :contentReference[oaicite:2]{index=2}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Penalty weights (see repo README: small stabilizers toward GARCH targets)  :contentReference[oaicite:3]{index=3}
LAMBDA_VaR_MSE = 0.5  # deviation penalty
LAMBDA_VaR_L1 = 0.1  # confidence penalty (L1)


# ============================
# Model (SRNN-VE-3)
# ============================
class LinearRNN(nn.Module):
    """Linear RNN with variational (time-constant) input dropout."""

    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        # Unconstrained (paper baseline style)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h0=None):
        """
        x:  [B, T, input_size]
        h0: [B, H] or None
        returns: out: [B, T, H], h_last: [B, H]
        """
        B, T, _ = x.shape

        # Variational input dropout: one mask per sequence (shared across time)
        if self.training and self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            mask = x.new_empty(B, 1, self.input_size).bernoulli_(keep) / keep
            x = x * mask

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h = h0 if h0.dim() == 2 else h0.squeeze(0)

        outs = []
        for t in range(T):
            xt = x[:, t, :]
            h = h @ self.W_hh.T + xt @ self.W_xh.T + self.b_h
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), h


class SRNNVE3(nn.Module):
    """
    SRNN-VE-3 (as summarized in the repo):
      - recurrent hidden state h_t (H=10)
      - hybrid head uses both h_t and sqrt-transformed branch
      - coherent mapping to (v, e) with softplus
    """

    def __init__(self, hidden_size: int = HIDDEN, dropout_p: float = DROPOUT_P):
        super().__init__()
        self.rnn = LinearRNN(input_size=1, hidden_size=hidden_size, dropout_p=dropout_p)
        # Head: concat [h_t, sqrt_pos(h_t)]  -> Linear(2H -> 2)
        self.head = nn.Linear(2 * hidden_size, 2)

        # Mildly negative VaR/ES at start (similar to neutral init)
        def softplus_inv(x):
            import math

            return math.log(math.exp(x) - 1.0)

        with torch.no_grad():
            # start with small weights; bias so v≈-0.02, e≈v-0.01
            nn.init.zeros_(self.head.weight)
            self.head.bias[:] = torch.tensor(
                [softplus_inv(0.02), softplus_inv(0.01)], dtype=torch.float32
            )

    @staticmethod
    def _sqrt_pos(h, eps=1e-8):
        # ensure non-negativity before sqrt: softplus for smooth positivity
        return torch.sqrt(F.softplus(h) + eps)

    def forward(self, x, h_prev=None):
        """
        x: [B, T, 1]
        returns:
          y_all: [B, T, 2]   (per-step (v,e))
          h_n:   [B, H]
        """
        rnn_out, h_n = self.rnn(x, h_prev)  # [B,T,H]
        z1 = rnn_out  # [B,T,H]
        z2 = self._sqrt_pos(rnn_out)  # [B,T,H]
        z = torch.cat([z1, z2], dim=-1)  # [B,T,2H]
        logits = self.head(z)  # [B,T,2]
        a, b = logits.unbind(dim=-1)  # each [B,T]
        v = -F.softplus(a)  # v < 0
        e = v - F.softplus(b)  # e < v
        y_all = torch.stack([v, e], dim=-1)
        return y_all, h_n


# ============================
# Loss (FZ0 + optional penalties toward GARCH targets)
# ============================
class FZ0WithPenalties(nn.Module):
    """
    FZ0(y_true, v, e) + λ1 * MSE(v, var_true) + λ2 * L1(v, var_true)
    If var_true/es_true are not provided, penalties are skipped.
    """

    def __init__(
        self, alpha=ALPHA, lambda_var_mse=LAMBDA_VaR_MSE, lambda_var_l1=LAMBDA_VaR_L1
    ):
        super().__init__()
        self.alpha = alpha
        self.lambda_var_mse = lambda_var_mse
        self.lambda_var_l1 = lambda_var_l1

    def forward(
        self,
        y_true: torch.Tensor,  # [B,T]
        y_pred: torch.Tensor,  # [B,T,2]
        var_true: Optional[torch.Tensor] = None,  # [B,T] or None
        es_true: Optional[
            torch.Tensor
        ] = None,  # [B,T] or None (unused here, but wired if you extend)
    ):
        v, e = y_pred[..., 0], y_pred[..., 1]  # [B,T]
        yt = y_true
        if yt.dim() == 1:
            yt = yt.unsqueeze(0).expand_as(v)
        # FZ0
        ind = (yt <= v).float()
        term1 = -(ind * (v - yt)) / (self.alpha * e)
        term2 = (v / e) + torch.log(-e)
        loss = (term1 + term2).mean()

        # penalties (if GARCH targets provided)
        if var_true is not None:
            if var_true.dim() == 1:
                var_true = var_true.unsqueeze(0).expand_as(v)
            loss = loss + self.lambda_var_mse * F.mse_loss(v, var_true)
            loss = loss + self.lambda_var_l1 * torch.mean(torch.abs(v - var_true))
        return loss


# ============================
# Data
# ============================
def build_inputs_from_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    df["target_return"] = df["log_ret"].shift(-1)  # next step
    df = df.dropna().reset_index(drop=True)
    return df


def split_and_make_feature(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    n = len(df)
    split = int(train_frac * n)
    mu_train = df.loc[: split - 1, "log_ret"].mean()
    df["x_cov"] = (
        df["log_ret"] - mu_train
    ) ** 2  # repo-style: squared returns (no scaling)  :contentReference[oaicite:4]{index=4}

    x = df["x_cov"].values.reshape(-1, 1).astype(np.float32)
    y = df["target_return"].values.astype(np.float32)

    # Optional GARCH targets if present in CSV (column names you can adapt)
    var_true = (
        df["var_true"].values.astype(np.float32) if "var_true" in df.columns else None
    )
    es_true = (
        df["es_true"].values.astype(np.float32) if "es_true" in df.columns else None
    )

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    vt_train = var_true[:split] if var_true is not None else None
    vt_test = var_true[split:] if var_true is not None else None
    et_train = es_true[:split] if es_true is not None else None
    et_test = es_true[split:] if es_true is not None else None

    meta = {"split_idx": split, "mu_train": float(mu_train)}
    return (
        (x_train, y_train, vt_train, et_train),
        (x_test, y_test, vt_test, et_test),
        meta,
    )


# ============================
# Stateful sequence batching (no shuffling)
# ============================
def build_stateful_streams(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seq_len: int,
    var_true: Optional[np.ndarray] = None,
    es_true: Optional[np.ndarray] = None,
):
    """
    Turn a single long series into B parallel streams (contiguous), trim to multiple of seq_len.
    Returns tensors ready for training: X:[B,S,1], Y:[B,S], VT:[B,S]? ET:[B,S]?
    """
    N = len(x)
    steps_per_stream = N // batch_size
    S = (steps_per_stream // seq_len) * seq_len  # trim so S % seq_len == 0
    total = batch_size * S

    Xs = x[:total].reshape(batch_size, S, 1)
    Ys = y[:total].reshape(batch_size, S)

    VTs = ETs = None
    if var_true is not None:
        VTs = var_true[:total].reshape(batch_size, S)
    if es_true is not None:
        ETs = es_true[:total].reshape(batch_size, S)

    num_chunks = S // seq_len
    return Xs, Ys, VTs, ETs, num_chunks, S


# ============================
# Train / Eval
# ============================
def train_srnn_stateful_sequences(
    x_train: np.ndarray,
    y_train: np.ndarray,
    var_true: Optional[np.ndarray],
    es_true: Optional[np.ndarray],
    alpha: float = ALPHA,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    dropout_p: float = DROPOUT_P,
) -> SRNNVE3:
    model = SRNNVE3(hidden_size=HIDDEN, dropout_p=dropout_p).to(DEVICE)
    loss_fn = FZ0WithPenalties(alpha=alpha)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # time split inside train for early stopping
    split = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:split], y_train[:split]
    x_va, y_va = x_train[split:], y_train[split:]
    vt_tr = var_true[:split] if (var_true is not None) else None
    vt_va = var_true[split:] if (var_true is not None) else None
    et_tr = es_true[:split] if (es_true is not None) else None
    et_va = es_true[split:] if (es_true is not None) else None

    Xtr, Ytr, VTtr, ETtr, Ktr, _ = build_stateful_streams(
        x_tr, y_tr, batch_size, seq_len, vt_tr, et_tr
    )
    Xva, Yva, VTva, ETva, Kva, _ = build_stateful_streams(
        x_va, y_va, batch_size, seq_len, vt_va, et_va
    )

    best_val, best_state, bad = float("inf"), None, 0

    for ep in range(MAX_EPOCHS):
        model.train()
        h = None  # reset state at epoch start (stateful across chunks within epoch)
        tr_losses = []

        for k in range(Ktr):
            s, e = k * seq_len, (k + 1) * seq_len
            xb = torch.tensor(Xtr[:, s:e, :], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(Ytr[:, s:e], dtype=torch.float32, device=DEVICE)
            vt = (
                torch.tensor(VTtr[:, s:e], dtype=torch.float32, device=DEVICE)
                if VTtr is not None
                else None
            )
            et = (
                torch.tensor(ETtr[:, s:e], dtype=torch.float32, device=DEVICE)
                if ETtr is not None
                else None
            )

            opt.zero_grad()
            yhat, h = model(xb, h)  # [B,T,2] and carry state to next chunk
            loss = loss_fn(yb, yhat, vt, et)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            h = h.detach()
            tr_losses.append(float(loss.item()))

        # ----- validation -----
        model.eval()
        with torch.no_grad():
            if Kva == 0:
                val = float(np.mean(tr_losses)) if tr_losses else float("inf")
            else:
                h_val = None
                vloss = []
                for k in range(Kva):
                    s, e = k * seq_len, (k + 1) * seq_len
                    xb = torch.tensor(
                        Xva[:, s:e, :], dtype=torch.float32, device=DEVICE
                    )
                    yb = torch.tensor(Yva[:, s:e], dtype=torch.float32, device=DEVICE)
                    vt = (
                        torch.tensor(VTva[:, s:e], dtype=torch.float32, device=DEVICE)
                        if VTva is not None
                        else None
                    )
                    et = (
                        torch.tensor(ETva[:, s:e], dtype=torch.float32, device=DEVICE)
                        if ETva is not None
                        else None
                    )
                    yhat, h_val = model(xb, h_val)
                    vloss.append(float(loss_fn(yb, yhat, vt, et).item()))
                val = float(np.mean(vloss))

        if val < best_val:
            best_val, best_state, bad = (
                val,
                {k: v.cpu() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_stateful(
    model: SRNNVE3, x_all: np.ndarray, y_all: np.ndarray, split_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warm state on train, then one-step prediction on test (feed x[t] -> predict y[t])
    """
    model.eval()
    var_list, es_list, y_list = [], [], []
    with torch.no_grad():
        h = None
        # warm on train
        for t in range(split_idx):
            xb = torch.tensor(
                x_all[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
            )
            _, h = model(xb, h)
        # test loop
        for t in range(split_idx, len(x_all)):
            xb = torch.tensor(
                x_all[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
            )
            yhat, h = model(xb, h)  # [1,1,2]
            v, e = float(yhat[0, 0, 0].cpu()), float(yhat[0, 0, 1].cpu())
            var_list.append(v)
            es_list.append(e)
            y_list.append(float(y_all[t]))
    return np.array(var_list), np.array(es_list), np.array(y_list)


# ============================
# Pipeline
# ============================
def pipeline(
    csv_path="data/merged_data_with_realised_volatility.csv",
    alpha=ALPHA,
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
    df = build_inputs_from_prices(df)
    (x_tr, y_tr, vt_tr, et_tr), (x_te, y_te, vt_te, et_te), meta = (
        split_and_make_feature(df, train_frac=TRAIN_FRAC)
    )

    # ----- train (repo-style: stateful sequences, VE-3) -----
    model = train_srnn_stateful_sequences(
        x_tr,
        y_tr,
        vt_tr,
        et_tr,
        alpha=alpha,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        dropout_p=DROPOUT_P,
    )

    # ----- evaluate -----
    split_idx = len(x_tr)
    x_all = np.concatenate([x_tr, x_te])
    y_all = np.concatenate([y_tr, y_te])
    v_pred, e_pred, y_test = evaluate_stateful(model, x_all, y_all, split_idx)

    # ----- optional exact calibration on TRAIN only -----
    if calibrate:
        vs_tr, es_tr, ys_tr = [], [], []
        model.eval()
        with torch.no_grad():
            h = None
            for t in range(len(x_tr)):
                xb = torch.tensor(
                    x_tr[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
                )
                yhat, h = model(xb, h)
                vs_tr.append(float(yhat[0, 0, 0].cpu()))
                es_tr.append(float(yhat[0, 0, 1].cpu()))
                ys_tr.append(float(y_tr[t]))
        vs_tr = np.array(vs_tr)
        es_tr = np.array(es_tr)
        ys_tr = np.array(ys_tr)

        c_v = exact_var_factor(ys_tr, vs_tr, alpha)
        c_e = exact_es_factor(ys_tr, vs_tr * c_v, es_tr, alpha)
        v_eval = v_pred * c_v
        e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)
        base = f"srnn_repo_{(run_tag + '_') if run_tag else ''}calibrated".replace(
            " ", ""
        )
        title = "SRNN-VE-3 (repo-style, calibrated)"
    else:
        v_eval, e_eval = v_pred, e_pred
        base = f"srnn_repo_{(run_tag + '_') if run_tag else ''}raw".replace(" ", "")
        title = "SRNN-VE-3 (repo-style, raw)"

    print("std(v_pred), std(e_pred) before calib:", np.std(v_pred), np.std(e_pred))
    print("std(v_eval), std(e_eval)  after  calib:", np.std(v_eval), np.std(e_eval))

    # ----- metrics -----
    hits = (y_test <= v_eval).astype(int)
    LR_pof, p_pof, _, _ = kupiec_pof(hits, alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    LR_cc, p_cc = christoffersen_cc(hits, alpha)
    fz0 = fz0_per_step(y_test, v_eval, e_eval, alpha)

    print("=" * 60)
    print(title + (f"  [{run_tag}]" if run_tag else ""))
    print("=" * 60)
    print(f"Hit rate: {hits.mean():.4f} (Target {alpha:.4f})")
    print(f"Kupiec: LR={LR_pof:.4f}, p={p_pof:.4f}")
    print(f"Christoffersen IND: LR={LR_ind:.4f}, p={p_ind:.4f}")
    print(f"Christoffersen CC : LR={LR_cc:.4f}, p={p_cc:.4f}")
    print(f"Avg FZ0: {fz0.mean():.6f}")

    # Save arrays/metrics
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        y=y_test,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=hits,
    )

    plot_var_es_diagnostics(
        y_true=y_test,
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

    return model, metrics, (v_eval, e_eval, y_test, fz0)


if __name__ == "__main__":
    pipeline(calibrate=True)
    # calibrate = True
