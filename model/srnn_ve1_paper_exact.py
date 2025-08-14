# srnn_ve1_paper_exact.py
# ------------------------------------------------------------
# SRNN-VE-1 (paper-exact):
#  • Architecture: 1-unit linear RNN + coherent head:
#       v = -softplus(a), e = v - softplus(b)  ⇒  e < v < 0
#  • Feature: volatility proxy x_t = (r_t - mean_train)^2  (standardized on train; pure reparam)
#  • Loss: FZ0 at level α  (matches eval_tools: ind-only + v/e + log(-e))
#  • Training: stateful, single-step; NO TBPTT
#    ↳ but we use gradient accumulation across K steps before optimizer.step()
#  • Optional exact calibration on train
# ------------------------------------------------------------

import os
import json
from typing import Tuple

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

MAX_EPOCHS = 200
PATIENCE = 30

# Optimizer settings (stable for this setup)
LR_BASE = 1e-3
LR_HEAD = 2e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

# Accumulate per-step losses before stepping the optimizer (no TBPTT)
ACCUM_STEPS = 256  # ~ a few months of days gives ~2–3 hits at α=1%

DROPOUT_P = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# Model
# ============================
class LinearRNN(nn.Module):
    """
    Linear recurrence with variational (time-constant) input dropout.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        # Unconstrained weights (paper-exact), small init
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h0=None):
        """
        x: [B, T, input_size]
        returns: out: [B, T, hidden], h_last: [B, hidden]
        """
        B, T, _ = x.shape

        if self.training and self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            mask = x.new_empty(B, 1, self.input_size).bernoulli_(keep) / keep
            x = x * mask  # broadcast across T

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h = h0 if h0.dim() == 2 else h0.squeeze(0)

        outs = []
        for t in range(T):
            xt = x[:, t, :]
            h = h @ self.W_hh.T + xt @ self.W_xh.T + self.b_h
            outs.append(h.unsqueeze(1))
        out = torch.cat(outs, dim=1)
        return out, h


class SRNNVE1(nn.Module):
    """
    Paper-exact SRNN-VE-1 head: coherent (e < v < 0).
    hidden_size = 1 (per paper).
    """

    def __init__(self, dropout_p: float = DROPOUT_P):
        super().__init__()
        hidden_size = 1  # paper-exact
        self.rnn = LinearRNN(input_size=1, hidden_size=hidden_size, dropout_p=dropout_p)
        self.head = nn.Linear(hidden_size, 2)  # logits -> (a, b)

        # Bias init so v≈-0.02 and e≈v-0.01 at start (neutral-ish)
        def softplus_inv(x):  # x > 0
            import math

            return math.log(math.exp(x) - 1.0)

        with torch.no_grad():
            nn.init.normal_(self.head.weight, 0.0, 0.01)  # non-zero slope at start
            self.head.bias[:] = torch.tensor(
                [softplus_inv(0.02), softplus_inv(0.01)], dtype=torch.float32
            )

    def forward(self, x, h_prev=None):
        """
        x: [B, T, 1]. We will use T=1 during training/eval (stateful).
        returns:
            y_pred_last: [B, 2] for the last timestep in x
            h_last     : [B, 1]
        """
        rnn_out, h_n = self.rnn(x, h_prev)  # [B, T, 1], [B,1]
        h_t = rnn_out[:, -1, :]  # [B, 1]
        a, b = self.head(h_t).unbind(dim=1)  # each [B]
        v = -F.softplus(a)  # v < 0
        e = v - F.softplus(b)  # e < v
        y_pred = torch.stack([v, e], dim=1)  # [B, 2]
        return y_pred, h_n


class FZ0Loss(nn.Module):
    """
    FZ0 at level alpha, matching eval_tools:
        L = -(I{y<=v} * (v - y)) / (α e) + (v / e) + log(-e)
    A tiny clamp keeps e<0 away from 0 **inside the loss only** for stability.
    """

    def __init__(self, alpha=ALPHA, eps=1e-4):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_true, y_pred):
        v, e = y_pred[:, 0], y_pred[:, 1]
        e_safe = torch.clamp(e, max=-self.eps)
        ind = (y_true <= v).float()
        term1 = -(ind * (v - y_true)) / (self.alpha * e_safe)
        term2 = (v / e_safe) + torch.log(-e_safe)
        return (term1 + term2).mean()


# ============================
# Data utils
# ============================
def build_inputs_from_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    df["target_return"] = df["log_ret"].shift(-1)  # next-day return
    df = df.dropna().reset_index(drop=True)
    return df


def split_and_make_feature(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    n = len(df)
    split = int(train_frac * n)
    mu_train = df.loc[: split - 1, "log_ret"].mean()

    # Paper feature: volatility proxy (standardize on TRAIN only; linear reparam of W_xh)
    df["x_cov"] = (df["log_ret"] - mu_train) ** 2

    x = df["x_cov"].values.reshape(-1, 1).astype(np.float32)
    y = df["target_return"].values.astype(np.float32)

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    m, s = x_train.mean(), x_train.std() + 1e-12
    x_train = (x_train - m) / s
    x_test = (x_test - m) / s

    meta = {
        "split_idx": split,
        "mu_train": float(mu_train),
        "x_mean": float(m),
        "x_std": float(s),
    }
    return x_train, y_train, x_test, y_test, meta


# ============================
# Train / Eval (stateful 1-step)
# ============================
def _make_param_groups(model: nn.Module):
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if n.startswith("head.") else base_params).append(p)
    return base_params, head_params


def train_srnn_stateful(x_train, y_train, alpha=ALPHA, dropout_p=DROPOUT_P) -> SRNNVE1:
    model = SRNNVE1(dropout_p=dropout_p).to(DEVICE)
    loss_fn = FZ0Loss(alpha=alpha)

    # Train/val split inside train
    best_val, best_state, bad = float("inf"), None, 0
    split = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:split], y_train[:split]
    x_va, y_va = x_train[split:], y_train[split:]

    base_params, head_params = _make_param_groups(model)
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": LR_BASE, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": LR_HEAD, "weight_decay": 0.0},
        ]
    )

    for ep in range(MAX_EPOCHS):
        model.train()
        h = None
        accum = 0
        opt.zero_grad()

        # Online, stateful; accumulate gradients for stability
        for t in range(len(x_tr)):
            xb = torch.tensor(
                x_tr[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
            )
            yb = torch.tensor([y_tr[t]], dtype=torch.float32, device=DEVICE)

            yhat, h = model(xb, h)  # [1,2], stateful
            loss = loss_fn(yb, yhat) / ACCUM_STEPS
            loss.backward()
            h = h.detach()
            accum += 1

            if accum % ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                opt.step()
                opt.zero_grad()

        # flush any remainder
        if accum % ACCUM_STEPS != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            opt.zero_grad()

        # ---- validation: stateful along validation timeline (fresh state) ----
        model.eval()
        with torch.no_grad():
            h_val, preds_v, ys_v = None, [], []
            for t in range(len(x_va)):
                xb = torch.tensor(
                    x_va[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
                )
                yb = y_va[t]
                yhat, h_val = model(xb, h_val)
                preds_v.append(yhat.squeeze(0).cpu())
                ys_v.append(yb)

            val = float("inf")
            if preds_v:
                preds_v = torch.stack(preds_v, dim=0)  # [T_va, 2]
                ys_v = torch.tensor(ys_v, dtype=torch.float32)  # [T_va]
                val = loss_fn(ys_v, preds_v).item()

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
    model: SRNNVE1, x_all, y_all, split_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll state through training inputs [0 .. split_idx-1].
    For each test day t >= split_idx: feed x[t], predict y[t].
    Returns arrays aligned to y_test indices.
    """
    model.eval()
    var_list, es_list, y_list = [], [], []

    with torch.no_grad():
        # Warm state on the entire training span
        h = None
        for t in range(split_idx):
            xb = torch.tensor(
                x_all[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
            )
            _, h = model(xb, h)

        # Test loop: x[t] -> y[t]
        for t in range(split_idx, len(x_all)):
            xb = torch.tensor(
                x_all[t : t + 1].reshape(1, 1, 1), dtype=torch.float32, device=DEVICE
            )
            yhat, h = model(xb, h)
            v = float(yhat[0, 0].cpu())
            e = float(yhat[0, 1].cpu())
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

    # ----- data -----
    df = pd.read_csv(csv_path)
    df = build_inputs_from_prices(df)
    x_tr, y_tr, x_te, y_te, meta = split_and_make_feature(df, train_frac=TRAIN_FRAC)

    # ----- train (paper-exact) -----
    model = train_srnn_stateful(x_tr, y_tr, alpha=alpha, dropout_p=DROPOUT_P)

    # ----- evaluate -----
    split_idx = len(x_tr)
    x_all = np.concatenate([x_tr, x_te])
    y_all = np.concatenate([y_tr, y_te])

    v_pred, e_pred, y_test = evaluate_stateful(model, x_all, y_all, split_idx)

    print("CHECK v<0, e<v:", np.mean(v_pred < 0), np.mean(e_pred < v_pred))
    print("Var/ES mean±std:", v_pred.mean(), v_pred.std(), e_pred.mean(), e_pred.std())
    print(
        "Head W,b (a,b):",
        float(model.head.weight[0, 0].cpu()),
        float(model.head.bias[0].cpu()),
        float(model.head.weight[1, 0].cpu()),
        float(model.head.bias[1].cpu()),
    )

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
                yb = y_tr[t]
                yhat, h = model(xb, h)
                vs_tr.append(float(yhat[0, 0].cpu()))
                es_tr.append(float(yhat[0, 1].cpu()))
                ys_tr.append(float(yb))

        vs_tr = np.array(vs_tr)
        es_tr = np.array(es_tr)
        ys_tr = np.array(ys_tr)

        c_v = exact_var_factor(ys_tr, vs_tr, alpha)
        c_e = exact_es_factor(ys_tr, vs_tr * c_v, es_tr, alpha)

        v_eval = v_pred * c_v
        e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)
        base = f"srnn_{(run_tag + '_') if run_tag else ''}calibrated".replace(" ", "")
        title = "SRNN-VE-1 (paper-exact, calibrated)"
    else:
        v_eval, e_eval = v_pred, e_pred
        base = f"srnn_{(run_tag + '_') if run_tag else ''}raw".replace(" ", "")
        title = "SRNN-VE-1 (paper-exact, raw)"

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

    # ----- persist -----
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
    pipeline()
