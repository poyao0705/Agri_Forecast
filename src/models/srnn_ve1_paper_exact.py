import os
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler

from src.utils.eval_tools import (
    fz0_per_step,
    exact_var_factor,
    exact_es_factor,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    plot_var_es_diagnostics,
)

# ============================
# Config (shared across models)
# ============================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ALPHA = 0.01
TRAIN_FRAC = 0.5
MAX_EPOCHS = 200
PATIENCE = 30

# Optimizer settings
LR_BASE = 1e-3
LR_HEAD = 2e-3
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 0.5

# LR scheduler
USE_SCHED = True
SCHED_STEP = 20
SCHED_GAMMA = 0.9

# Truncated BPTT unroll length
UNROLL = 64

DROPOUT_P = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Easy calibration defaults ---
CAL_LATE_FRAC = 0.70  # calibrate on the last 30% of train
CAL_TARGET_HITS = 12  # aim for ~12 tail hits in cal set
CAL_MIN_HITS = 6  # below this, shrink factors toward 1
CAL_MAX_STRIDE = 32  # don't stride sparser than this
CAL_FACTOR_CLAMP = (0.7, 1.5)  # soft safety bounds on factors


def _choose_cal_stride(
    n_train, seq_len, alpha, target_hits=CAL_TARGET_HITS, max_stride=CAL_MAX_STRIDE
):
    approx_windows = max(n_train - seq_len - 1, 1)  # stride-1 windows
    target_windows = max(int(target_hits / alpha), 1)
    stride = max(1, min(max_stride, approx_windows // target_windows))
    return stride


# ============================
# Shared utilities (identical to transformer file)
# ============================
def build_inputs_from_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()

    # contemporaneous features (allowed when predicting next step)
    df["r2"] = df["log_ret"] ** 2
    df["neg_ret"] = (df["log_ret"] < 0).astype(np.float32)

    # volatility proxies using past-only information
    df["ewma94_var"] = df["r2"].ewm(alpha=1 - 0.94, adjust=False).mean().shift(1)
    df["ewma97_var"] = df["r2"].ewm(alpha=1 - 0.97, adjust=False).mean().shift(1)
    df["ewma94"] = np.sqrt(df["ewma94_var"])
    df["ewma97"] = np.sqrt(df["ewma97_var"])

    # target = next-step return
    df["target_return"] = df["log_ret"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


def _make_features(df: pd.DataFrame, feature_parity: bool) -> List[str]:
    return (
        ["x_cov"]
        if feature_parity
        else ["ewma94", "ewma97", "x_cov", "neg_xcov", "neg_ret"]
    )


def split_and_make_features(
    df: pd.DataFrame, feature_parity: bool, train_frac: float = TRAIN_FRAC
):
    n = len(df)
    split = int(train_frac * n)
    mu_train = df.loc[: split - 1, "log_ret"].mean()

    df = df.copy()
    df["x_cov"] = (df["log_ret"] - mu_train) ** 2
    df["neg_xcov"] = df["neg_ret"] * df["x_cov"]

    feature_cols = _make_features(df, feature_parity)
    X = df[feature_cols].values.astype(np.float32)
    y = df["target_return"].values.astype(np.float32)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    meta = {
        "split_idx": split,
        "mu_train": float(mu_train),
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return X_train, y_train, X_test, y_test, meta


def easy_calibrate(model, Xtr, ytr, alpha, seq_len):
    start = int(CAL_LATE_FRAC * len(Xtr))
    Xc = Xtr[start:]
    yc = ytr[start:]

    # choose a stride; for stateful forecasts seq_len is irrelevant → pass 1
    stride = _choose_cal_stride(len(Xc), 1, alpha)

    vs, es, ys = [], [], []
    model.eval()
    with torch.no_grad():
        # ---- warm the hidden state on the pre-calibration portion ----
        h = None
        for t in range(start):
            xb = torch.tensor(
                Xtr[t : t + 1].reshape(1, 1, -1), dtype=torch.float32, device=DEVICE
            )
            _, h = model(xb, h)

        # ---- generate stateful predictions over the late-train calibration slice ----
        for t in range(len(Xc)):
            xb = torch.tensor(
                Xc[t : t + 1].reshape(1, 1, -1), dtype=torch.float32, device=DEVICE
            )
            yb = float(yc[t])
            yhat, h = model(xb, h)
            vs.append(float(yhat[0, 0].cpu()))
            es.append(float(yhat[0, 1].cpu()))
            ys.append(yb)

    vs = np.asarray(vs)[::stride]
    es = np.asarray(es)[::stride]
    ys = np.asarray(ys)[::stride]

    if len(vs) == 0:
        print("[cal] no samples → identity factors")
        return 1.0, 1.0

    c_v = exact_var_factor(ys, vs, alpha)
    c_e = exact_es_factor(ys, vs * c_v, es, alpha)

    hits = int((ys <= vs * c_v).sum())
    N = len(ys)
    if hits < CAL_MIN_HITS:
        lam = hits / max(CAL_MIN_HITS, 1)
        c_v = 1.0 + lam * (c_v - 1.0)
        c_e = 1.0 + lam * (c_e - 1.0)

    c_v = float(np.clip(c_v, *CAL_FACTOR_CLAMP))
    c_e = float(np.clip(c_e, *CAL_FACTOR_CLAMP))

    print(
        f"[cal] stride={stride}  N={N}  hits={hits}/{N} ({hits/N:.4f})  c_v={c_v:.4f}  c_e={c_e:.4f}"
    )
    mask = ys <= vs * c_v
    if mask.any():
        es_real = float(ys[mask].mean())
        es_pred = float((es * c_e)[mask].mean())
        print(f"[cal] ES(real)={es_real:.5f}  ES(pred)={es_pred:.5f}")
    return c_v, c_e


# ============================
# Loss function
# ============================
class FZ0Loss(nn.Module):
    def __init__(self, alpha=ALPHA, eps=1e-6):
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
# SRNN-VE-1 model
# ============================
class LinearRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h0=None):
        B, T, _ = x.shape
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
        out = torch.cat(outs, dim=1)
        return out, h


class SRNNVE1(nn.Module):
    def __init__(self, input_dim: int, dropout_p: float = DROPOUT_P):
        super().__init__()
        hidden_size = 1
        self.rnn = LinearRNN(
            input_size=input_dim, hidden_size=hidden_size, dropout_p=dropout_p
        )
        self.head = nn.Linear(hidden_size, 2)

        def softplus_inv(x):
            import math

            return math.log(math.exp(x) - 1.0)

        with torch.no_grad():
            nn.init.normal_(self.head.weight, 0.0, 0.01)
            self.head.bias[:] = torch.tensor(
                [softplus_inv(0.02), softplus_inv(0.01)], dtype=torch.float32
            )

    def forward(self, x, h_prev=None):
        rnn_out, h_n = self.rnn(x, h_prev)
        h_t = rnn_out[:, -1, :]
        a, b = self.head(h_t).unbind(dim=1)
        v = -F.softplus(a)
        e = v - F.softplus(b)
        y_pred = torch.stack([v, e], dim=1)
        return y_pred, h_n


def train_srnn_stateful(
    x_train, y_train, input_dim: int, alpha=ALPHA, dropout_p=DROPOUT_P
) -> SRNNVE1:
    model = SRNNVE1(input_dim=input_dim, dropout_p=dropout_p).to(DEVICE)
    loss_fn = FZ0Loss(alpha=alpha)

    # internal split
    best_val, best_state, bad = float("inf"), None, 0
    split = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:split], y_train[:split]
    x_va, y_va = x_train[split:], y_train[split:]

    base_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if n.startswith("head.") else base_params).append(p)

    opt = torch.optim.AdamW(
        [
            {
                "params": base_params,
                "lr": LR_BASE,
                "weight_decay": 0.0,
            },  # no L2 on core
            {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        ]
    )
    sched = StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA) if USE_SCHED else None

    for ep in range(MAX_EPOCHS):
        model.train()
        h = None
        opt.zero_grad()

        loss_accum = None
        steps_in_unroll = 0

        for t in range(len(x_tr)):
            xb = torch.tensor(
                x_tr[t : t + 1].reshape(1, 1, -1), dtype=torch.float32, device=DEVICE
            )
            yb = torch.tensor([y_tr[t]], dtype=torch.float32, device=DEVICE)

            yhat, h = model(xb, h)  # do not detach here
            loss_t = loss_fn(yb, yhat)

            loss_accum = loss_t if loss_accum is None else (loss_accum + loss_t)
            steps_in_unroll += 1

            if steps_in_unroll == UNROLL:
                (loss_accum / UNROLL).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                opt.step()
                opt.zero_grad()
                h = h.detach()
                loss_accum = None
                steps_in_unroll = 0

        if steps_in_unroll > 0:
            (loss_accum / steps_in_unroll).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            opt.zero_grad()
            h = None

        # validation
        model.eval()
        with torch.no_grad():
            h_val, preds_v, ys_v = None, [], []
            for t in range(len(x_va)):
                xb = torch.tensor(
                    x_va[t : t + 1].reshape(1, 1, -1),
                    dtype=torch.float32,
                    device=DEVICE,
                )
                yb = y_va[t]
                yhat, h_val = model(xb, h_val)
                preds_v.append(yhat.squeeze(0).cpu())
                ys_v.append(yb)

            val = float("inf")
            if preds_v:
                preds_v = torch.stack(preds_v, dim=0)
                ys_v = torch.tensor(ys_v, dtype=torch.float32)
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
                if sched is not None:
                    sched.step()
                break
        if sched is not None:
            sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_stateful(model: SRNNVE1, x_all, y_all, split_idx: int):
    model.eval()
    var_list, es_list, y_list = [], [], []
    with torch.no_grad():
        # warm on training
        h = None
        for t in range(split_idx):
            xb = torch.tensor(
                x_all[t : t + 1].reshape(1, 1, -1), dtype=torch.float32, device=DEVICE
            )
            _, h = model(xb, h)

        for t in range(split_idx, len(x_all)):
            xb = torch.tensor(
                x_all[t : t + 1].reshape(1, 1, -1), dtype=torch.float32, device=DEVICE
            )
            yhat, h = model(xb, h)
            var_list.append(float(yhat[0, 0].cpu()))
            es_list.append(float(yhat[0, 1].cpu()))
            y_list.append(float(y_all[t]))
    return np.array(var_list), np.array(es_list), np.array(y_list)


# ============================
# Pipeline
# ============================
def pipeline(
    csv_path="data/merged_data_with_realised_volatility.csv",
    alpha=ALPHA,
    calibrate=False,
    feature_parity=True,
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
    X_tr, y_tr, X_te, y_te, meta = split_and_make_features(
        df, feature_parity=feature_parity, train_frac=TRAIN_FRAC
    )
    feat_label = "parity" if feature_parity else "full"
    input_dim = X_tr.shape[1]

    model = train_srnn_stateful(
        X_tr, y_tr, input_dim=input_dim, alpha=alpha, dropout_p=DROPOUT_P
    )

    split_idx = len(X_tr)
    X_all = np.concatenate([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])

    v_pred, e_pred, y_aligned = evaluate_stateful(model, X_all, y_all, split_idx)

    print("CHECK v<0, e<v:", np.mean(v_pred < 0), np.mean(e_pred < v_pred))
    print("Var/ES mean±std:", v_pred.mean(), v_pred.std(), e_pred.mean(), e_pred.std())
    print(
        "Head W,b (a,b):",
        float(model.head.weight[0, 0].cpu()),
        float(model.head.bias[0].cpu()),
        float(model.head.weight[1, 0].cpu()),
        float(model.head.bias[1].cpu()),
    )

    if calibrate:
        # easy, late-train, stateful calibration
        c_v, c_e = easy_calibrate(model, X_tr, y_tr, alpha=alpha, seq_len=UNROLL)
        v_eval = v_pred * c_v
        e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)
        # drift diagnostics (guard against very short tests)
        n_test = len(y_aligned)
        if n_test >= 2:
            mid = n_test // 2
            print(
                "Test hit rate (first half):", np.mean(y_aligned[:mid] <= v_eval[:mid])
            )
            print(
                "Test hit rate (second half):", np.mean(y_aligned[mid:] <= v_eval[mid:])
            )

        k1 = n_test // 3
        k2 = 2 * n_test // 3
        print(
            "Hit rate terciles:",
            np.mean(y_aligned[:k1] <= v_eval[:k1]),
            np.mean(y_aligned[k1:k2] <= v_eval[k1:k2]),
            np.mean(y_aligned[k2:] <= v_eval[k2:]),
        )
    else:
        v_eval, e_eval = v_pred, e_pred
        c_v, c_e = 1.0, 1.0

    # sanity
    print(
        "VaR summary:",
        float(np.min(v_eval)),
        float(np.median(v_eval)),
        float(np.max(v_eval)),
    )
    print("Is VaR always negative?", bool(np.all(v_eval < 0)))
    print("ES < VaR everywhere?", bool(np.all(e_eval < v_eval)))
    print(
        "Return quantiles:",
        np.quantile(y_aligned, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]),
    )
    print("Hit rate:", float((y_aligned <= v_eval).mean()))

    hits = (y_aligned <= v_eval).astype(int)
    LR_pof, p_pof, _, _ = kupiec_pof(hits, alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    LR_cc, p_cc = christoffersen_cc(hits, alpha)
    fz0 = fz0_per_step(y_aligned, v_eval, e_eval, alpha)

    title = f"SRNN-VE-1 ({feat_label}, {'calibrated' if calibrate else 'raw'})"
    print("=" * 60)
    print(title + (f"  [{run_tag}]" if run_tag else ""))
    print("=" * 60)
    print(f"Hit rate: {hits.mean():.4f} (Target {alpha:.4f})")
    print(f"Kupiec: LR={LR_pof:.4f}, p={p_pof:.4f}")
    print(f"Christoffersen IND: LR={LR_ind:.4f}, p={p_ind:.4f}")
    print(f"Christoffersen CC : LR={LR_cc:.4f}, p={p_cc:.4f}")
    print(f"Avg FZ0: {fz0.mean():.6f}")

    base = f"srnn_{(run_tag + '_') if run_tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
        " ", ""
    )
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=hits,
        features=list(meta["feature_cols"]),
        feature_parity=bool(feature_parity),
        c_v=float(c_v),
        c_e=float(c_e),
    )
    with open(os.path.join(out_dir, f"{base}.json"), "w") as f:
        json.dump(
            dict(
                model=title,
                model_desc=title,  # Add model_desc field
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
                feature_parity=bool(feature_parity),
                features=list(meta["feature_cols"]),
                n=int(len(y_aligned)),
                calibrated=bool(calibrate),
                c_v=float(c_v),
                c_e=float(c_e),
            ),
            f,
            indent=2,
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

    return (
        model,
        dict(
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
            feature_parity=bool(feature_parity),
            features=list(meta["feature_cols"]),
            n=int(len(y_aligned)),
        ),
        (v_eval, e_eval, y_aligned, fz0),
    )


if __name__ == "__main__":
    pipeline()
