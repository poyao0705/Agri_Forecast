#!/usr/bin/env python3
"""
Transformer Model for VaR/ES Prediction

A comprehensive implementation of a transformer-based model for predicting
Value-at-Risk (VaR) and Expected Shortfall (ES) from financial time series data.

This module provides:
- Transformer architecture for sequence modeling
- Feature engineering from price data
- Training pipeline with sliding windows
- Evaluation with statistical backtesting
- Calibration utilities for improved accuracy

Key Features:
- Multi-head attention mechanism for capturing temporal dependencies
- Feature parity mode for fair comparison studies
- Online calibration for improved prediction accuracy
- Comprehensive backtesting with statistical tests
- Model persistence and reuse capabilities

Usage:
    from transformer_var_es_paper_exact import pipeline

    model, metrics, (v_eval, e_eval, y_aligned, fz0) = pipeline(
        csv_path="data.csv",
        alpha=0.01,
        feature_parity=True,
        calibrate=True
    )
"""

import os
import json
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler

from utils.eval_tools import (
    fz0_per_step,
    exact_var_factor,
    exact_es_factor,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    plot_var_es_diagnostics,
    print_online_drift,
    _choose_window_for_alpha,
)

# ============================
# Configuration Parameters
# ============================

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model hyperparameters
ALPHA = 0.01  # VaR/ES confidence level (1%)
TRAIN_FRAC = 0.5  # Training data fraction
CONTEXT_LEN = 64  # Sequence length for transformer
BATCH_SIZE = 64  # Training batch size
MAX_EPOCHS = 200  # Maximum training epochs
PATIENCE = 30  # Early stopping patience
LR = 1e-4  # Learning rate
WEIGHT_DECAY = 1e-3  # Weight decay for regularization

# Sliding window parameters
TRAIN_STRIDE = 1  # Stride for training windows
CALIB_STRIDE = 16  # Stride for calibration (reduces correlation)
VAL_GAP_WINDOWS = CONTEXT_LEN  # Gap between train and validation

# Calibration parameters
CAL_LATE_FRAC = 0.70  # Use last 30% of training data for calibration
CAL_TARGET_HITS = 12  # Target number of tail hits in calibration set
CAL_MIN_HITS = 6  # Minimum hits before applying calibration
CAL_MAX_STRIDE = 32  # Maximum stride for calibration windows
CAL_FACTOR_CLAMP = (0.7, 1.5)  # Safety bounds for calibration factors

# Fallback values for calibration
CAL_MIN_HITS = globals().get("CAL_MIN_HITS", 10)
CAL_FACTOR_CLAMP = globals().get("CAL_FACTOR_CLAMP", (0.25, 4.0))


def _choose_cal_stride(
    n_train: int,
    seq_len: int,
    alpha: float,
    target_hits: int = CAL_TARGET_HITS,
    max_stride: int = CAL_MAX_STRIDE,
) -> int:
    """
    Choose optimal stride for calibration windows.

    Determines the stride that will give approximately the target number
    of tail hits in the calibration set while respecting maximum stride limits.

    Args:
        n_train (int): Number of training observations
        seq_len (int): Sequence length for transformer
        alpha (float): VaR/ES confidence level
        target_hits (int): Target number of tail hits in calibration set
        max_stride (int): Maximum allowed stride

    Returns:
        int: Optimal stride for calibration windows
    """
    approx_windows = max(n_train - seq_len - 1, 1)  # stride-1 windows
    target_windows = max(int(target_hits / alpha), 1)
    stride = max(1, min(max_stride, approx_windows // target_windows))
    return stride


# ============================
# Data Processing and Feature Engineering
# ============================


def build_inputs_from_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from price data for VaR/ES prediction.

    Creates engineered features from closing prices including:
    - Log returns and squared returns
    - Negative return indicators
    - Exponentially weighted moving average volatility proxies
    - Target variable (next-step return)

    Args:
        df (pd.DataFrame): DataFrame with 'close' column

    Returns:
        pd.DataFrame: DataFrame with engineered features

    Note:
        Features are designed to be causal (no look-ahead bias)
        except for contemporaneous features which are allowed
        when predicting the next step.
    """
    df = df.copy()

    # Calculate log returns
    df["log_ret"] = np.log(df["close"]).diff()

    # Contemporaneous features (allowed when predicting next step)
    df["r2"] = df["log_ret"] ** 2
    df["neg_ret"] = (df["log_ret"] < 0).astype(np.float32)

    # Volatility proxies using past-only information
    df["ewma94_var"] = df["r2"].ewm(alpha=1 - 0.94, adjust=False).mean().shift(1)
    df["ewma97_var"] = df["r2"].ewm(alpha=1 - 0.97, adjust=False).mean().shift(1)
    df["ewma94"] = np.sqrt(df["ewma94_var"])
    df["ewma97"] = np.sqrt(df["ewma97_var"])

    # Target variable: next-step return
    df["target_return"] = df["log_ret"].shift(-1)

    # Remove NaN values and reset index
    df = df.dropna().reset_index(drop=True)
    return df


def _make_features(df: pd.DataFrame, feature_parity: bool) -> List[str]:
    """
    Select feature columns based on feature parity setting.

    Args:
        df (pd.DataFrame): DataFrame with engineered features
        feature_parity (bool): If True, use only x_cov feature for fair comparison

    Returns:
        List[str]: List of feature column names to use

    Note:
        Feature parity mode uses only the x_cov feature to ensure
        fair comparison with other models that may have limited
        feature access.
    """
    if feature_parity:
        return ["x_cov"]
    else:
        return ["ewma94", "ewma97", "x_cov", "neg_xcov", "neg_ret"]


def split_and_make_features(
    df: pd.DataFrame, feature_parity: bool, train_frac: float = TRAIN_FRAC
):
    n = len(df)
    split = int(train_frac * n)
    mu_train = df.loc[: split - 1, "log_ret"].mean()

    df = df.copy()
    df["x_cov"] = (df["log_ret"] - mu_train) ** 2  # paper-style volatility proxy
    df["neg_xcov"] = df["neg_ret"] * df["x_cov"]

    feature_cols = _make_features(df, feature_parity)
    X = df[feature_cols].values.astype(np.float32)
    y = df["target_return"].values.astype(np.float32)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)

    # Handle case where X_test is empty (when train_frac=1.0)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test).astype(np.float32)
    else:
        X_test = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_test = np.empty(0, dtype=np.float32)

    meta = {
        "split_idx": split,
        "mu_train": float(mu_train),
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return X_train, y_train, X_test, y_test, meta


def _ensure_2d_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


def _normalize_unfold_shape(
    xwin: torch.Tensor, seq_len: int, feat_dim: int
) -> torch.Tensor:
    if xwin.dim() != 3:
        raise RuntimeError(f"Expected 3D windows, got {tuple(xwin.shape)}")
    if xwin.size(1) == seq_len and xwin.size(2) == feat_dim:
        return xwin.contiguous()
    if xwin.size(1) == feat_dim and xwin.size(2) == seq_len:
        return xwin.permute(0, 2, 1).contiguous()
    raise RuntimeError(f"Unrecognized window shape {tuple(xwin.shape)}")


def make_windows_with_stride(
    X: np.ndarray, y: np.ndarray, seq_len: int = CONTEXT_LEN, stride: int = TRAIN_STRIDE
):
    X2 = _ensure_2d_features(X)
    xt = torch.as_tensor(X2, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    Fdim = xt.size(1)

    if xt.size(0) <= seq_len:
        return torch.empty(0, seq_len, Fdim), torch.empty(0)

    X_win = xt.unfold(dimension=0, size=seq_len, step=stride)
    # X_win = X_win[:-1]
    X_win = _normalize_unfold_shape(X_win, seq_len, Fdim)

    num = X_win.size(0)
    # y_idx = torch.arange(seq_len, seq_len + num * stride, step=stride, device=yt.device)
    # align to window end t (since y[t] is already r_{t+1})
    y_idx = torch.arange(
        seq_len - 1, seq_len - 1 + num * stride, step=stride, device=yt.device
    )
    # --- Alignment guards (dev only) ---
    assert int(y_idx[0]) == seq_len - 1, "First label must align to window end"
    assert int(y_idx[-1]) == (seq_len - 1) + (num - 1) * stride, "Last label misaligned"
    y_next = yt.index_select(0, y_idx).contiguous()
    return X_win.contiguous(), y_next


def easy_calibrate(model, Xtr, ytr, alpha, seq_len):
    start = int(CAL_LATE_FRAC * len(Xtr))
    Xc = Xtr[start:]
    yc = ytr[start:]
    stride = _choose_cal_stride(len(Xc), seq_len, alpha)

    X_cal_t, y_cal_t = make_windows_with_stride(Xc, yc, seq_len=seq_len, stride=stride)
    if len(X_cal_t) == 0:
        print("[cal] no windows → using identity factors")
        return 1.0, 1.0

    with torch.no_grad():
        cal_pred = model(X_cal_t).cpu().numpy()
    v_tr, e_tr = cal_pred[:, 0], cal_pred[:, 1]
    y_cal = y_cal_t.cpu().numpy()

    c_v = exact_var_factor(y_cal, v_tr, alpha)
    c_e = exact_es_factor(y_cal, v_tr * c_v, e_tr, alpha)

    hits = int((y_cal <= v_tr * c_v).sum())
    N = len(y_cal)
    if hits < CAL_MIN_HITS:
        lam = hits / max(CAL_MIN_HITS, 1)
        c_v = 1.0 + lam * (c_v - 1.0)
        c_e = 1.0 + lam * (c_e - 1.0)

    c_v = float(np.clip(c_v, *CAL_FACTOR_CLAMP))
    c_e = float(np.clip(c_e, *CAL_FACTOR_CLAMP))

    print(
        f"[cal] stride={stride}  N={N}  hits={hits}/{N} ({hits/N:.4f})  c_v={c_v:.4f}  c_e={c_e:.4f}"
    )
    mask = y_cal <= v_tr * c_v
    if mask.any():
        es_real = float(y_cal[mask].mean())
        es_pred = float((e_tr * c_e)[mask].mean())
        print(f"[cal] ES(real)={es_real:.5f}  ES(pred)={es_pred:.5f}")
    return c_v, c_e


def _compute_factor_pair(y_hist, v_hist, e_hist, alpha):
    """
    Compute (c_v, c_e) on a history slice; shrink toward 1 if too few hits;
    and clamp to a safe range. Uses your existing exact_* functions.
    """
    c_v = exact_var_factor(y_hist, v_hist, alpha)
    c_e = exact_es_factor(y_hist, v_hist * c_v, e_hist, alpha)

    hits = int((y_hist <= v_hist * c_v).sum())
    if hits < CAL_MIN_HITS:
        lam = hits / max(CAL_MIN_HITS, 1)
        c_v = 1.0 + lam * (c_v - 1.0)
        c_e = 1.0 + lam * (c_e - 1.0)

    c_v = float(np.clip(c_v, *CAL_FACTOR_CLAMP))
    c_e = float(np.clip(c_e, *CAL_FACTOR_CLAMP))
    return c_v, c_e


def rolling_online_factors(y, v, e, alpha, W=None):
    """
    Rolling-window calibration: at time t use the last W observations strictly before t.
    Returns arrays c_v_t, c_e_t with length len(y).
    """
    y = np.asarray(y)
    v = np.asarray(v)
    e = np.asarray(e)
    n = len(y)
    if W is None:
        W = _choose_window_for_alpha(alpha, n)
    c_v = np.ones(n, dtype=float)
    c_e = np.ones(n, dtype=float)

    for t in range(n):
        lo = max(0, t - W)
        hi = t  # history up to t-1
        if hi - lo < 5:  # not enough history yet
            continue
        cv, ce = _compute_factor_pair(y[lo:hi], v[lo:hi], e[lo:hi], alpha)
        c_v[t] = cv
        c_e[t] = ce
    return c_v, c_e


def expanding_online_factors(y, v, e, alpha, warmup=None):
    """
    Expanding-window calibration: at time t use all data up to t-1 (after a warmup).
    """
    y = np.asarray(y)
    v = np.asarray(v)
    e = np.asarray(e)
    n = len(y)
    if warmup is None:
        warmup = _choose_warmup_for_alpha(alpha, n)
    c_v = np.ones(n, dtype=float)
    c_e = np.ones(n, dtype=float)

    for t in range(n):
        hi = t
        if hi < warmup:
            continue
        cv, ce = _compute_factor_pair(y[:hi], v[:hi], e[:hi], alpha)
        c_v[t] = cv
        c_e[t] = ce
    return c_v, c_e


# put near your other calib helpers
# Note: _choose_window_for_alpha is now imported from src.utils.eval_tools


def _choose_warmup_for_alpha(alpha, n, target_exceedances=30, min_warmup=250):
    """
    Pick an expanding warmup so we have ≈ target_exceedances exceedances
    before starting online calibration.
    """
    warm = int(np.ceil(target_exceedances / max(alpha, 1e-12)))
    return int(np.clip(warm, min_warmup, max(n // 2, min_warmup)))


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
# Transformer model
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


class BasicVaRTransformer(nn.Module):
    """
    Basic Transformer model for VaR/ES prediction.

    A transformer-based neural network that predicts Value-at-Risk (VaR) and
    Expected Shortfall (ES) from sequential financial data. The model uses
    multi-head attention to capture temporal dependencies in the data.

    Architecture:
    - Input linear layer to project features to model dimension
    - Positional encoding for sequence awareness
    - Transformer encoder layers with multi-head attention
    - Output layer producing VaR and ES predictions

    The model ensures VaR < 0 and ES < VaR through appropriate activation functions.

    Args:
        input_dim (int): Number of input features
        model_dim (int): Hidden dimension of the transformer
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, 2)
        with torch.no_grad():
            self.output_layer.bias[:] = torch.tensor([-0.5, 0.5])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, 2) where:
                - x[:, 0] = VaR predictions (negative values)
                - x[:, 1] = ES predictions (more negative than VaR)
        """
        # Project input to model dimension
        x = self.input_linear(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        h = self.transformer(x)

        # Use last hidden state for prediction
        h_last = h[:, -1, :]

        # Generate raw predictions
        raw = self.output_layer(h_last)
        a, b = raw[:, 0], raw[:, 1]

        # Apply constraints: VaR < 0 and ES < VaR
        v = -F.softplus(a)  # VaR < 0
        e = v - F.softplus(b)  # ES < VaR

        return torch.stack([v, e], dim=1)


def train_with_stride(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    alpha: float = ALPHA,
    seq_len: int = CONTEXT_LEN,
    train_stride: int = TRAIN_STRIDE,
) -> BasicVaRTransformer:
    """
    Train a transformer model using sliding windows with specified stride.

    Creates sliding windows from training data, splits into train/validation sets,
    and trains the transformer model with early stopping and learning rate scheduling.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets (returns)
        input_dim (int): Number of input features
        alpha (float): VaR/ES confidence level
        seq_len (int): Sequence length for transformer
        train_stride (int): Stride for creating training windows

    Returns:
        BasicVaRTransformer: Trained transformer model

    Note:
        Uses FZ0 loss function for VaR/ES prediction and includes
        gradient clipping and early stopping for stable training.
    """
    X_win, y_win = make_windows_with_stride(
        X_train, y_train, seq_len=seq_len, stride=train_stride
    )

    split = int(0.8 * len(X_win))
    gap = min(VAL_GAP_WINDOWS, max(0, len(X_win) - split - 1))
    train_ds = TensorDataset(X_win[: split - gap], y_win[: split - gap])
    val_ds = TensorDataset(X_win[split + gap :], y_win[split + gap :])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BasicVaRTransformer(
        input_dim=input_dim, model_dim=32, num_heads=2, num_layers=1, dropout=0.2
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
                va += loss.item()
        va /= max(1, len(val_loader))

        if va < best_val:
            best_val, best_state, bad = (
                va,
                {k: v.cpu() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_with_sliding_batch(
    model: BasicVaRTransformer,
    X_all: np.ndarray,
    y_all: np.ndarray,
    start_idx: int,
    seq_len: int = CONTEXT_LEN,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(X_all)
    if start_idx < seq_len:
        start_idx = seq_len
    n_pred = T - start_idx
    if n_pred <= 0:
        return np.empty(0), np.empty(0)

    X2 = _ensure_2d_features(X_all)
    xt = torch.as_tensor(X2, dtype=torch.float32)
    Fdim = xt.size(1)

    # xwin = xt.unfold(dimension=0, size=seq_len, step=1)[:-1]
    xwin = xt.unfold(dimension=0, size=seq_len, step=1)
    xwin = _normalize_unfold_shape(xwin, seq_len, Fdim)

    # s = start_idx - seq_len
    # x_eval = xwin[s : s + n_pred]
    # window index j has end = j + seq_len - 1
    # want first window to end at start_idx → j0 = start_idx - (seq_len - 1)
    j0 = start_idx - (seq_len - 1)
    # --- Alignment/bounds guards (dev only) ---
    assert j0 >= 0, "start_idx too small for seq_len; adjust guards"
    assert (j0 + n_pred) <= xwin.size(0), "Not enough eval windows for n_pred"
    x_eval = xwin[j0 : j0 + n_pred]

    preds_v, preds_e = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, n_pred, batch_size):
            xb = x_eval[i : i + batch_size]
            out = model(xb)
            preds_v.append(out[:, 0].cpu())
            preds_e.append(out[:, 1].cpu())
    v = torch.cat(preds_v).numpy()
    e = torch.cat(preds_e).numpy()
    return v, e


def _scalarize_factor(x, mode="mean"):
    """Turn scalar or array-like into a single float for logging/metadata."""
    if np.isscalar(x):
        return float(x)
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 1.0
    if mode == "last":
        return float(arr[-1])
    if mode == "median":
        return float(np.median(arr))
    # default
    return float(np.nanmean(arr))


# ============================
# Main Pipeline
# ============================


def pipeline(
    csv_path: str = "data/merged_data_with_realised_volatility.csv",
    alpha: float = ALPHA,
    feature_parity: bool = True,
    calibrate: bool = False,
    run_tag: Optional[str] = None,
    out_dir: str = "saved_models",
    fig_dir: str = "figures",
) -> Tuple[
    BasicVaRTransformer,
    Dict[str, Any],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Complete pipeline for training and evaluating a transformer model for VaR/ES prediction.

    This is the main entry point for the transformer model. It handles:
    - Data loading and feature engineering
    - Model training with sliding windows
    - Evaluation on test data
    - Optional online calibration
    - Statistical backtesting
    - Results visualization and saving

    Args:
        csv_path (str): Path to CSV file with price data
        alpha (float): VaR/ES confidence level (e.g., 0.01 for 1%)
        feature_parity (bool): If True, use only x_cov feature for fair comparison
        calibrate (bool): Whether to apply online calibration
        run_tag (str, optional): Tag for naming output files
        out_dir (str): Directory for saving models and results
        fig_dir (str): Directory for saving diagnostic plots

    Returns:
        Tuple containing:
        - BasicVaRTransformer: Trained model
        - Dict: Evaluation metrics and metadata
        - Tuple: (v_eval, e_eval, y_aligned, fz0) predictions and targets

    Example:
        model, metrics, (v_eval, e_eval, y_aligned, fz0) = pipeline(
            csv_path="data.csv",
            alpha=0.01,
            feature_parity=True,
            calibrate=True
        )
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = build_inputs_from_prices(df)
    X_tr, y_tr, X_te, y_te, meta = split_and_make_features(
        df, feature_parity=feature_parity, train_frac=TRAIN_FRAC
    )
    feat_label = "parity" if feature_parity else "full"
    input_dim = X_tr.shape[1]

    model = train_with_stride(
        X_tr,
        y_tr,
        input_dim=input_dim,
        alpha=alpha,
        seq_len=CONTEXT_LEN,
        train_stride=TRAIN_STRIDE,
    )

    split_idx = len(X_tr)
    X_all = np.concatenate([X_tr, X_te]).astype(np.float32)
    y_all = np.concatenate([y_tr, y_te]).astype(np.float32)
    v_pred, e_pred = evaluate_with_sliding_batch(
        model,
        X_all,
        y_all,
        start_idx=split_idx,
        seq_len=CONTEXT_LEN,
        batch_size=BATCH_SIZE,
    )
    y_aligned = y_all[split_idx : split_idx + len(v_pred)]

    # === EASY CALIBRATION HERE ===
    if calibrate:
        # Adaptive late-train calibration (chooses stride; clamps; prints diagnostics)
        # c_v, c_e = easy_calibrate(model, X_tr, y_tr, alpha=alpha, seq_len=CONTEXT_LEN)
        c_v, c_e = rolling_online_factors(y_aligned, v_pred, e_pred, alpha)
        # or
        # c_v, c_e = expanding_online_factors(y_aligned, v_pred, e_pred, alpha)
        v_eval = v_pred * c_v
        # keep coherence strictly: ES < VaR
        e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)
        # drift diagnostics (guard against very short tests)
        # n_test = len(y_aligned)
        # if n_test >= 2:
        #     mid = n_test // 2
        #     print(
        #         "Test hit rate (first half):", np.mean(y_aligned[:mid] <= v_eval[:mid])
        #     )
        #     print(
        #         "Test hit rate (second half):", np.mean(y_aligned[mid:] <= v_eval[mid:])
        #     )

        # k1 = n_test // 3
        # k2 = 2 * n_test // 3
        # print(
        #     "Hit rate terciles:",
        #     np.mean(y_aligned[:k1] <= v_eval[:k1]),
        #     np.mean(y_aligned[k1:k2] <= v_eval[k1:k2]),
        #     np.mean(y_aligned[k2:] <= v_eval[k2:]),
        # )
        print_online_drift(y_aligned, v_eval, e_eval, c_v, c_e, alpha, label="ONLINE")

    else:
        v_eval, e_eval = v_pred, e_pred
        c_v, c_e = np.ones_like(v_pred), np.ones_like(e_pred)

    c_v_meta = _scalarize_factor(c_v, mode="mean") if calibrate else 1.0
    c_e_meta = _scalarize_factor(c_e, mode="mean") if calibrate else 1.0

    # --- sanity checks
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

    title = f"Transformer ({feat_label}, {'calibrated' if calibrate else 'raw'})"
    print("=" * 60)
    print(title + (f"  [{run_tag}]" if run_tag else ""))
    print("=" * 60)
    print(f"Hit rate: {hits.mean():.4f} (Target {alpha:.4f})")
    print(f"Kupiec: LR={LR_pof:.4f}, p={p_pof:.4f}")
    print(f"Christoffersen IND: LR={LR_ind:.4f}, p={p_ind:.4f}")
    print(f"Christoffersen CC : LR={LR_cc:.4f}, p={p_cc:.4f}")
    print(f"Avg FZ0: {fz0.mean():.6f}")

    base = f"transformer_{(run_tag + '_') if run_tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
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
        c_v=c_v_meta,
        c_e=c_e_meta,
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
                c_v=c_v_meta,
                c_e=c_e_meta,
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

    metrics = dict(
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
    )

    # Save metrics to JSON (using the same base variable)
    with open(os.path.join(out_dir, f"{base}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics, (v_eval, e_eval, y_aligned, fz0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Transformer model for VaR/ES prediction"
    )
    parser.add_argument(
        "--csv",
        default="data/merged_data_with_realised_volatility.csv",
        help="Path to CSV file (default: data/merged_data_with_realised_volatility.csv)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Alpha level for VaR/ES (default: 0.01)",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Apply calibration (default: False)"
    )
    parser.add_argument(
        "--no-feature-parity",
        dest="feature_parity",
        action="store_false",
        help="Use full features instead of parity (default: True)",
    )
    parser.add_argument(
        "--out-dir",
        default="saved_models",
        help="Output directory for results (default: saved_models)",
    )
    parser.add_argument(
        "--fig-dir",
        default="figures",
        help="Output directory for figures (default: figures)",
    )
    parser.add_argument("--run-tag", help="Optional run tag for file naming")

    args = parser.parse_args()

    pipeline(
        csv_path=args.csv,
        alpha=args.alpha,
        feature_parity=args.feature_parity,
        calibrate=args.calibrate,
        run_tag=args.run_tag,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
    )
