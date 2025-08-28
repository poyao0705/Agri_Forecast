#!/usr/bin/env python3
"""
Evaluation Tools for VaR/ES Prediction

A comprehensive collection of tools for evaluating Value-at-Risk (VaR) and
Expected Shortfall (ES) predictions using statistical backtesting methods.

This module provides:
- FZ0 loss function for VaR/ES evaluation
- Statistical backtesting tests (Kupiec, Christoffersen)
- Calibration utilities for improving predictions
- Diagnostic plotting functions
- Online calibration factors computation

Key Features:
- FZ0 loss function (Patton 2019) for proper scoring
- Unconditional coverage test (Kupiec)
- Independence test (Christoffersen)
- Conditional coverage test
- Rolling online calibration
- Diagnostic visualization

Usage:
    from eval_tools import fz0_per_step, kupiec_pof, christoffersen_independence
    
    # Calculate FZ0 loss
    loss = fz0_per_step(y_true, v_pred, e_pred, alpha=0.01)
    
    # Perform statistical tests
    LR, p_val, x, n = kupiec_pof(hits, alpha=0.01)
    LR_ind, p_ind = christoffersen_independence(hits)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from typing import Tuple, List, Optional, Union

# Headless-safe matplotlib for server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================
# FZ0 Loss Function and Calibration
# ============================

def fz0_per_step(y: Union[np.ndarray, List[float]], 
                 v: Union[np.ndarray, List[float]], 
                 e: Union[np.ndarray, List[float]], 
                 alpha: float) -> np.ndarray:
    """
    FZ0 loss function (Patton 2019) for VaR/ES evaluation.
    
    The FZ0 loss is a proper scoring rule for VaR and ES predictions that
    penalizes both coverage violations and the magnitude of losses beyond VaR.
    It enforces the constraints ES <= VaR and ES < 0.
    
    Args:
        y (Union[np.ndarray, List[float]]): True returns
        v (Union[np.ndarray, List[float]]): VaR predictions (should be negative)
        e (Union[np.ndarray, List[float]]): ES predictions (should be < VaR)
        alpha (float): VaR/ES confidence level (e.g., 0.01 for 1%)
        
    Returns:
        np.ndarray: FZ0 loss values (lower is better)
        
    Note:
        The FZ0 loss is a proper scoring rule, meaning it is minimized when
        predictions are equal to the true conditional quantiles and expectations.
    """
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    e = np.asarray(e, float)
    e = np.minimum(e, v - 1e-12)  # ES <= VaR
    e = np.minimum(e, -1e-12)  # ES < 0
    ind = (y <= v).astype(float)
    term1 = -(1.0 / (alpha * e)) * ind * (v - y)
    term2 = (v / e) + np.log(-e)
    return term1 + term2


def exact_var_factor(y_train: Union[np.ndarray, List[float]], 
                     v_train: Union[np.ndarray, List[float]], 
                     alpha: float, 
                     lo: float = 0.2, 
                     hi: float = 5.0, 
                     iters: int = 40) -> float:
    """
    Find exact VaR calibration factor using bisection method.
    
    Finds the factor c such that mean(y <= c * v) == alpha, where
    y are true returns and v are VaR predictions. Uses bisection
    to efficiently find the optimal scaling factor.
    
    Args:
        y_train (Union[np.ndarray, List[float]]): Training returns
        v_train (Union[np.ndarray, List[float]]): Training VaR predictions
        alpha (float): Target VaR confidence level
        lo (float): Lower bound for bisection search
        hi (float): Upper bound for bisection search
        iters (int): Maximum number of bisection iterations
        
    Returns:
        float: Calibration factor c such that hit rate = alpha
    """
    y = np.asarray(y_train, float)
    v = np.asarray(v_train, float)
    lo, hi = float(lo), float(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        hr = np.mean(y <= mid * v)
        if hr > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_es_factor(y_train: Union[np.ndarray, List[float]], 
                   v_train: Union[np.ndarray, List[float]], 
                   e_train: Union[np.ndarray, List[float]], 
                   alpha: float) -> float:
    """
    Find exact ES calibration factor.
    
    Scales ES predictions so that the mean of true returns given VaR violations
    equals the mean of ES predictions given VaR violations on training data.
    
    Args:
        y_train (Union[np.ndarray, List[float]]): Training returns
        v_train (Union[np.ndarray, List[float]]): Training VaR predictions
        e_train (Union[np.ndarray, List[float]]): Training ES predictions
        alpha (float): VaR confidence level
        
    Returns:
        float: Calibration factor for ES predictions
        
    Note:
        This ensures that ES predictions are properly calibrated to match
        the true conditional expectation of returns given VaR violations.
    """
    y = np.asarray(y_train, float)
    v = np.asarray(v_train, float)
    e = np.asarray(e_train, float)
    hits = y <= v
    if hits.sum() == 0:
        return 1.0
    target = y[hits].mean()
    pred = e[hits].mean()
    if pred == 0:
        return 1.0
    return target / pred


# ============================
# Statistical Backtesting Tests
# ============================

def kupiec_pof(hits: Union[np.ndarray, List[int]], 
               alpha: float) -> Tuple[float, float, int, int]:
    """
    Kupiec test for unconditional coverage (proportion of failures).
    
    Tests whether the proportion of VaR violations equals the expected
    proportion alpha. The null hypothesis is that the hit rate equals alpha.
    
    Args:
        hits (Union[np.ndarray, List[int]]): Binary sequence of VaR violations (1 = violation, 0 = no violation)
        alpha (float): Expected VaR violation rate (e.g., 0.01 for 1% VaR)
        
    Returns:
        Tuple[float, float, int, int]: (LR_statistic, p_value, violations_count, total_observations)
        
    Note:
        The test statistic follows a chi-squared distribution with 1 degree of freedom
        under the null hypothesis. Low p-values indicate rejection of the null.
    """
    h = np.asarray(hits).astype(int)
    n = len(h)
    x = int(h.sum())
    if n == 0:
        return np.nan, np.nan, 0, 0
    p_hat = x / n
    if p_hat in (0, 1):
        # degenerate; LR=inf if mismatch, but keep finite guard
        return np.inf, 0.0, x, n
    ll0 = (n - x) * np.log(max(1 - alpha, 1e-12)) + x * np.log(max(alpha, 1e-12))
    ll1 = (n - x) * np.log(max(1 - p_hat, 1e-12)) + x * np.log(max(p_hat, 1e-12))
    LR = -2 * (ll0 - ll1)
    p = 1 - chi2.cdf(LR, df=1)
    return LR, p, x, n


def christoffersen_independence(hits: Union[np.ndarray, List[int]]) -> Tuple[float, float]:
    """
    Christoffersen independence test for VaR violations.
    
    Tests whether VaR violations are independent over time using a Markov chain
    approach. The null hypothesis is that violations are independent.
    
    Args:
        hits (Union[np.ndarray, List[int]]): Binary sequence of VaR violations (1 = violation, 0 = no violation)
        
    Returns:
        Tuple[float, float]: (LR_statistic, p_value)
        
    Note:
        The test examines whether the probability of a violation depends on
        whether there was a violation in the previous period. The test statistic
        follows a chi-squared distribution with 1 degree of freedom.
    """
    h = np.asarray(hits).astype(int)
    if len(h) < 2:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(h)):
        i, j = h[t - 1], h[t]
        if i == 0 and j == 0:
            n00 += 1
        if i == 0 and j == 1:
            n01 += 1
        if i == 1 and j == 0:
            n10 += 1
        if i == 1 and j == 1:
            n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    def ll_bin(n, k, p):
        if n == 0:
            return 0.0
        p = min(max(p, 1e-12), 1 - 1e-12)
        return k * np.log(p) + (n - k) * np.log(1 - p)

    ll_u = ll_bin(n0, n01, pi0) + ll_bin(n1, n11, pi1)
    ll_r = ll_bin(n0 + n1, n01 + n11, pi)
    LR = -2 * (ll_r - ll_u)
    p = 1 - chi2.cdf(LR, df=1)
    return LR, p


def christoffersen_cc(hits: Union[np.ndarray, List[int]], 
                     alpha: float) -> Tuple[float, float]:
    """
    Christoffersen conditional coverage test.
    
    Combines the Kupiec test (unconditional coverage) and Christoffersen
    independence test into a single test for conditional coverage.
    The null hypothesis is that violations have the correct probability
    and are independent over time.
    
    Args:
        hits (Union[np.ndarray, List[int]]): Binary sequence of VaR violations
        alpha (float): Expected VaR violation rate
        
    Returns:
        Tuple[float, float]: (LR_statistic, p_value)
        
    Note:
        The test statistic is the sum of Kupiec and Independence test statistics
        and follows a chi-squared distribution with 2 degrees of freedom.
    """
    LR_pof, _, _, _ = kupiec_pof(hits, alpha)
    LR_ind, _ = christoffersen_independence(hits)
    LR_cc = LR_pof + LR_ind
    p = 1 - chi2.cdf(LR_cc, df=2)
    return LR_cc, p


# Convenience: p-only wrappers (if you ever want them)
def kupiec_pof_p(alpha_or_hits, hits_or_alpha=None):
    if hits_or_alpha is None:
        raise TypeError(
            "kupiec_pof_p expects two arguments: (alpha, hits) or (hits, alpha)."
        )
    a, h = (
        (alpha_or_hits, hits_or_alpha)
        if isinstance(alpha_or_hits, (float, int))
        else (hits_or_alpha, alpha_or_hits)
    )
    _, p, _, _ = kupiec_pof(h, float(a))
    return p


def christoffersen_independence_p(hits):
    _, p = christoffersen_independence(hits)
    return p


def christoffersen_cc_p(alpha_or_hits, hits_or_alpha=None):
    if hits_or_alpha is None:
        raise TypeError(
            "christoffersen_cc_p expects two arguments: (alpha, hits) or (hits, alpha)."
        )
    a, h = (
        (alpha_or_hits, hits_or_alpha)
        if isinstance(alpha_or_hits, (float, int))
        else (hits_or_alpha, alpha_or_hits)
    )
    _, p = christoffersen_cc(h, float(a))
    return p


# ============================
# Diebold-Mariano Test
# ============================

def newey_west_variance(d: Union[np.ndarray, List[float]], 
                       lag: int = 5) -> float:
    """
    Compute Newey-West variance estimator for time series data.
    
    Estimates the variance of a time series accounting for autocorrelation
    using the Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent)
    estimator.
    
    Args:
        d (Union[np.ndarray, List[float]]): Time series data
        lag (int): Maximum lag for autocorrelation adjustment
        
    Returns:
        float: Newey-West variance estimate
    """
    d = np.asarray(d, float)
    n = len(d)
    mu = d.mean()
    u = d - mu
    gamma0 = (u @ u) / n
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma = (u[l:] * u[:-l]).sum() / n
        var += 2 * w * gamma
    return var


def diebold_mariano(loss1: Union[np.ndarray, List[float]], 
                   loss2: Union[np.ndarray, List[float]], 
                   lag: int = 1) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests whether two forecasting models have significantly different
    predictive accuracy using the Diebold-Mariano test statistic.
    
    Args:
        loss1 (Union[np.ndarray, List[float]]): Loss values from model 1
        loss2 (Union[np.ndarray, List[float]]): Loss values from model 2
        lag (int): Maximum lag for Newey-West variance estimation
        
    Returns:
        Tuple[float, float]: (DM_statistic, p_value)
        
    Note:
        The null hypothesis is that both models have equal predictive accuracy.
        The test uses Newey-West variance estimation to account for autocorrelation.
    """
    l1 = np.asarray(loss1, float)
    l2 = np.asarray(loss2, float)
    m = min(len(l1), len(l2))
    d = l1[:m] - l2[:m]
    n = len(d)
    var = newey_west_variance(d, lag=max(1, min(lag, n // 4)))
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / n)
    p = 2 * (1 - norm.cdf(abs(dm)))
    return dm, p


# ============================
# Diagnostic Plotting
# ============================

def plot_var_es_diagnostics(
    y_true: Union[np.ndarray, List[float]],
    var_pred: Union[np.ndarray, List[float]],
    es_pred: Union[np.ndarray, List[float]],
    alpha: float,
    title: str,
    out_dir: str,
    fname_prefix: str
) -> str:
    """
    Create comprehensive diagnostic plots for VaR/ES evaluation.
    
    Generates a 2x2 diagnostic figure with:
    - VaR backtest with breach indicators
    - Rolling hit rate with automatic window selection
    - Tail Q-Q plot of breach returns
    - ES vs Actual comparison on breaches
    
    Args:
        y_true (Union[np.ndarray, List[float]]): True returns
        var_pred (Union[np.ndarray, List[float]]): VaR predictions
        es_pred (Union[np.ndarray, List[float]]): ES predictions
        alpha (float): VaR confidence level
        title (str): Plot title
        out_dir (str): Output directory for saving
        fname_prefix (str): Filename prefix
        
    Returns:
        str: Filepath of the saved PNG file
    """
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    var_pred = np.asarray(var_pred)
    es_pred = np.minimum(np.asarray(es_pred), var_pred - 1e-8)  # safety

    hits = (y_true <= var_pred).astype(int)
    breach = y_true[hits == 1]
    idx = np.arange(len(y_true))
    win = min(20, max(5, len(hits) // 3))

    fig = plt.figure(figsize=(14, 10))

    # (1) VaR backtest
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(idx, y_true, lw=0.8, alpha=0.85, label="Actual")
    ax1.plot(idx, var_pred, lw=1.2, label=f"VaR ({alpha:.0%})")
    if hits.sum() > 0:
        ax1.scatter(
            np.where(hits == 1)[0],
            breach,
            s=14,
            c="k",
            zorder=5,
            label=f"Breaches ({hits.sum()})",
        )
    ax1.set_title(f"{title} — VaR backtest")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # (2) Rolling hit rate
    ax2 = plt.subplot(2, 2, 2)
    roll = pd.Series(hits).rolling(window=win).mean()
    ax2.plot(idx, roll, label=f"Rolling hit rate ({win})")
    ax2.axhline(y=alpha, linestyle="--", label=f"Target ({alpha:.0%})")
    ax2.set_title("Rolling hit rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # (3) Tail Q–Q of breach returns
    ax3 = plt.subplot(2, 2, 3)
    if hits.sum() > 0:
        srt = np.sort(breach)
        q = np.linspace(0, 1, len(srt))
        ax3.scatter(q, srt, s=14, alpha=0.75)
        ax3.set_xlabel("Theoretical quantiles")
        ax3.set_ylabel("Breach returns")
        ax3.set_title("Tail Q–Q (breaches)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No breaches", ha="center", va="center")
        ax3.axis("off")

    # (4) ES vs Actual on breaches
    ax4 = plt.subplot(2, 2, 4)
    if hits.sum() > 0:
        es_b = es_pred[hits == 1]
        lo = min(es_b.min(), breach.min())
        hi = max(es_b.max(), breach.max())
        ax4.scatter(es_b, breach, s=14, alpha=0.75)
        ax4.plot([lo, hi], [lo, hi], "r--", lw=1.0, label="y=x")
        ax4.set_xlabel("Predicted ES")
        ax4.set_ylabel("Actual returns")
        ax4.set_title("ES calibration (breaches)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best")
    else:
        ax4.text(0.5, 0.5, "No breaches", ha="center", va="center")
        ax4.axis("off")

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{fname_prefix}_diagnostics.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _safe_mean(x: Union[np.ndarray, List[float]]) -> float:
    """
    Safely compute mean of array, returning NaN if empty.
    
    Args:
        x (Union[np.ndarray, List[float]]): Input array
        
    Returns:
        float: Mean value or NaN if array is empty
    """
    return float(np.mean(x)) if len(x) else float("nan")


def _pct(x: Union[np.ndarray, List[float]], q: float) -> float:
    """
    Safely compute quantile of array, returning NaN if empty.
    
    Args:
        x (Union[np.ndarray, List[float]]): Input array
        q (float): Quantile (0-1)
        
    Returns:
        float: Quantile value or NaN if array is empty
    """
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _choose_window_for_alpha(alpha: float, n: int, target_exceedances: int = 30, 
                           min_w: int = 200) -> int:
    """
    Choose rolling window size for target number of exceedances.
    
    Selects a window size W such that the expected number of exceedances
    is approximately target_exceedances. W ≈ target_exceedances / α.
    
    Args:
        alpha (float): VaR confidence level
        n (int): Total number of observations
        target_exceedances (int): Target number of exceedances in window
        min_w (int): Minimum window size
        
    Returns:
        int: Optimal window size clipped to [min_w, n]
    """
    W = int(np.ceil(target_exceedances / max(alpha, 1e-12)))
    return int(np.clip(W, min_w, n))


def print_online_drift(y: Union[np.ndarray, List[float]], 
                      v: Union[np.ndarray, List[float]], 
                      e: Union[np.ndarray, List[float]], 
                      c_v: Union[np.ndarray, List[float], float], 
                      c_e: Union[np.ndarray, List[float], float], 
                      alpha: float, 
                      label: str = "ONLINE") -> None:
    """
    Print comprehensive online diagnostics for VaR/ES predictions.
    
    Provides detailed analysis of:
    - Hit rate drift across different time periods
    - Rolling hit rate statistics
    - ES coherence and calibration
    - Calibration factor drift over time
    
    Args:
        y (Union[np.ndarray, List[float]]): Realized returns/aligned targets
        v (Union[np.ndarray, List[float]]): Calibrated VaR predictions
        e (Union[np.ndarray, List[float]]): Calibrated ES predictions
        c_v (Union[np.ndarray, List[float], float]): VaR calibration factors
        c_e (Union[np.ndarray, List[float], float]): ES calibration factors
        alpha (float): Nominal VaR exceedance level
        label (str): Label for output identification
    """
    y = np.asarray(y)
    v = np.asarray(v)
    e = np.asarray(e)
    hits = y <= v
    n = len(y)
    mid = n // 2
    k1 = n // 3
    k2 = 2 * n // 3

    # overall / halves / terciles
    hr_all = _safe_mean(hits)
    hr_1 = _safe_mean(hits[:mid])
    hr_2 = _safe_mean(hits[mid:])
    hr_t1 = _safe_mean(hits[:k1])
    hr_t2 = _safe_mean(hits[k1:k2])
    hr_t3 = _safe_mean(hits[k2:])

    # short rolling window to see local drift (target ~30 exceedances)
    W = _choose_window_for_alpha(alpha, n, target_exceedances=30, min_w=200)
    if n >= W:
        roll = (
            np.convolve(hits.astype(float), np.ones(W, dtype=float), mode="valid") / W
        )
        rmin, rmed, rmax = float(roll.min()), float(np.median(roll)), float(roll.max())
        print(
            f"[{label}] rolling hit-rate (W={W})  min={rmin:.4f}  med={rmed:.4f}  max={rmax:.4f}"
        )

    print(
        f"[{label}] hit-rate overall={hr_all:.4f} vs α={alpha:.3f}; "
        f"halves=({hr_1:.4f},{hr_2:.4f}); "
        f"terciles=({hr_t1:.4f},{hr_t2:.4f},{hr_t3:.4f})"
    )

    # ES coherence check on exceedances
    if hits.any():
        es_real = float(y[hits].mean())
        es_pred = float(e[hits].mean())
        print(
            f"[{label}] ES(real)={es_real:.5f}  ES(pred)={es_pred:.5f}  n_ex={int(hits.sum())}"
        )

    # factor drift summary (works if c_v/c_e are arrays; falls back if scalars)
    c_v = np.asarray(c_v)
    c_e = np.asarray(c_e)
    if c_v.ndim == 1 and len(c_v) == n:
        print(
            f"[{label}] c_v drift  first={c_v[0]:.4f}  last={c_v[-1]:.4f}  "
            f"mean={_safe_mean(c_v):.4f}  IQR=[{_pct(c_v,0.25):.4f},{_pct(c_v,0.75):.4f}]"
        )
    if c_e.ndim == 1 and len(c_e) == n:
        print(
            f"[{label}] c_e drift  first={c_e[0]:.4f}  last={c_e[-1]:.4f}  "
            f"mean={_safe_mean(c_e):.4f}  IQR=[{_pct(c_e,0.25):.4f},{_pct(c_e,0.75):.4f}]"
        )
