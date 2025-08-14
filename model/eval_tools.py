# eval_tools.py
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

# Headless-safe matplotlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# FZ0 loss & calibration utils
# -----------------------------
def fz0_per_step(y, v, e, alpha):
    """
    FZ0 (Patton 2019) for lower tail. Enforces ES <= VaR and ES < 0.
    Returns a scalar score (lower is better).
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


def exact_var_factor(y_train, v_train, alpha, lo=0.2, hi=5.0, iters=40):
    """Find c so that mean( y <= c * v ) == alpha (bisection)."""
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


def exact_es_factor(y_train, v_train, e_train, alpha):
    """Scale ES so that mean(y | y<=v) equals mean(e | hits) on train."""
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


# -----------------------------
# Backtesting helpers (legacy: return LR and p)
# -----------------------------
def kupiec_pof(hits, alpha):
    """
    Unconditional coverage (Kupiec) LR test.
    Returns (LR_pof, p_value, x, n).
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


def christoffersen_independence(hits):
    """
    Christoffersen independence test (Markov).
    Returns (LR_ind, p_value).
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


def christoffersen_cc(hits, alpha):
    """
    Conditional coverage = Kupiec + Independence.
    Returns (LR_cc, p_value).
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


# -----------------------------
# Diebold–Mariano
# -----------------------------
def newey_west_variance(d, lag=5):
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


def diebold_mariano(loss1, loss2, lag=1):
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


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_var_es_diagnostics(
    y_true, var_pred, es_pred, alpha, title, out_dir, fname_prefix
):
    """
    Save a 2x2 diagnostic figure:
      (1) VaR backtest with breaches
      (2) Rolling hit rate (window auto)
      (3) Tail Q–Q of breach returns
      (4) ES vs Actual on breaches (with y=x)
    Returns the filepath of the saved PNG.
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
