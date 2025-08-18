# baseline_classic_var_es.py
import os
import json
import numpy as np
import pandas as pd
from typing import Optional

# ---------- Optional eval tools ----------
try:
    from src.utils.eval_tools import (
        kupiec_pof,
        christoffersen_independence,
        christoffersen_cc,
        exact_var_factor,
        exact_es_factor,
        fz0_per_step as _fz0_evaltools,
    )

    HAVE_EVAL_TOOLS = True
except Exception:
    HAVE_EVAL_TOOLS = False

# ---------- Optional ARCH ----------
try:
    from arch.univariate import ConstantMean, GARCH, StudentsT

    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

from scipy.stats import norm
from scipy.stats import t as student_t


# ============================================================
# Math helpers
# ============================================================
def _normal_var_es(mu: float, sigma: float, alpha: float):
    z = norm.ppf(alpha)
    var = mu + sigma * z
    es = mu - sigma * norm.pdf(z) / alpha
    return float(var), float(es)


def _t_var_es_standardized(mu: float, sigma: float, nu: float, alpha: float):
    if nu <= 2.0:
        raise ValueError("nu must be > 2 for finite variance.")
    scale = np.sqrt((nu - 2.0) / nu)
    z = student_t.ppf(alpha, df=nu)  # standard t lower-tail
    fz = student_t.pdf(z, df=nu)
    q_std = scale * z
    var = mu + sigma * q_std
    es = mu - sigma * scale * ((nu + z * z) / ((nu - 1.0) * alpha)) * fz
    return float(var), float(es)


def fz0_loss_per_step(y: float, v: float, e: float, alpha: float) -> float:
    if HAVE_EVAL_TOOLS:
        return float(_fz0_evaltools(y, v, e, alpha))
    # fallback
    if e >= v:
        e = v - 1e-12
    if e >= 0:
        e = -1e-12
    hit = 1.0 if y <= v else 0.0
    return -(hit * (v - y)) / (alpha * e) + (v / e) + np.log(-e) - 1.0


# ============================================================
# Baselines
# ============================================================
def _baseline_garch_t(
    returns: pd.Series, alpha: float, init_window: int, show_progress: bool
):
    if not HAVE_ARCH:
        raise RuntimeError(
            "arch is not installed; use method='rm_normal'/'rw' or install 'arch'."
        )

    from arch.univariate import ConstantMean, GARCH, StudentsT
    from scipy.stats import t as student_t

    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n <= init_window + 1:
        raise ValueError("Not enough data for expanding evaluation.")
    dates = r.index

    preds, fz_losses, hits = [], [], []

    for t in range(init_window, n - 1):
        # expanding window up to t
        y_hist = r.iloc[: t + 1].astype(float)

        # --- Robust scaling: work in percent to avoid DataScaleWarning ---
        y_hist_s = 100.0 * y_hist.values  # percent

        # Build model in scaled units and disable internal rescaling
        am = ConstantMean(y_hist_s)
        am.volatility = GARCH(p=1, o=0, q=1)
        am.distribution = StudentsT()
        am.rescale = False

        # Fit with robust fallback
        try:
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 4000})
        except Exception:
            # fallback to a derivative-free method if SLSQP struggles
            res = am.fit(
                disp="off",
                show_warning=False,
                method="powell",
                options={"maxiter": 4000},
            )

        # 1-step-ahead forecast (still in percent units)
        f = res.forecast(horizon=1, reindex=False)
        mu_p = float(f.mean.iloc[-1, 0])  # percent
        sig_p = float(np.sqrt(f.variance.iloc[-1, 0]))  # percent

        # Convert back to decimal units
        mu = mu_p / 100.0
        sig = sig_p / 100.0

        # Degrees of freedom; guard against pathological estimates
        try:
            nu = float(res.params.get("nu", 8.0))
        except Exception:
            nu = 8.0
        if nu <= 2.05:
            nu = 2.05

        # --- Student-t VaR/ES in UNIT-VARIANCE parameterization ---
        # arch's StudentsT is standardized to unit variance, so:
        # q_std = t_ppf * sqrt((nu-2)/nu)
        qz = float(student_t.ppf(alpha, df=nu))  # standard t quantile
        q_std = qz * np.sqrt((nu - 2.0) / nu)  # unit-variance t quantile

        # ES of unit-variance t (left tail, returns are negative):
        # Start from standard-t ES and re-scale to unit variance
        pdf = float(student_t.pdf(qz, df=nu))
        es_std = -((nu + qz * qz) / ((nu - 1.0) * alpha)) * pdf
        es_std *= np.sqrt((nu - 2.0) / nu)

        # Final VaR/ES in data units
        v = mu + sig * q_std
        e = mu + sig * es_std

        # Next realized return (decimal units)
        y_next = float(r.iloc[t + 1])

        # Loss & hit
        loss = fz0_loss_per_step(y_next, v, e, alpha)
        hit = 1.0 if y_next <= v else 0.0

        preds.append((dates[t + 1], v, e, mu, sig, nu))
        fz_losses.append(loss)
        hits.append(hit)

        if show_progress and (t - init_window) % 250 == 0:
            print(f"{t-init_window:5d}/{n-init_window-1}  VaR={v:.6f} ES={e:.6f}")

    df = pd.DataFrame(
        preds, columns=["date", "VaR", "ES", "mu", "sigma", "nu"]
    ).set_index("date")

    out = {
        "preds": df,
        "avg_fz0_loss": float(np.mean(fz_losses)),
        "loss_series": pd.Series(fz_losses, index=df.index, name="fz0"),
        "hit_rate": float(np.mean(hits)),
        "n": len(df),
    }

    if HAVE_EVAL_TOOLS:
        breaches = np.array(hits, dtype=int)
        LR_p, p_p, _, _ = kupiec_pof(breaches, alpha)
        LR_i, p_i = christoffersen_independence(breaches)
        LR_c, p_c = christoffersen_cc(breaches, alpha)
        out.update(
            {
                "kupiec_LR": float(LR_p),
                "kupiec_pof_p": float(p_p),
                "ind_LR": float(LR_i),
                "chr_ind_p": float(p_i),
                "cc_LR": float(LR_c),
                "chr_cc_p": float(p_c),
            }
        )
    return out


def _baseline_rm_normal(
    returns: pd.Series, alpha: float, init_window: int, show_progress: bool
):
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n <= init_window + 1:
        raise ValueError("Not enough data for expanding evaluation.")
    dates = r.index

    # RiskMetrics EWMA volatility (lambda=0.94 daily by default)
    lam = 0.94
    var = np.zeros(n)
    var[:init_window] = np.var(r.iloc[:init_window], ddof=1)
    for i in range(init_window, n):
        prev = r.iloc[i - 1]
        var[i] = lam * var[i - 1] + (1.0 - lam) * (prev**2)
    sigma = np.sqrt(var)

    preds, fz_losses, hits = [], [], []
    for t in range(init_window, n - 1):
        mu = 0.0
        sig = float(sigma[t])
        v, e = _normal_var_es(mu, sig, alpha)
        y_next = float(r.iloc[t + 1])
        loss = fz0_loss_per_step(y_next, v, e, alpha)
        hit = 1.0 if y_next <= v else 0.0
        preds.append((dates[t + 1], v, e, mu, sig, np.nan))
        fz_losses.append(loss)
        hits.append(hit)
        if show_progress and (t - init_window) % 250 == 0:
            print(f"{t-init_window:5d}/{n-init_window-1}  VaR={v:.5f} ES={e:.5f}")

    df = pd.DataFrame(
        preds, columns=["date", "VaR", "ES", "mu", "sigma", "nu"]
    ).set_index("date")
    out = {
        "preds": df,
        "avg_fz0_loss": float(np.mean(fz_losses)),
        "loss_series": pd.Series(fz_losses, index=df.index, name="fz0"),
        "hit_rate": float(np.mean(hits)),
        "n": len(df),
    }
    if HAVE_EVAL_TOOLS:
        breaches = np.array(hits, dtype=int)
        LR_p, p_p, _, _ = kupiec_pof(breaches, alpha)
        LR_i, p_i = christoffersen_independence(breaches)
        LR_c, p_c = christoffersen_cc(breaches, alpha)
        out.update(
            {
                "kupiec_LR": float(LR_p),
                "kupiec_pof_p": float(p_p),
                "ind_LR": float(LR_i),
                "chr_ind_p": float(p_i),
                "cc_LR": float(LR_c),
                "chr_cc_p": float(p_c),
            }
        )
    return out


def run_rw_baseline(returns: pd.Series, alpha: float = 0.01, window: int = 250):
    r = pd.Series(returns).astype(float).dropna()
    dates, var_list, es_list, hits, losses = [], [], [], [], []
    for t in range(window, len(r)):
        hist = r.iloc[t - window : t]
        try:
            v = float(np.quantile(hist, alpha, method="lower"))
        except TypeError:
            v = float(np.quantile(hist, alpha))
        tail = hist[hist <= v]
        e = float(tail.mean()) if len(tail) else v
        y_next = float(r.iloc[t])
        hits.append(1.0 if y_next <= v else 0.0)
        losses.append(fz0_loss_per_step(y_next, v, e, alpha))
        dates.append(r.index[t])
        var_list.append(v)
        es_list.append(e)
    df = pd.DataFrame(
        {"VaR": var_list, "ES": es_list}, index=pd.Index(dates, name="date")
    )
    out = {
        "preds": df,
        "avg_fz0_loss": float(np.mean(losses)),
        "loss_series": pd.Series(losses, index=df.index, name="fz0"),
        "hit_rate": float(np.mean(hits)),
        "n": len(df),
    }
    if HAVE_EVAL_TOOLS and len(hits) > 0:
        breaches = np.array(hits, dtype=int)
        LR_p, p_p, _, _ = kupiec_pof(breaches, alpha)
        LR_i, p_i = christoffersen_independence(breaches)
        LR_c, p_c = christoffersen_cc(breaches, alpha)
        out.update(
            {
                "kupiec_LR": float(LR_p),
                "kupiec_pof_p": float(p_p),
                "ind_LR": float(LR_i),
                "chr_ind_p": float(p_i),
                "cc_LR": float(LR_c),
                "chr_cc_p": float(p_c),
            }
        )
    return out


def run_rw_baseline_aligned(
    returns: pd.Series,
    split_idx: int,
    alpha: float = 0.01,
    window: int = 250,
    ramp: bool = True,
    include_train_history: bool = True,
):
    """
    Historical-simulation RW VaR/ES evaluated ONLY on the post-split test window.

    Args:
        returns: full return series (pd.Series).
        split_idx: index (integer position) where test starts (same split as models).
        alpha: tail level.
        window: rolling window length (e.g., 250).
        ramp: if True, emit a forecast for every test day by growing history
              inside the allowed region until `window` is reached.
        include_train_history: if True, history for day t can include pre-split data
              (i.e., left bound >= 0). If False, history is restricted to test-only
              data (i.e., left bound >= split_idx).

    Returns:
        dict with keys: preds(DataFrame with VaR/ES), avg_fz0_loss, loss_series,
        hit_rate, n, and (if eval_tools available) Kupiec/Christoffersen stats.
    """
    r = pd.Series(returns).astype(float).dropna()
    assert 0 <= split_idx < len(r), "Bad split_idx"

    dates, var_list, es_list, hits, losses = [], [], [], [], []

    for t in range(split_idx, len(r)):
        if include_train_history:
            left_limit = 0
        else:
            left_limit = split_idx

        if ramp:
            left = max(left_limit, t - window)
        else:
            left = t - window
            if left < left_limit:
                # need a full window inside the allowed region
                continue

        hist = r.iloc[left:t]
        if len(hist) == 0:
            # shouldn't happen with ramp=True, but guard anyway
            v = np.nan
            e = np.nan
        else:
            try:
                v = float(np.quantile(hist, alpha, method="lower"))
            except TypeError:
                v = float(np.quantile(hist, alpha))
            tail = hist[hist <= v]
            e = float(tail.mean()) if len(tail) else v

        y_next = float(r.iloc[t])
        hit = float(y_next <= v) if np.isfinite(v) else np.nan
        loss = (
            float(fz0_loss_per_step(y_next, v, e, alpha)) if np.isfinite(v) else np.nan
        )

        dates.append(r.index[t])
        var_list.append(v)
        es_list.append(e)
        hits.append(hit)
        losses.append(loss)

    df = pd.DataFrame(
        {"VaR": var_list, "ES": es_list}, index=pd.Index(dates, name="date")
    )

    out = {
        "preds": df,
        "avg_fz0_loss": float(np.nanmean(losses)) if len(losses) else np.nan,
        "loss_series": (
            pd.Series(losses, index=df.index, name="fz0")
            if len(df)
            else pd.Series(dtype=float)
        ),
        "hit_rate": float(np.nanmean(hits)) if len(hits) else np.nan,
        "n": int(df.shape[0]),
    }

    if HAVE_EVAL_TOOLS and len(df) > 0:
        # drop any NaNs that might appear if ramp produced very short early windows
        mask = np.isfinite(df["VaR"].values)
        breaches = np.array([int(h) for h, m in zip(hits, mask) if m])
        if len(breaches) > 0:
            LR_p, p_p, _, _ = kupiec_pof(breaches, alpha)
            LR_i, p_i = christoffersen_independence(breaches)
            LR_c, p_c = christoffersen_cc(breaches, alpha)
            out.update(
                dict(
                    kupiec_LR=float(LR_p),
                    kupiec_pof_p=float(p_p),
                    ind_LR=float(LR_i),
                    chr_ind_p=float(p_i),
                    cc_LR=float(LR_c),
                    chr_cc_p=float(p_c),
                )
            )
    return out


# ============================================================
# Pipeline interface to match SRNN/Transformer
# ============================================================
def _load_returns_from_csv(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    for c in ["date", "Date", "timestamp", "Timestamp"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c)
            break
    for c in ["ret", "return", "returns", "log_ret", "log_return"]:
        if c in df.columns:
            r = pd.Series(df[c].astype(float), index=df.index)
            return r.dropna()
    if "close" not in df.columns:
        raise ValueError(
            "CSV must have a 'close' column or an explicit returns column."
        )
    px = pd.Series(df["close"].astype(float), index=df.index)
    r = np.log(px / px.shift(1))
    return r.dropna()


def _recompute_metrics(preds_df: pd.DataFrame, r_eval: pd.Series, alpha: float):
    y = r_eval.reindex(preds_df.index).astype(float)
    losses = [
        fz0_loss_per_step(float(y_t), float(v), float(e), alpha)
        for y_t, v, e in zip(y.values, preds_df["VaR"].values, preds_df["ES"].values)
    ]
    hits = [
        1.0 if float(y_t) <= float(v) else 0.0
        for y_t, v in zip(y.values, preds_df["VaR"].values)
    ]
    out = {
        "loss_series": pd.Series(losses, index=preds_df.index, name="fz0"),
        "avg_fz0_loss": float(np.mean(losses)),
        "hit_rate": float(np.mean(hits)),
        "n": len(preds_df),
    }
    if HAVE_EVAL_TOOLS and len(hits) > 0:
        breaches = np.array(hits, dtype=int)
        LR_p, p_p, _, _ = kupiec_pof(breaches, alpha)
        LR_i, p_i = christoffersen_independence(breaches)
        LR_c, p_c = christoffersen_cc(breaches, alpha)
        out.update(
            {
                "kupiec_LR": float(LR_p),
                "kupiec_pof_p": float(p_p),
                "ind_LR": float(LR_i),
                "chr_ind_p": float(p_i),
                "cc_LR": float(LR_c),
                "chr_cc_p": float(p_c),
            }
        )
    return out


def _calibrate_if_requested(
    out: dict, r_eval: pd.Series, alpha: float, do_calibrate: bool
) -> dict:
    if not do_calibrate or not HAVE_EVAL_TOOLS:
        return out
    preds = out["preds"].copy()
    y = r_eval.reindex(preds.index).astype(float)
    fV = exact_var_factor(y.values, preds["VaR"].values, alpha)
    fE = exact_es_factor(y.values, preds["VaR"].values, preds["ES"].values, alpha)
    preds["VaR"] = fV * preds["VaR"]
    preds["ES"] = fE * preds["ES"]
    out = out.copy()
    out["preds"] = preds
    out.update(_recompute_metrics(preds, r_eval, alpha))
    return out


def run_baseline(
    returns: pd.Series,
    alpha: float = 0.01,
    method: str = "garch_t",  # "garch_t" | "rm_normal"
    init_window: int = 2000,
    show_progress: bool = False,
):
    if method == "garch_t":
        return _baseline_garch_t(returns, alpha, init_window, show_progress)
    elif method == "rm_normal":
        return _baseline_rm_normal(returns, alpha, init_window, show_progress)
    else:
        raise ValueError("method must be 'garch_t' or 'rm_normal'")


def pipeline(
    csv_path: str,
    alpha: float = 0.01,
    method: str = "garch_t",  # "garch_t" | "rm_normal" | "rw"
    init_window: int = 250,
    calibrate: bool = False,
    run_tag: Optional[str] = None,
    show_progress: bool = False,
    train_frac: float = 0.5,
    ramp: bool = True,
    include_train_history: bool = True,
    out_dir: str = "saved_models",
    fig_dir: str = "figures",
):
    r = _load_returns_from_csv(csv_path)
    if method == "rw":
        split_idx = int(len(r) * train_frac)
        out = run_rw_baseline_aligned(
            returns=r,
            split_idx=split_idx,
            alpha=alpha,
            window=init_window,
            ramp=ramp,
            include_train_history=include_train_history,
        )
    else:
        out = run_baseline(
            r,
            alpha=alpha,
            method=method,
            init_window=init_window,
            show_progress=show_progress,
        )

    out = _calibrate_if_requested(out, r_eval=r, alpha=alpha, do_calibrate=calibrate)

    # Set appropriate model name based on method
    if method == "rw":
        model_name = f"rw-{init_window}"
    else:
        model_name = run_tag or f"Baseline[{method}]"

    # Extract predictions and metrics
    preds = out["preds"]
    y_aligned = preds.index.astype("datetime64[ns]").astype("int64")
    v_eval = preds["VaR"].values.astype(float)
    e_eval = preds["ES"].values.astype(float)
    fz0 = np.asarray(out["loss_series"].values, float)

    # Calculate hits
    hits = (preds.index <= preds["VaR"]).astype(int)

    # Create model description
    # Note: We need to determine the actual feature type from the run_tag or context
    # For now, we'll use a placeholder that will be corrected by aggregate_results.py
    if method == "rw":
        model_desc = f"{model_name} (features, {'calibrated' if calibrate else 'raw'})"
    else:
        model_desc = (
            f"Baseline[{method}] (features, {'calibrated' if calibrate else 'raw'})"
        )

    # Create metrics dictionary
    metrics = {
        "model": model_name,
        "model_desc": model_desc,
        "alpha": float(alpha),
        "hit_rate": float(out["hit_rate"]),
        "kupiec_LR": float(out.get("kupiec_LR", 0.0)),
        "kupiec_p": float(out.get("kupiec_pof_p", 0.0)),
        "ind_LR": float(out.get("ind_LR", 0.0)),
        "ind_p": float(out.get("chr_ind_p", 0.0)),
        "cc_LR": float(out.get("cc_LR", 0.0)),
        "cc_p": float(out.get("chr_cc_p", 0.0)),
        "avg_fz0": float(out["avg_fz0_loss"]),
        "tag": run_tag or "",
        "n": int(len(preds)),
    }

    # Save metrics to JSON
    os.makedirs(out_dir, exist_ok=True)
    base = f"baseline_{(run_tag + '_') if run_tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
        " ", ""
    )
    with open(os.path.join(out_dir, f"{base}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions to .npz file for consistency with other models
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=hits,
        features=(
            [] if method == "rw" else ["baseline_features"]
        ),  # Placeholder for baseline
        feature_parity=False,  # Baseline doesn't use feature parity
        c_v=1.0 if not calibrate else out.get("c_v", 1.0),
        c_e=1.0 if not calibrate else out.get("c_e", 1.0),
    )

    # Generate diagnostic plots (same as transformer and GARCH)
    from src.utils.eval_tools import plot_var_es_diagnostics

    # Create title for plotting
    title = f"{model_name} (features, {'calibrated' if calibrate else 'raw'})"

    # Generate diagnostic plots
    plot_var_es_diagnostics(
        y_true=y_aligned,
        var_pred=v_eval,
        es_pred=e_eval,
        alpha=alpha,
        title=title,
        out_dir=fig_dir,
        fname_prefix=base,
    )

    return model_name, metrics, (v_eval, e_eval, y_aligned, fz0)


# CLI quick check
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=False, default=None)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument(
        "--method", type=str, default="garch_t", choices=["garch_t", "rm_normal", "rw"]
    )
    p.add_argument("--init_window", type=int, default=2000)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--show_progress", action="store_true")
    args = p.parse_args()

    if args.csv is None:
        np.random.seed(0)
        idx = pd.date_range("2005-01-03", periods=3000, freq="B")
        r = pd.Series(student_t.rvs(df=8, size=3000) * 0.01, index=idx)
        px = np.exp(r.cumsum()) * 100.0
        tmp = "_tmp_baseline_prices.csv"
        pd.DataFrame({"date": idx, "close": px}).to_csv(tmp, index=False)
        args.csv = tmp

    res = pipeline(
        args.csv,
        alpha=args.alpha,
        method=args.method,
        init_window=args.init_window,
        calibrate=args.calibrate,
        show_progress=args.show_progress,
    )
    print(
        json.dumps(
            {
                "model_name": res.get("model_name"),
                "config": res.get("config"),
                "avg_fz0_loss": res["avg_fz0_loss"],
                "hit_rate": res["hit_rate"],
                "n": res.get("n"),
                "kupiec_LR": res.get("kupiec_LR"),
                "kupiec_p": res.get("kupiec_pof_p"),
                "ind_LR": res.get("ind_LR"),
                "ind_p": res.get("chr_ind_p"),
                "cc_LR": res.get("cc_LR"),
                "cc_p": res.get("chr_cc_p"),
            },
            indent=2,
        )
    )
