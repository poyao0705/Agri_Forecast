# run_sim_models.py
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader
from eval_tools import diebold_mariano, plot_var_es_diagnostics


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)


def returns_to_prices(log_ret, p0=100.0):
    prices = [p0]
    for r in log_ret:
        prices.append(prices[-1] * float(np.exp(r)))
    return np.array(prices[1:])


def save_csv_from_returns_and_sigma(log_ret, sigma_t, out_path, p0=100.0):
    close = returns_to_prices(log_ret, p0=p0)
    df = pd.DataFrame({"close": close})
    if sigma_t is not None:
        df["sigma_t"] = np.asarray(sigma_t, float)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


# -----------------------------
# DGPs (return (y, sigma_t))
# -----------------------------
def simulate_iid_gaussian(n=6000, mu=0.0, sigma=0.01):
    eps = np.random.randn(n)
    y = mu + sigma * eps
    sigma_t = np.full(n, sigma)
    return y, sigma_t


def simulate_garch11_t(
    n=6000, alpha=0.05, beta=0.94, nu=8, mu=0.0, sigma2_target=1e-4, burn=2000
):
    from scipy.stats import t

    omega = (1.0 - alpha - beta) * sigma2_target
    z = t.rvs(df=nu, size=n + burn) / np.sqrt(nu / (nu - 2))
    h = np.empty(n + burn)
    y = np.empty(n + burn)
    h[0] = sigma2_target
    for t_ in range(n + burn):
        y[t_] = mu + np.sqrt(max(h[t_], 1e-12)) * z[t_]
        if t_ + 1 < n + burn:
            h[t_ + 1] = omega + alpha * (y[t_] - mu) ** 2 + beta * h[t_]
    y = y[burn:]
    h = h[burn:]
    sigma_t = np.sqrt(h)
    return y, sigma_t


def simulate_sv(n=6000, mu=-1.0, phi=0.98, sigma_eta=0.2):
    log_sig = np.zeros(n)
    log_sig[0] = mu
    eta = np.random.randn(n) * sigma_eta
    for t_ in range(1, n):
        log_sig[t_] = mu + phi * (log_sig[t_ - 1] - mu) + eta[t_]
    sigma_t = np.exp(log_sig)
    y = sigma_t * np.random.randn(n)
    return y, sigma_t


def simulate_srnn_like(n=6000, ah=0.98, ax=0.2, w=0.5, b=-3.0):
    h_state = np.zeros(n)
    y = np.zeros(n)
    sigma_t = np.zeros(n)
    for t_ in range(1, n):
        h_state[t_] = ah * h_state[t_ - 1] + ax * y[t_ - 1] ** 2
        sigma_t[t_] = np.log(1 + np.exp(w * h_state[t_] + b))  # softplus
        y[t_] = sigma_t[t_] * np.random.randn()
    sigma_t[0] = sigma_t[1]
    return y, sigma_t


def simulate_garch11_skt(
    n=6000,
    alpha=0.05,  # GARCH alpha (ARCH)
    beta=0.94,  # GARCH beta
    nu=8,  # skew-t dof
    lam=-0.5,  # skewness in (-1, 1)
    mu=0.0,
    sigma2_target=1e-4,
    burn=2000,
    seed=None,
):
    """
    GARCH(1,1) with Hansen-style skew-t innovations (mean 0, var 1).
    Uses 'skewstudent' if available; otherwise falls back to a two-piece skew-t
    standardized to mean 0 and variance 1.
    """
    if seed is not None:
        np.random.seed(seed)
    if not (0 <= alpha and 0 <= beta and alpha + beta < 1):
        raise ValueError(
            "Require alpha>=0, beta>=0, and alpha+beta<1 for stationarity."
        )
    if not (-0.999 < lam < 0.999):
        raise ValueError("lam must be in (-1,1).")

    # 1) Innovations eps ~ skew-t( nu, lam ), standardized to mean 0, var 1
    try:
        from skewstudent.skewstudent import SkewStudent

        sk = SkewStudent(eta=nu, lam=lam)  # already mean 0, var 1
        eps = sk.rvs(size=n + burn)
    except Exception:
        # Fallback: two-piece transform of symmetric t, then standardize
        from scipy.stats import t as student_t

        z = student_t.rvs(df=nu, size=n + burn) / np.sqrt(nu / (nu - 2))  # var 1
        scale = np.where(z >= 0.0, 1.0 + lam, 1.0 - lam)
        eps = z * scale
        # standardize to mean 0, var 1
        eps = (eps - eps.mean()) / (eps.std(ddof=0) + 1e-12)

    # 2) GARCH(1,1) with target unconditional variance
    omega = (1.0 - alpha - beta) * sigma2_target
    h = np.empty(n + burn)
    y = np.empty(n + burn)
    h[0] = sigma2_target
    for t_ in range(n + burn):
        y[t_] = mu + np.sqrt(max(h[t_], 1e-12)) * eps[t_]
        if t_ + 1 < n + burn:
            h[t_ + 1] = omega + alpha * (y[t_] - mu) ** 2 + beta * h[t_]

    return y[burn:], np.sqrt(h[burn:])


DGPS = {
    "iid_gauss": simulate_iid_gaussian,
    "garch11_t": simulate_garch11_t,
    "garch11_skt": simulate_garch11_skt,
    "sv": simulate_sv,
    "srnn_like": simulate_srnn_like,
}

SKT_PRESETS = {
    "heavy_left_3": (3, -0.8),
    "left_5": (5, -0.5),
    "left_10": (10, -1.0),
}


# -----------------------------
# Run models
# -----------------------------
def run_transformer(csv_path, transformer_path, alpha=0.01, tag="", calibrate=False):
    mod = SourceFileLoader(
        "transformer_var_es_paper_exact", transformer_path
    ).load_module()
    model, metrics, (v_eval, e_eval, y_aligned, fz0) = mod.pipeline(
        csv_path=csv_path,
        alpha=alpha,
        feature_parity=True,
        calibrate=calibrate,
        run_tag=tag,
        out_dir="saved_models",
        fig_dir="figures",
    )
    # don't attach preds (it isn't a DF here)
    metrics["n"] = int(len(y_aligned))
    base = f"transformer_{tag+'_ ' if tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
        " ", ""
    )
    return (
        metrics,
        os.path.join("saved_models", f"{base}.npz"),
        os.path.join("saved_models", f"{base}.json"),
    )


def run_srnn(csv_path, srnn_path, alpha=0.01, tag="", calibrate=False):
    mod = SourceFileLoader("srnn_ve1_paper_exact", srnn_path).load_module()
    model, metrics, (v_eval, e_eval, y_aligned, fz0) = mod.pipeline(
        csv_path=csv_path,
        alpha=alpha,
        calibrate=calibrate,
        run_tag=tag,
        out_dir="saved_models",
        fig_dir="figures",
    )
    metrics["n"] = int(len(y_aligned))
    base = (
        f"srnn_{tag+'_ ' if tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
            " ", ""
        )
    )
    return (
        metrics,
        os.path.join("saved_models", f"{base}.npz"),
        os.path.join("saved_models", f"{base}.json"),
    )


def run_baseline(
    csv_path,
    baseline_path="baseline_classic_var_es.py",
    alpha=0.01,
    tag="",
    calibrate=False,
    method="garch_t",
    init_window=2000,
):
    mod = SourceFileLoader("baseline_classic_var_es", baseline_path).load_module()
    res = mod.pipeline(
        csv_path=csv_path,
        alpha=alpha,
        method=method,
        init_window=init_window,
        calibrate=calibrate,
        tag=f"Baseline[{method}]{('_'+tag) if tag else ''}",
    )
    os.makedirs("saved_models", exist_ok=True)
    base_name = (
        f"baseline_{method}_{tag}_{'calibrated' if calibrate else 'raw'}".replace(
            " ", ""
        ).replace("/", "-")
    )
    npz_path = os.path.join("saved_models", f"{base_name}.npz")
    preds = res["preds"]
    fz0 = np.asarray(res["loss_series"].values, float)
    np.savez(
        npz_path,
        dates=np.asarray(preds.index.astype("datetime64[ns]").astype("int64")),
        var=np.asarray(preds["VaR"].values, float),
        es=np.asarray(preds["ES"].values, float),
        fz0=fz0,
    )
    json_path = os.path.join("saved_models", f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "model_name": res.get("model_name"),
                "config": res.get("config"),
                "avg_fz0_loss": res["avg_fz0_loss"],
                "hit_rate": res["hit_rate"],
                "kupiec_LR": res.get("kupiec_LR"),
                "kupiec_pof_p": res.get("kupiec_pof_p"),
                "ind_LR": res.get("ind_LR"),
                "chr_ind_p": res.get("chr_ind_p"),
                "cc_LR": res.get("cc_LR"),
                "chr_cc_p": res.get("chr_cc_p"),
                "n": int(len(preds)),
            },
            f,
            indent=2,
        )
    metrics = {
        "model_name": res.get("model_name", f"Baseline[{method}]"),
        "avg_fz0_loss": res["avg_fz0_loss"],
        "hit_rate": res["hit_rate"],
        "kupiec_pof_p": res.get("kupiec_pof_p"),
        "chr_ind_p": res.get("chr_ind_p"),
        "chr_cc_p": res.get("chr_cc_p"),
        "kupiec_LR": res.get("kupiec_LR"),
        "ind_LR": res.get("ind_LR"),
        "cc_LR": res.get("cc_LR"),
        "n": int(len(preds)),
        "preds": preds,
    }
    return metrics, npz_path, json_path


# -----------------------------
# Main experiment
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dgp", type=str, default="garch11_t", choices=list(DGPS.keys())
    )
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--out_csv", type=str, default="data/simulated_from_paper.csv")
    parser.add_argument(
        "--transformer_path", type=str, default="transformer_var_es_paper_exact.py"
    )
    parser.add_argument("--srnn_path", type=str, default="srnn_ve1_paper_exact.py")
    parser.add_argument("--dgp_params", type=str, default="")
    parser.add_argument("--no_srnn", action="store_true")
    parser.add_argument("--no_transformer", action="store_true")
    parser.add_argument(
        "--plot", action="store_true", help="Plot returns vs latent sigma_t"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply exact-factor calibration to all models",
    )

    # Skew-t options
    parser.add_argument(
        "--skt_preset", type=str, default="", choices=[""] + list(SKT_PRESETS.keys())
    )
    parser.add_argument("--skt_nu", type=float, default=None)
    parser.add_argument("--skt_lambda", type=float, default=None)

    # Baseline options
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument(
        "--baseline_path", type=str, default="baseline_classic_var_es.py"
    )
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="garch_t",
        choices=["garch_t", "rm_normal", "rw"],
    )
    parser.add_argument(
        "--baseline_init_window",
        type=int,
        default=2000,
        help="Expanding start for garch_t/rm_normal or window length for rw",
    )

    # Multi-alpha runs
    parser.add_argument(
        "--alphas", type=str, default="", help="e.g. '0.10,0.05,0.025,0.01'"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    params = json.loads(args.dgp_params) if args.dgp_params else {}

    # 1) Simulate returns + latent sigma
    if args.dgp == "garch11_skt":
        nu, lam = SKT_PRESETS.get(args.skt_preset, (5, -0.5))
        if args.skt_nu is not None:
            nu = args.skt_nu
        if args.skt_lambda is not None:
            lam = args.skt_lambda
        y, sigma_t = simulate_garch11_skt(
            n=args.n,
            nu=nu,
            lam=lam,
            **{k: v for k, v in params.items() if k not in ("nu", "lam")},
        )
        tag_extra = f"_skt_nu{int(nu)}_lam{lam}".replace(".", "p").replace("-", "m")
    else:
        y, sigma_t = DGPS[args.dgp](n=args.n, **params)
        tag_extra = ""

    # 2) Save CSV (close + sigma_t for inspection)
    save_csv_from_returns_and_sigma(y, sigma_t, args.out_csv)

    print("=" * 60)
    print(
        f"SIMULATION: {args.dgp}{(' '+args.skt_preset) if args.skt_preset else ''}  n={args.n}  alpha={args.alpha}  seed={args.seed}"
    )
    if args.dgp == "garch11_skt":
        print(f"Skew-t params: nu={nu}, lambda={lam}")
    print(f"CSV written to: {args.out_csv}")
    print("=" * 60)

    # 2b) Optional plot
    if args.plot:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(y, linewidth=0.8, label="returns (y_t)")
        plt.twinx()
        plt.plot(sigma_t, linewidth=0.8, label="sigma_t (latent volatility)")
        plt.title(f"{args.dgp}: returns vs latent sigma_t")
        plt.tight_layout()
        plt.show()

    # Parse alphas
    alpha_list = (
        [float(x) for x in args.alphas.split(",")]
        if args.alphas.strip()
        else [float(args.alpha)]
    )
    all_rows = []
    os.makedirs("results", exist_ok=True)

    for a in alpha_list:
        print("\n" + "=" * 70)
        print(f"RUN @ alpha={a}")
        print("=" * 70)
        tag_a = f"{args.dgp}{tag_extra}_n{args.n}_a{int(a*100)}_seed{args.seed}{'_cal' if args.calibrate else ''}"

        results = {}
        npz_paths = {}

        # Transformer
        if not args.no_transformer:
            print("\n>>> Running Transformer")
            trans_metrics, trans_npz, _ = run_transformer(
                args.out_csv,
                args.transformer_path,
                alpha=a,
                tag=tag_a,
                calibrate=args.calibrate,
            )
            results["Transformer"] = trans_metrics
            npz_paths["Transformer"] = trans_npz

        # SRNN
        if not args.no_srnn:
            print("\n>>> Running SRNN-VE-1")
            srnn_metrics, srnn_npz, _ = run_srnn(
                args.out_csv,
                args.srnn_path,
                alpha=a,
                tag=tag_a,
                calibrate=args.calibrate,
            )
            results["SRNN-VE-1"] = srnn_metrics
            npz_paths["SRNN-VE-1"] = srnn_npz

        # Baseline
        if not args.no_baseline:
            print(f"\n>>> Running Baseline [{args.baseline_method}]")
            base_metrics, base_npz, _ = run_baseline(
                args.out_csv,
                baseline_path=args.baseline_path,
                alpha=a,
                tag=tag_a,
                calibrate=args.calibrate,
                method=args.baseline_method,
                init_window=args.baseline_init_window,
            )
            results[f"Baseline[{args.baseline_method}]"] = base_metrics
            npz_paths[f"Baseline[{args.baseline_method}]"] = base_npz

        # --- Paper-style summary table ---
        rows = []
        for name, m in results.items():
            avg_fz0 = m.get("avg_fz0_loss", m.get("avg_fz0"))
            rows.append(
                {
                    "model": name,
                    "avg_fz0": avg_fz0,
                    "hit_rate": m.get("hit_rate"),
                    "kupiec_p": m.get("kupiec_pof_p") or m.get("kupiec_p"),
                    "ind_p": m.get("chr_ind_p") or m.get("ind_p"),
                    "cc_p": m.get("chr_cc_p") or m.get("cc_p"),
                    "kupiec_LR": m.get("kupiec_LR"),
                    "ind_LR": m.get("ind_LR"),
                    "cc_LR": m.get("cc_LR"),
                    "n": m.get("n"),
                }
            )
        tbl = pd.DataFrame(rows).set_index("model").sort_values("avg_fz0")
        print("\n== SUMMARY (lower avg_fz0 is better) ==")
        print(
            tbl.to_string(
                float_format=lambda x: f"{x:0.6f}" if isinstance(x, float) else str(x)
            )
        )

        # Save summary
        basefile = f"results/summary_{args.dgp}{tag_extra}_n{args.n}_a{int(a*100)}_seed{args.seed}{'_cal' if args.calibrate else ''}"
        tbl.to_csv(basefile + ".csv")
        try:
            tbl.to_latex(basefile + ".tex", float_format="%.6f")
        except Exception:
            pass  # latex optional

        # --- Diebold–Mariano tests (pairwise on FZ0) ---
        pair_rows = []
        names = list(npz_paths.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = names[i], names[j]
                try:
                    A_fz = np.load(npz_paths[A])["fz0"]
                    B_fz = np.load(npz_paths[B])["fz0"]
                    m = min(len(A_fz), len(B_fz))
                    stat, p = diebold_mariano(A_fz[:m], B_fz[:m], lag=1)
                    pair_rows.append(
                        {"alpha": a, "A": A, "B": B, "DM": stat, "p_value": p}
                    )
                    print(f"DM({A} vs {B}) = {stat:0.4f}, p={p:0.4g}")
                except Exception as e:
                    print(f"[warn] DM failed for {A} vs {B}: {e}")
        if pair_rows:
            dm_df = pd.DataFrame(pair_rows)
            dm_df.to_csv(basefile + "_dm.csv", index=False)

        # --- Diagnostics plots (2×2) ---
        try:
            dfp = pd.read_csv(args.out_csv)
            if "date" in dfp.columns:
                dfp["date"] = pd.to_datetime(dfp["date"])
                dfp = dfp.set_index("date")
            if "close" not in dfp.columns:
                raise ValueError(
                    "CSV missing 'close' column; cannot compute returns for plots."
                )
            r = np.log(dfp["close"] / dfp["close"].shift(1)).dropna()

            out_dir = os.path.join("figures", f"{args.dgp}_a{int(a*100)}")
            os.makedirs(out_dir, exist_ok=True)

            for name, m in results.items():
                preds = m.get("preds")
                if preds is None or len(preds) == 0:
                    continue
                y_true = r.reindex(preds.index).astype(float).values
                var_pred = preds["VaR"].astype(float).values
                es_pred = preds["ES"].astype(float).values
                fname_prefix = f"{name.replace('/','-')}_{tag_a}".replace(" ", "")
                png = plot_var_es_diagnostics(
                    y_true, var_pred, es_pred, a, name, out_dir, fname_prefix
                )
                print("Saved diagnostics:", png)
        except Exception as e:
            print("[info] Skipping diagnostics plots:", e)

        # Keep for multi-alpha aggregation
        tbl_reset = tbl.reset_index()
        tbl_reset.insert(0, "alpha", a)
        all_rows.append(tbl_reset)

    # --- Aggregate across alphas (if multiple provided) ---
    if len(all_rows) > 1:
        agg = pd.concat(all_rows, ignore_index=True)
        print("\n== ACROSS-ALPHAS SUMMARY ==")
        print(
            agg.to_string(
                index=False,
                float_format=lambda x: f"{x:0.6f}" if isinstance(x, float) else str(x),
            )
        )
        agg.to_csv(
            f"results/summary_{args.dgp}{tag_extra}_n{args.n}_seed{args.seed}_ALL_ALPHAS.csv",
            index=False,
        )

    print("\nDone. See results/*.csv, results/*_dm.csv and figures/* for plots.")


if __name__ == "__main__":
    main()
