# run_sim_models.py
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from importlib.machinery import SourceFileLoader
from src.utils.eval_tools import diebold_mariano, plot_var_es_diagnostics


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


def create_artifact_dir(dgp, alpha, seed, calibrate, feature_parity, n_samples):
    """Create artifact directory following the new structure."""
    cal_str = "y" if calibrate else "n"
    feat_str = "parity" if feature_parity else "full"

    # Format alpha as percentage with leading zeros
    alpha_str = f"{int(alpha*1000):03d}"  # e.g., 0.01 -> 010, 0.05 -> 050

    # Format seed with leading zeros
    seed_str = f"{seed:04d}"  # e.g., 1 -> 0001, 42 -> 0042

    artifact_path = os.path.join(
        "artifacts",
        f"dgp={dgp}",
        f"n={n_samples}",
        f"alpha={alpha_str}",
        f"seed={seed_str}",
        f"cal={cal_str}",
        f"feat={feat_str}",
    )

    # Create subdirectories
    os.makedirs(os.path.join(artifact_path, "models", "transformer"), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, "models", "garch"), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, "figures"), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, "tables"), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, "logs"), exist_ok=True)

    return artifact_path


# -----------------------------
# Import DGPs from refactored package
# -----------------------------
from src.dgp import DGPS, SKT_PRESETS


# -----------------------------
# Model runners
# -----------------------------
def run_transformer(
    csv_path, model_path, alpha, tag, calibrate, feature_parity, seed, out_dir, fig_dir
):
    """Run transformer model and return metrics, npz path, and model object."""
    print(f"Loading transformer from: {model_path}")
    transformer_module = SourceFileLoader("transformer", model_path).load_module()

    # Run the pipeline
    model, metrics, (v_eval, e_eval, y_aligned, fz0) = transformer_module.pipeline(
        csv_path=csv_path,
        alpha=alpha,
        feature_parity=feature_parity,
        calibrate=calibrate,
        run_tag=tag,
        out_dir=out_dir,
        fig_dir=fig_dir,
    )

    # Save predictions
    npz_path = os.path.join(out_dir, f"transformer_{tag}.npz")
    np.savez(
        npz_path,
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=(y_aligned <= v_eval).astype(int),
    )

    return metrics, npz_path, model


def run_garch(
    csv_path, model_path, alpha, tag, calibrate, feature_parity, seed, out_dir, fig_dir
):
    """Run GARCH model and return metrics, npz path, and model object."""
    print(f"Loading GARCH from: {model_path}")
    garch_module = SourceFileLoader("garch", model_path).load_module()

    # Run the pipeline
    model, metrics, (v_eval, e_eval, y_aligned, fz0) = garch_module.pipeline(
        csv_path=csv_path,
        alpha=alpha,
        feature_parity=feature_parity,
        calibrate=calibrate,
        run_tag=tag,
        out_dir=out_dir,
        fig_dir=fig_dir,
    )

    # Save predictions
    npz_path = os.path.join(out_dir, f"garch_{tag}.npz")
    np.savez(
        npz_path,
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=(y_aligned <= v_eval).astype(int),
    )

    return metrics, npz_path, model


def run_baseline(
    csv_path, baseline_path, alpha, tag, method, init_window, out_dir, calibrate=False
):
    """Run baseline model and return metrics, npz path, and model object."""
    print(f"Loading baseline from: {baseline_path}")
    baseline_module = SourceFileLoader("baseline", baseline_path).load_module()

    # Run the baseline
    model, metrics, (v_eval, e_eval, y_aligned, fz0) = baseline_module.pipeline(
        csv_path=csv_path,
        alpha=alpha,
        method=method,
        init_window=init_window,
        calibrate=calibrate,  # Pass calibration flag
        run_tag=tag,
        out_dir=out_dir,
    )

    # Save predictions
    npz_path = os.path.join(out_dir, f"baseline_{tag}.npz")
    np.savez(
        npz_path,
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=(y_aligned <= v_eval).astype(int),
    )

    return metrics, npz_path, model


# -----------------------------
# Main execution
# -----------------------------
def _parse_seeds(seeds_str, default_seed):
    """Parse seeds string into list of integers."""
    if not seeds_str.strip():
        return [default_seed]

    seeds = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def main():
    parser = argparse.ArgumentParser(description="Run simulation models")
    parser.add_argument(
        "--dgp", type=str, default="garch11_skt", choices=list(DGPS.keys())
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument(
        "--dgp_params", type=str, default="", help="JSON dict of DGP params"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot", action="store_true", help="Plot simulation diagnostics"
    )

    # Model flags
    parser.add_argument("--no_transformer", action="store_true")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="src/models/transformer_var_es_paper_exact.py",
    )
    parser.add_argument("--no_garch", action="store_true")
    parser.add_argument("--garch_path", type=str, default="src/models/garch.py")

    # Calibration
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply exact-factor calibration to all models",
    )

    # Features
    parser.add_argument(
        "--feature_parity",
        dest="feature_parity",
        action="store_true",
        help="Use parity features (x_cov only).",
    )
    parser.add_argument(
        "--no_feature_parity",
        dest="feature_parity",
        action="store_false",
        help="Use full features (log_ret + x_cov).",
    )
    parser.set_defaults(feature_parity=True)

    # Data reuse
    parser.add_argument(
        "--reuse_csv",
        action="store_true",
        help="If set and per-seed CSV exists, skip simulation and reuse it.",
    )

    # Skew-t specific
    parser.add_argument(
        "--skt_preset", type=str, default="", choices=[""] + list(SKT_PRESETS.keys())
    )
    parser.add_argument("--skt_nu", type=float, default=None)
    parser.add_argument("--skt_lambda", type=float, default=None)

    # Baseline
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument(
        "--baseline_path", type=str, default="src/baselines/baseline_classic_var_es.py"
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

    # Multi-alpha and multi-seed
    parser.add_argument(
        "--alphas", type=str, default="", help="e.g. '0.10,0.05,0.025,0.01'"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma or range, e.g. '1,2,5' or '1-10'. If empty, uses --seed.",
    )

    args = parser.parse_args()
    dgp_extra_params = json.loads(args.dgp_params) if args.dgp_params else {}

    # Parse alpha list
    alpha_list = (
        [float(x) for x in args.alphas.split(",")]
        if args.alphas.strip()
        else [float(args.alpha)]
    )
    seeds = _parse_seeds(args.seeds, args.seed)

    # Aggregate across seeds
    all_seed_tables = []

    for seed in seeds:
        set_seed(seed)

        # Per-seed data directory
        data_dir = os.path.join("data", "interim", f"seed{seed}")
        os.makedirs(data_dir, exist_ok=True)

        # --- Simulate or reuse per-seed CSV ---
        nu, lam = None, None
        tag_extra = ""
        out_csv = os.path.join(
            data_dir,
            f"{args.dgp}{('_'+args.skt_preset) if args.skt_preset else ''}_n{args.n}_seed{seed}.csv",
        )

        if args.reuse_csv and os.path.exists(out_csv):
            print("=" * 60)
            print(f"REUSING existing CSV (seed={seed}): {out_csv}")
            print("=" * 60)
        else:
            if args.dgp == "garch11_skt":
                nu, lam = SKT_PRESETS.get(args.skt_preset, (5, -0.5))
                if args.skt_nu is not None:
                    nu = args.skt_nu
                if args.skt_lambda is not None:
                    lam = args.skt_lambda
                # Update dgp_extra_params with skew-t parameters
                dgp_extra_params_updated = dgp_extra_params.copy()
                dgp_extra_params_updated.update({"nu": nu, "lam": lam})
                y, sigma_t = DGPS[args.dgp](n=args.n, **dgp_extra_params_updated)
                tag_extra = f"_skt_nu{int(nu)}_lam{lam}".replace(".", "p").replace(
                    "-", "m"
                )
            else:
                y, sigma_t = DGPS[args.dgp](n=args.n, **dgp_extra_params)
                tag_extra = ""

            save_csv_from_returns_and_sigma(y, sigma_t, out_csv)

            print("=" * 60)
            print(
                f"SIMULATION: {args.dgp}{(' '+args.skt_preset) if args.skt_preset else ''}  n={args.n}  seed={seed}"
            )
            if args.dgp == "garch11_skt":
                print(f"Skew-t params: nu={nu}, lambda={lam}")
            print(f"CSV written to: {out_csv}")
            print("=" * 60)

        # Optional plot for this seed
        if args.plot:
            try:
                dfp_plot = pd.read_csv(out_csv)
                y_plot = (
                    np.log(dfp_plot["close"] / dfp_plot["close"].shift(1))
                    .dropna()
                    .values
                )
                sigma_plot = (
                    dfp_plot["sigma_t"].values
                    if "sigma_t" in dfp_plot.columns
                    else None
                )

                fig = plt.figure(figsize=(10, 4))
                plt.plot(y_plot, linewidth=0.8, label="returns (y_t)")
                if sigma_plot is not None:
                    ax2 = plt.twinx()
                    ax2.plot(
                        sigma_plot, linewidth=0.8, label="sigma_t (latent volatility)"
                    )
                title = f"{args.dgp}: returns vs latent sigma_t (seed={seed})"
                plt.title(title)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print("[info] Skipping diagnostics plots:", e)

        # --- RUN PER-ALPHA FOR THIS SEED ---
        all_rows = []  # per-seed, all alphas

        for a in alpha_list:
            print("\n" + "=" * 70)
            print(f"RUN @ alpha={a}  (seed={seed})")
            print("=" * 70)

            # Create artifact directory for this configuration
            artifact_dir = create_artifact_dir(
                args.dgp, a, seed, args.calibrate, args.feature_parity, args.n
            )

            # Copy input data for provenance
            shutil.copy2(out_csv, os.path.join(artifact_dir, "data.csv"))

            # Create tag for this run
            tag_a = f"{args.dgp}{tag_extra}_n{args.n}_a{int(a*100)}_seed{seed}{'_cal' if args.calibrate else ''}"
            fp = "parity" if args.feature_parity else "full"
            tag_a = f"{tag_a}_{fp}"

            results = {}
            npz_paths = {}

            # Transformer
            if not args.no_transformer:
                print("\n>>> Running Transformer")
                trans_metrics, trans_npz, _ = run_transformer(
                    out_csv,
                    args.transformer_path,
                    alpha=a,
                    tag=tag_a,
                    calibrate=args.calibrate,
                    feature_parity=args.feature_parity,
                    seed=seed,
                    out_dir=os.path.join(artifact_dir, "models", "transformer"),
                    fig_dir=os.path.join(artifact_dir, "figures"),
                )
                results["Transformer"] = trans_metrics
                npz_paths["Transformer"] = trans_npz

            # GARCH
            if not args.no_garch:
                print("\n>>> Running GARCH")
                garch_metrics, garch_npz, _ = run_garch(
                    out_csv,
                    args.garch_path,
                    alpha=a,
                    tag=tag_a,
                    calibrate=args.calibrate,
                    feature_parity=args.feature_parity,
                    seed=seed,
                    out_dir=os.path.join(artifact_dir, "models", "garch"),
                    fig_dir=os.path.join(artifact_dir, "figures"),
                )
                results["GARCH"] = garch_metrics
                npz_paths["GARCH"] = garch_npz

            # Baseline
            if not args.no_baseline:
                print("\n>>> Running Baseline")
                base_metrics, base_npz, _ = run_baseline(
                    out_csv,
                    args.baseline_path,
                    alpha=a,
                    tag=tag_a,
                    method=args.baseline_method,
                    init_window=args.baseline_init_window,
                    out_dir=os.path.join(artifact_dir, "baseline"),
                    calibrate=args.calibrate,  # Pass calibration flag
                )
                # Use the model name from metrics instead of hardcoding "Baseline"
                baseline_model_name = base_metrics.get("model", "Baseline")
                results[baseline_model_name] = base_metrics
                npz_paths[baseline_model_name] = base_npz

            # --- Create summary table ---
            if results:
                tbl = pd.DataFrame(results).T
                print("\n== RESULTS SUMMARY ==")

                # Create a clean display version without the verbose loss_series
                display_cols = [col for col in tbl.columns if col != "loss_series"]
                tbl_display = tbl[display_cols]

                print(
                    tbl_display.to_string(
                        float_format=lambda x: (
                            f"{x:0.6f}" if isinstance(x, float) else str(x)
                        )
                    )
                )

                # Save summary table - preserve model names as a column
                summary_path = os.path.join(artifact_dir, "tables", "summary.csv")
                # The DataFrame already has a 'model' column from the metrics, so we need to handle this properly
                # First, let's rename the existing 'model' column to avoid conflicts
                if "model" in tbl.columns:
                    tbl = tbl.rename(columns={"model": "model_desc_from_metrics"})
                # Now reset the index to get the model names as a column
                tbl.index.name = "model"
                tbl.reset_index().to_csv(summary_path, index=False)
                print(f"Saved summary to: {summary_path}")

                # Diebold-Mariano tests
                if len(results) > 1:
                    print("\n== DIEBOLD-MARIANO TESTS ==")
                    names = list(results.keys())
                    pair_rows = []

                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            A, B = names[i], names[j]
                            try:
                                A_fz = np.load(npz_paths[A])["fz0"]
                                B_fz = np.load(npz_paths[B])["fz0"]
                                mlen = min(len(A_fz), len(B_fz))
                                stat, p = diebold_mariano(
                                    A_fz[:mlen], B_fz[:mlen], lag=1
                                )
                                pair_rows.append(
                                    {
                                        "alpha": a,
                                        "A": A,
                                        "B": B,
                                        "DM": stat,
                                        "p_value": p,
                                    }
                                )
                                print(f"DM({A} vs {B}) = {stat:0.4f}, p={p:0.4g}")
                            except Exception as e:
                                print(f"[warn] DM failed for {A} vs {B}: {e}")

                    if pair_rows:
                        dm_df = pd.DataFrame(pair_rows)
                        dm_path = os.path.join(artifact_dir, "tables", "dm_pairs.csv")
                        dm_df.to_csv(dm_path, index=False)
                        print(f"Saved DM results to: {dm_path}")

                # Keep per-seed, per-alpha rows to aggregate per seed
                # Handle the model column conflict here too
                if "model" in tbl.columns:
                    tbl = tbl.rename(columns={"model": "model_desc_from_metrics"})
                tbl.index.name = "model"
                tbl_reset = tbl.reset_index()
                if "alpha" not in tbl_reset.columns:
                    tbl_reset.insert(0, "alpha", a)
                else:
                    tbl_reset["alpha"] = a
                if "seed" not in tbl_reset.columns:
                    tbl_reset.insert(1, "seed", seed)
                else:
                    tbl_reset["seed"] = seed
                all_rows.append(tbl_reset)

        # --- Per-seed across-alphas summary ---
        if len(all_rows) > 1:
            agg = pd.concat(all_rows, ignore_index=True)
            print("\n== PER-SEED ACROSS-ALPHAS SUMMARY ==")
            print(
                agg.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:0.6f}" if isinstance(x, float) else str(x)
                    ),
                )
            )

            # Save to results directory
            results_dir = os.path.join("results", "tables")
            os.makedirs(results_dir, exist_ok=True)
            agg.to_csv(
                os.path.join(
                    results_dir, f"summary_{args.dgp}_seed{seed}_all_alphas.csv"
                ),
                index=False,
            )

        all_seed_tables.extend(all_rows)

    # --- Optional: aggregate across seeds too ---
    if len(seeds) > 1 and all_seed_tables:
        agg_all = pd.concat(all_seed_tables, ignore_index=True)
        print("\n== ACROSS-SEEDS (and ALPHAS) SUMMARY ==")
        print(
            agg_all.groupby(["alpha", "model"])[["avg_fz0", "hit_rate"]]
            .mean()
            .reset_index()
            .to_string(
                index=False,
                float_format=lambda x: f"{x:0.6f}" if isinstance(x, float) else str(x),
            )
        )

        # Save to results directory
        results_dir = os.path.join("results", "tables")
        os.makedirs(results_dir, exist_ok=True)
        agg_all.to_csv(
            os.path.join(results_dir, f"summary_{args.dgp}_all_seeds_all_alphas.csv"),
            index=False,
        )

    print("\nDone. See artifacts/ for organized outputs per configuration.")


if __name__ == "__main__":
    main()
