#!/usr/bin/env python3
"""
Generate summary tables from artifacts directory with flexible filtering options.
"""

import os
import sys
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_artifacts(
    artifacts_dir: str = "artifacts", filters: dict = None
) -> pd.DataFrame:
    """Load results from artifacts directory into a DataFrame with optional filtering."""
    results = []

    for artifact_path in Path(artifacts_dir).rglob("*.json"):
        if "summary" in artifact_path.name or "dm_pairs" in artifact_path.name:
            continue

        # Parse path to extract configuration
        parts = artifact_path.parts
        if len(parts) < 7:
            continue

        # Extract configuration from path
        dgp = parts[1].replace("dgp=", "")
        n_samples = int(parts[2].replace("n=", ""))
        alpha = (
            int(parts[3].replace("alpha=", "")) / 1000.0
        )  # Convert back from millipercent
        seed = int(parts[4].replace("seed=", ""))
        calibrate = parts[5].replace("cal=", "") == "y"
        features = parts[6].replace("feat=", "")

        # Apply filters if specified
        if filters:
            if "dgp" in filters and dgp != filters["dgp"]:
                continue
            if "features" in filters and features != filters["features"]:
                continue
            if "calibrate" in filters and calibrate != filters["calibrate"]:
                continue
            if "alpha" in filters and alpha != filters["alpha"]:
                continue
            if "seed" in filters and seed != filters["seed"]:
                continue

        # Load metrics
        with open(artifact_path, "r") as f:
            metrics = json.load(f)

        # For baseline models, set feature_parity and features based on path
        if artifact_path.parent.name == "baseline":
            metrics["feature_parity"] = features == "parity"
            metrics["features"] = (
                ["x_cov"]
                if features == "parity"
                else ["ewma94", "ewma97", "x_cov", "neg_xcov", "neg_ret"]
            )

        # Construct model_desc for transformer and SRNN models if missing
        if (
            artifact_path.parent.name in ["transformer", "srnn"]
            and "model_desc" not in metrics
        ):
            cal_status = "calibrated" if calibrate else "raw"
            model_name = (
                "Transformer"
                if artifact_path.parent.name == "transformer"
                else "SRNN-VE-1"
            )
            metrics["model_desc"] = f"{model_name} ({features}, {cal_status})"

        # Add configuration info
        metrics.update(
            {
                "dgp": dgp,
                "n_samples": n_samples,
                "alpha": alpha,
                "seed": seed,
                "calibrate": calibrate,
                "features": features,
                "model_type": (
                    artifact_path.parent.name
                    if artifact_path.parent.name in ["transformer", "srnn", "baseline"]
                    else "unknown"
                ),
                "artifact_path": str(artifact_path),
            }
        )

        results.append(metrics)

    return pd.DataFrame(results)


def create_summary_tables(
    df: pd.DataFrame, output_dir: str = "results/tables", prefix: str = ""
):
    """Create summary tables for different aggregations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Overall summary by model and configuration
    summary = (
        df.groupby(["model_type", "dgp", "alpha", "calibrate", "features"])
        .agg(
            {
                "avg_fz0": ["mean", "std", "count"],
                "hit_rate": ["mean", "std"],
                "kupiec_p": ["mean", "std"],
                "ind_p": ["mean", "std"],
                "cc_p": ["mean", "std"],
            }
        )
        .round(6)
    )

    # Fix the MultiIndex columns by flattening them
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]

    summary.to_csv(os.path.join(output_dir, f"{prefix}overall_summary.csv"))

    # 2. Best model per configuration
    best_models = df.loc[
        df.groupby(["dgp", "alpha", "calibrate", "features"])["avg_fz0"].idxmin()
    ]
    best_models.to_csv(
        os.path.join(output_dir, f"{prefix}best_models_per_config.csv"), index=False
    )

    # 3. Model comparison across all configurations
    model_comparison = (
        df.groupby("model_type")
        .agg(
            {
                "avg_fz0": ["mean", "std", "count"],
                "hit_rate": ["mean", "std"],
                "kupiec_p": ["mean", "std"],
                "ind_p": ["mean", "std"],
                "cc_p": ["mean", "std"],
            }
        )
        .round(6)
    )

    # Fix the MultiIndex columns by flattening them
    model_comparison.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in model_comparison.columns
    ]

    model_comparison.to_csv(os.path.join(output_dir, f"{prefix}model_comparison.csv"))

    # 4. Calibration effect analysis
    if "calibrate" in df.columns:
        cal_effect = (
            df.groupby(["model_type", "calibrate"])
            .agg({"avg_fz0": "mean", "hit_rate": "mean", "kupiec_p": "mean"})
            .round(6)
        )

        cal_effect.to_csv(os.path.join(output_dir, f"{prefix}calibration_effect.csv"))

    return summary, best_models, model_comparison


def create_summary_figures(
    df: pd.DataFrame, output_dir: str = "results/figures", prefix: str = ""
):
    """Create summary figures for the results."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Create a descriptive title based on the data
    title_suffix = ""
    if len(df["features"].unique()) == 1:
        title_suffix = f" ({df['features'].iloc[0]} features)"
    if len(df["alpha"].unique()) == 1:
        title_suffix += f" (α={df['alpha'].iloc[0]})"
    if len(df["calibrate"].unique()) == 1:
        cal_status = "calibrated" if df["calibrate"].iloc[0] else "raw"
        title_suffix += f" ({cal_status})"

    # 1. FZ0 Loss comparison by model
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="model_type", y="avg_fz0", ax=ax)
    ax.set_title(f"FZ0 Loss Comparison by Model{title_suffix}")
    ax.set_ylabel("Average FZ0 Loss")
    ax.set_xlabel("Model Type")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix}fz0_loss_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Hit rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="model_type", y="hit_rate", ax=ax)
    ax.set_title(f"Hit Rate Comparison by Model{title_suffix}")
    ax.set_ylabel("Hit Rate")
    ax.set_xlabel("Model Type")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix}hit_rate_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Model performance by alpha (only if multiple alphas)
    if len(df["alpha"].unique()) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # FZ0 by alpha
        for model in df["model_type"].unique():
            model_data = df[df["model_type"] == model]
            ax1.plot(
                model_data["alpha"], model_data["avg_fz0"], "o-", label=model, alpha=0.7
            )
        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Average FZ0 Loss")
        ax1.set_title(f"FZ0 Loss by Alpha{title_suffix}")
        ax1.legend()
        ax1.set_xscale("log")

        # Hit rate by alpha
        for model in df["model_type"].unique():
            model_data = df[df["model_type"] == model]
            ax2.plot(
                model_data["alpha"],
                model_data["hit_rate"],
                "o-",
                label=model,
                alpha=0.7,
            )
        ax2.set_xlabel("Alpha")
        ax2.set_ylabel("Hit Rate")
        ax2.set_title(f"Hit Rate by Alpha{title_suffix}")
        ax2.legend()
        ax2.set_xscale("log")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}performance_by_alpha.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Feature comparison (if multiple feature types)
    if len(df["features"].unique()) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # FZ0 by features
        sns.boxplot(data=df, x="model_type", y="avg_fz0", hue="features", ax=ax1)
        ax1.set_title(f"FZ0 Loss: Full vs Parity Features{title_suffix}")
        ax1.set_ylabel("Average FZ0 Loss")

        # Hit rate by features
        sns.boxplot(data=df, x="model_type", y="hit_rate", hue="features", ax=ax2)
        ax2.set_title(f"Hit Rate: Full vs Parity Features{title_suffix}")
        ax2.set_ylabel("Hit Rate")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}feature_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 5. Calibration effect (if available and multiple calibration types)
    if "calibrate" in df.columns and len(df["calibrate"].unique()) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # FZ0 with/without calibration
        sns.boxplot(data=df, x="model_type", y="avg_fz0", hue="calibrate", ax=ax1)
        ax1.set_title(f"FZ0 Loss: Calibrated vs Raw{title_suffix}")
        ax1.set_ylabel("Average FZ0 Loss")

        # Hit rate with/without calibration
        sns.boxplot(data=df, x="model_type", y="hit_rate", hue="calibrate", ax=ax2)
        ax2.set_title(f"Hit Rate: Calibrated vs Raw{title_suffix}")
        ax2.set_ylabel("Hit Rate")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}calibration_effect.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Main function to generate summaries with flexible filtering."""
    parser = argparse.ArgumentParser(
        description="Generate summary tables from artifacts"
    )
    parser.add_argument("--dgp", type=str, help="Filter by DGP (e.g., garch11_skt)")
    parser.add_argument(
        "--features",
        type=str,
        choices=["full", "parity"],
        help="Filter by feature type",
    )
    parser.add_argument(
        "--calibrate", type=str, choices=["true", "false"], help="Filter by calibration"
    )
    parser.add_argument("--alpha", type=float, help="Filter by alpha value")
    parser.add_argument("--seed", type=int, help="Filter by seed")
    parser.add_argument(
        "--output-prefix", type=str, default="", help="Prefix for output files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/tables", help="Output directory"
    )

    args = parser.parse_args()

    # Build filters dictionary
    filters = {}
    if args.dgp:
        filters["dgp"] = args.dgp
    if args.features:
        filters["features"] = args.features
    if args.calibrate:
        filters["calibrate"] = args.calibrate.lower() == "true"
    if args.alpha:
        filters["alpha"] = args.alpha
    if args.seed:
        filters["seed"] = args.seed

    print("Loading artifacts...")
    df = load_artifacts(filters=filters)

    if df.empty:
        print("No artifacts found matching the specified filters.")
        return

    print(f"Loaded {len(df)} results from artifacts.")

    # Create summary tables
    print("Creating summary tables...")
    summary, best_models, model_comparison = create_summary_tables(
        df, args.output_dir, args.output_prefix
    )

    # Create summary figures
    print("Creating summary figures...")
    create_summary_figures(df, "results/figures", args.output_prefix)

    print("Summary generation complete!")
    print(f"Results saved to {args.output_dir}/ and results/figures/")

    # Print quick summary
    print("\nQuick Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Models: {df['model_type'].unique()}")
    print(f"DGPs: {df['dgp'].unique()}")
    print(f"Alphas: {sorted(df['alpha'].unique())}")
    print(f"Features: {df['features'].unique()}")

    # Save the full dataset
    output_file = os.path.join(
        args.output_dir, f"{args.output_prefix}summary_full_dataset.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"Saved full dataset to {output_file}")


if __name__ == "__main__":
    main()
