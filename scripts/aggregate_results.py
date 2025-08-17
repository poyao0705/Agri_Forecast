#!/usr/bin/env python3
"""
Aggregate results from artifacts directory and generate summary tables and figures.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_artifacts(artifacts_dir: str = "artifacts") -> pd.DataFrame:
    """Load all results from artifacts directory into a DataFrame."""
    results = []

    for artifact_path in Path(artifacts_dir).rglob("*.json"):
        if "summary" in artifact_path.name or "dm_pairs" in artifact_path.name:
            continue

        # Parse path to extract configuration
        parts = artifact_path.parts
        if len(parts) < 7:
            continue

        try:
            # Extract configuration from path
            dgp = parts[1].replace("dgp=", "")
            n_samples = int(parts[2].replace("n=", ""))
            alpha = (
                int(parts[3].replace("alpha=", "")) / 1000.0
            )  # Convert back from millipercent
            seed = int(parts[4].replace("seed=", ""))
            calibrate = parts[5].replace("cal=", "") == "y"
            features = parts[6].replace("feat=", "")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse path {artifact_path}: {e}")
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
            # Fix baseline model_desc to reflect correct features
            if "model_desc" in metrics:
                # Extract the model name (e.g., "rw-250" or "rw-2000")
                model_name = (
                    metrics.get("model", "rw-250").split("-")[0]
                    + "-"
                    + metrics.get("model", "rw-250").split("-")[1]
                    if "-" in metrics.get("model", "rw-250")
                    else metrics.get("model", "rw-250")
                )
                cal_status = "calibrated" if calibrate else "raw"
                metrics["model_desc"] = f"{model_name} ({features}, {cal_status})"
            else:
                # Create model_desc if missing
                model_name = (
                    metrics.get("model", "rw-250").split("-")[0]
                    + "-"
                    + metrics.get("model", "rw-250").split("-")[1]
                    if "-" in metrics.get("model", "rw-250")
                    else metrics.get("model", "rw-250")
                )
                cal_status = "calibrated" if calibrate else "raw"
                metrics["model_desc"] = f"{model_name} ({features}, {cal_status})"

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


def create_summary_tables(df: pd.DataFrame, output_dir: str = "results/tables"):
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

    summary.to_csv(os.path.join(output_dir, "overall_summary.csv"))

    # 2. Best model per configuration
    best_models = df.loc[
        df.groupby(["dgp", "alpha", "calibrate", "features"])["avg_fz0"].idxmin()
    ]
    best_models.to_csv(
        os.path.join(output_dir, "best_models_per_config.csv"), index=False
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

    model_comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"))

    # 4. Calibration effect analysis
    if "calibrate" in df.columns:
        cal_effect = (
            df.groupby(["model_type", "calibrate"])
            .agg({"avg_fz0": "mean", "hit_rate": "mean", "kupiec_p": "mean"})
            .round(6)
        )

        cal_effect.to_csv(os.path.join(output_dir, "calibration_effect.csv"))

    return summary, best_models, model_comparison


def create_summary_figures(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Create summary figures for the results."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. FZ0 Loss comparison by model
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="model_type", y="avg_fz0", ax=ax)
    ax.set_title("FZ0 Loss Comparison by Model")
    ax.set_ylabel("Average FZ0 Loss")
    ax.set_xlabel("Model Type")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fz0_loss_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Hit rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="model_type", y="hit_rate", ax=ax)
    ax.set_title("Hit Rate Comparison by Model")
    ax.set_ylabel("Hit Rate")
    ax.set_xlabel("Model Type")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "hit_rate_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Model performance by alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # FZ0 by alpha
    for model in df["model_type"].unique():
        model_data = df[df["model_type"] == model]
        ax1.plot(
            model_data["alpha"], model_data["avg_fz0"], "o-", label=model, alpha=0.7
        )
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Average FZ0 Loss")
    ax1.set_title("FZ0 Loss by Alpha")
    ax1.legend()
    ax1.set_xscale("log")

    # Hit rate by alpha
    for model in df["model_type"].unique():
        model_data = df[df["model_type"] == model]
        ax2.plot(
            model_data["alpha"], model_data["hit_rate"], "o-", label=model, alpha=0.7
        )
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Hit Rate")
    ax2.set_title("Hit Rate by Alpha")
    ax2.legend()
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "performance_by_alpha.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Calibration effect (if available)
    if "calibrate" in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # FZ0 with/without calibration
        sns.boxplot(data=df, x="model_type", y="avg_fz0", hue="calibrate", ax=ax1)
        ax1.set_title("FZ0 Loss: Calibrated vs Raw")
        ax1.set_ylabel("Average FZ0 Loss")

        # Hit rate with/without calibration
        sns.boxplot(data=df, x="model_type", y="hit_rate", hue="calibrate", ax=ax2)
        ax2.set_title("Hit Rate: Calibrated vs Raw")
        ax2.set_ylabel("Hit Rate")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "calibration_effect.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def load_dm_results(artifacts_dir: str = "artifacts") -> pd.DataFrame:
    """Load Diebold-Mariano test results."""
    dm_results = []

    for dm_file in Path(artifacts_dir).rglob("dm_pairs.csv"):
        # Parse path to extract configuration
        parts = dm_file.parts
        if len(parts) < 7:
            continue

        # Extract configuration from path
        dgp = parts[1].replace("dgp=", "")
        n_samples = int(parts[2].replace("n=", ""))
        alpha = int(parts[3].replace("alpha=", "")) / 1000.0
        seed = int(parts[4].replace("seed=", ""))
        calibrate = parts[5].replace("cal=", "") == "y"
        features = parts[6].replace("feat=", "")

        # Load DM results
        dm_df = pd.read_csv(dm_file)
        dm_df["dgp"] = dgp
        dm_df["n_samples"] = n_samples
        dm_df["alpha"] = alpha
        dm_df["seed"] = seed
        dm_df["calibrate"] = calibrate
        dm_df["features"] = features

        dm_results.append(dm_df)

    if dm_results:
        return pd.concat(dm_results, ignore_index=True)
    else:
        return pd.DataFrame()


def create_dm_summary(dm_df: pd.DataFrame, output_dir: str = "results/tables"):
    """Create summary of Diebold-Mariano test results."""
    if dm_df.empty:
        print("No DM results found.")
        return

    # Overall DM summary
    dm_summary = (
        dm_df.groupby(["A", "B"])
        .agg({"DM": ["mean", "std", "count"], "p_value": ["mean", "std"]})
        .round(6)
    )

    dm_summary.to_csv(os.path.join(output_dir, "dm_test_summary.csv"))

    # Significant differences
    significant = dm_df[dm_df["p_value"] < 0.05].copy()
    if not significant.empty:
        significant.to_csv(
            os.path.join(output_dir, "significant_dm_tests.csv"), index=False
        )

    return dm_summary


def main():
    """Main function to aggregate all results."""
    print("Loading artifacts...")
    df = load_artifacts()

    if df.empty:
        print("No artifacts found. Make sure to run experiments first.")
        return

    print(f"Loaded {len(df)} results from artifacts.")

    # Create summary tables
    print("Creating summary tables...")
    summary, best_models, model_comparison = create_summary_tables(df)

    # Create summary figures
    print("Creating summary figures...")
    create_summary_figures(df)

    # Load and process DM results
    print("Processing DM test results...")
    dm_df = load_dm_results()
    if not dm_df.empty:
        dm_summary = create_dm_summary(dm_df)

    # Save the full dataset
    output_file = os.path.join(
        "results/tables", "summary_garch11_skt_all_seeds_all_alphas.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"Saved full dataset to {output_file}")

    print("Aggregation complete!")
    print(f"Results saved to results/tables/ and results/figures/")

    # Print quick summary
    print("\nQuick Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Models: {df['model_type'].unique()}")
    print(f"DGPs: {df['dgp'].unique()}")
    print(f"Alphas: {sorted(df['alpha'].unique())}")

    if "calibrate" in df.columns:
        print(f"Calibration: {df['calibrate'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
