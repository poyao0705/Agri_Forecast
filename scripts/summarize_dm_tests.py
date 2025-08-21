#!/usr/bin/env python3
"""
Summarize Diebold-Mariano test results for n=5000 across all configurations.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_dm_results(base_path="artifacts/dgp=garch11_skt/n=5000"):
    """Load all DM test results from the n=5000 directory."""
    results = []

    # Find all dm_pairs.csv files
    dm_files = list(Path(base_path).rglob("dm_pairs.csv"))

    for file_path in dm_files:
        try:
            # Parse path to extract configuration
            parts = file_path.parts
            alpha_part = [p for p in parts if p.startswith("alpha=")][0]
            seed_part = [p for p in parts if p.startswith("seed=")][0]
            cal_part = [p for p in parts if p.startswith("cal=")][0]
            feat_part = [p for p in parts if p.startswith("feat=")][0]

            # Extract values
            alpha = (
                float(alpha_part.replace("alpha=", "")) / 1000
            )  # Convert from 010 to 0.01
            seed = int(seed_part.replace("seed=", ""))
            calibrate = cal_part.replace("cal=", "") == "y"
            feature_parity = feat_part.replace("feat=", "") == "parity"

            # Load DM results
            df = pd.read_csv(file_path)

            # Add configuration info
            df["alpha"] = alpha
            df["seed"] = seed
            df["calibrate"] = calibrate
            df["feature_parity"] = feature_parity

            results.append(df)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def summarize_dm_results(df):
    """Create comprehensive summary of DM test results."""

    # Basic statistics by model pair
    summary = (
        df.groupby(["alpha", "A", "B"])
        .agg(
            {
                "DM": ["count", "mean", "std", "min", "max"],
                "p_value": ["mean", "std", "min", "max"],
            }
        )
        .round(4)
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    # Add significance analysis
    df["significant_5pct"] = df["p_value"] < 0.05
    df["significant_1pct"] = df["p_value"] < 0.01

    significance_summary = (
        df.groupby(["alpha", "A", "B"])
        .agg({"significant_5pct": "sum", "significant_1pct": "sum", "DM": "count"})
        .reset_index()
    )

    significance_summary["pct_significant_5pct"] = (
        significance_summary["significant_5pct"] / significance_summary["DM"] * 100
    ).round(1)
    significance_summary["pct_significant_1pct"] = (
        significance_summary["significant_1pct"] / significance_summary["DM"] * 100
    ).round(1)

    # Merge summaries
    final_summary = summary.merge(significance_summary, on=["alpha", "A", "B"])

    return final_summary


def create_model_comparison_table(df):
    """Create a table showing which model wins in each comparison."""

    def determine_winner(row):
        if row["p_value"] < 0.05:  # Significant difference
            if row["DM"] > 0:
                return f"{row['A']} wins"
            else:
                return f"{row['B']} wins"
        else:
            return "No significant difference"

    df["winner"] = df.apply(determine_winner, axis=1)

    # Count wins by model
    wins_by_model = df.groupby(["alpha", "winner"]).size().unstack(fill_value=0)

    return wins_by_model


def main():
    parser = argparse.ArgumentParser(description="Summarize DM test results for n=5000")
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for summaries"
    )
    parser.add_argument(
        "--base-path",
        default="artifacts/dgp=garch11_skt/n=5000",
        help="Base path to search for DM results",
    )

    args = parser.parse_args()

    print("Loading DM test results...")
    df = load_dm_results(args.base_path)

    if df.empty:
        print("No DM test results found!")
        return

    print(f"Loaded {len(df)} DM test results")
    print(f"Configuration breakdown:")
    print(f"  Alphas: {sorted(df['alpha'].unique())}")
    print(f"  Seeds: {len(df['seed'].unique())} unique seeds")
    print(f"  Calibration: {df['calibrate'].value_counts().to_dict()}")
    print(f"  Feature parity: {df['feature_parity'].value_counts().to_dict()}")

    # Create summaries
    print("\nCreating summaries...")

    # Overall summary
    summary = summarize_dm_results(df)

    # Model comparison table
    comparison_table = create_model_comparison_table(df)

    # Summary by alpha
    alpha_summary = (
        df.groupby("alpha")
        .agg(
            {
                "DM": ["mean", "std"],
                "p_value": ["mean", "std"],
                "significant_5pct": "sum",
                "significant_1pct": "sum",
            }
        )
        .round(4)
    )

    # Summary by calibration
    cal_summary = (
        df.groupby("calibrate")
        .agg(
            {
                "DM": ["mean", "std"],
                "p_value": ["mean", "std"],
                "significant_5pct": "sum",
                "significant_1pct": "sum",
            }
        )
        .round(4)
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    summary.to_csv(f"{args.output_dir}/dm_summary_n5000.csv", index=False)
    comparison_table.to_csv(f"{args.output_dir}/dm_model_comparison_n5000.csv")
    alpha_summary.to_csv(f"{args.output_dir}/dm_alpha_summary_n5000.csv")
    cal_summary.to_csv(f"{args.output_dir}/dm_calibration_summary_n5000.csv")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\nTotal DM tests: {len(df)}")
    print(
        f"Significant at 5% level: {(df['p_value'] < 0.05).sum()} ({(df['p_value'] < 0.05).mean()*100:.1f}%)"
    )
    print(
        f"Significant at 1% level: {(df['p_value'] < 0.01).sum()} ({(df['p_value'] < 0.01).mean()*100:.1f}%)"
    )

    print(f"\nMean DM statistic: {df['DM'].mean():.4f} (std: {df['DM'].std():.4f})")
    print(f"Mean p-value: {df['p_value'].mean():.4f} (std: {df['p_value'].std():.4f})")

    # Model pair analysis
    print(f"\nModel pair analysis:")
    for (alpha, A, B), group in df.groupby(["alpha", "A", "B"]):
        wins_A = (group["DM"] > 0).sum()
        wins_B = (group["DM"] < 0).sum()
        ties = (group["DM"] == 0).sum()
        significant = (group["p_value"] < 0.05).sum()

        print(
            f"  {A} vs {B} (α={alpha}): {wins_A} wins, {wins_B} losses, {ties} ties, {significant} significant"
        )

    print(f"\nResults saved to {args.output_dir}/")
    print("Files created:")
    print("  - dm_summary_n5000.csv: Detailed statistics by model pair")
    print("  - dm_model_comparison_n5000.csv: Win/loss counts by model")
    print("  - dm_alpha_summary_n5000.csv: Summary by alpha level")
    print("  - dm_calibration_summary_n5000.csv: Summary by calibration setting")


if __name__ == "__main__":
    main()
