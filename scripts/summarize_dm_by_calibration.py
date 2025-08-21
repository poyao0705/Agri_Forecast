#!/usr/bin/env python3
"""
Summarize Diebold-Mariano test results for n=5000, separated by calibration status.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_dm_results_by_calibration(base_path="artifacts/dgp=garch11_skt/n=5000"):
    """Load all DM test results from the n=5000 directory, separated by calibration."""
    results_calibrated = []
    results_raw = []

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
            alpha = float(alpha_part.replace("alpha=", "")) / 1000
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

            # Separate by calibration status
            if calibrate:
                results_calibrated.append(df)
            else:
                results_raw.append(df)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Combine results
    df_calibrated = (
        pd.concat(results_calibrated, ignore_index=True)
        if results_calibrated
        else pd.DataFrame()
    )
    df_raw = (
        pd.concat(results_raw, ignore_index=True) if results_raw else pd.DataFrame()
    )

    return df_calibrated, df_raw


def summarize_dm_results(df, calibration_type):
    """Create comprehensive summary of DM test results."""

    if df.empty:
        print(f"No {calibration_type} DM test results found!")
        return None

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


def create_model_comparison_table(df, calibration_type):
    """Create a table showing which model wins in each comparison."""

    def determine_winner(row):
        if row["p_value"] < 0.05:  # Significant difference
            if row["DM"] < 0:  # Negative DM means A has lower losses (better)
                return f"{row['A']} wins"
            else:
                return f"{row['B']} wins"
        else:
            return "No significant difference"

    df["winner"] = df.apply(determine_winner, axis=1)

    # Count wins by model
    wins_by_model = df.groupby(["alpha", "winner"]).size().unstack(fill_value=0)

    return wins_by_model


def analyze_model_performance(df, calibration_type):
    """Analyze model performance ranking. Lower DM = better performance."""

    # Calculate average DM statistics for each model
    model_performance = {}

    for _, row in (
        df.groupby(["alpha", "A", "B"]).agg({"DM": "mean"}).reset_index().iterrows()
    ):
        A, B = row["A"], row["B"]
        dm_mean = row["DM"]

        if A not in model_performance:
            model_performance[A] = []
        if B not in model_performance:
            model_performance[B] = []

        # DM = Loss_A - Loss_B
        # Negative DM: A has lower losses (better)
        # Positive DM: B has lower losses (better)
        if dm_mean < 0:  # A is better
            model_performance[A].append(
                -abs(dm_mean)
            )  # Better performance (lower score)
            model_performance[B].append(
                abs(dm_mean)
            )  # Worse performance (higher score)
        else:  # B is better
            model_performance[A].append(
                abs(dm_mean)
            )  # Worse performance (higher score)
            model_performance[B].append(
                -abs(dm_mean)
            )  # Better performance (lower score)

    # Calculate average performance (lower is better)
    avg_performance = {}
    for model, scores in model_performance.items():
        avg_performance[model] = np.mean(scores)

    # Sort by performance (ascending - lower is better)
    sorted_models = sorted(avg_performance.items(), key=lambda x: x[1])

    return sorted_models


def main():
    parser = argparse.ArgumentParser(
        description="Summarize DM test results for n=5000 by calibration"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for summaries"
    )
    parser.add_argument(
        "--base-path",
        default="artifacts/dgp=garch11_skt/n=5000",
        help="Base path to search for DM results",
    )

    args = parser.parse_args()

    print("Loading DM test results by calibration status...")
    df_calibrated, df_raw = load_dm_results_by_calibration(args.base_path)

    print(f"Loaded {len(df_calibrated)} calibrated DM test results")
    print(f"Loaded {len(df_raw)} raw DM test results")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze calibrated results
    if not df_calibrated.empty:
        print("\n" + "=" * 60)
        print("CALIBRATED MODEL RESULTS")
        print("=" * 60)

        summary_cal = summarize_dm_results(df_calibrated, "calibrated")
        comparison_cal = create_model_comparison_table(df_calibrated, "calibrated")
        performance_cal = analyze_model_performance(df_calibrated, "calibrated")

        # Save calibrated results
        summary_cal.to_csv(
            f"{args.output_dir}/dm_summary_n5000_calibrated.csv", index=False
        )
        comparison_cal.to_csv(
            f"{args.output_dir}/dm_model_comparison_n5000_calibrated.csv"
        )

        # Print calibrated analysis
        print(f"\nConfiguration breakdown (calibrated):")
        print(f"  Alphas: {sorted(df_calibrated['alpha'].unique())}")
        print(f"  Seeds: {len(df_calibrated['seed'].unique())} unique seeds")
        print(
            f"  Feature parity: {df_calibrated['feature_parity'].value_counts().to_dict()}"
        )

        print(f"\nTotal calibrated DM tests: {len(df_calibrated)}")
        print(
            f"Significant at 5% level: {(df_calibrated['p_value'] < 0.05).sum()} ({(df_calibrated['p_value'] < 0.05).mean()*100:.1f}%)"
        )
        print(
            f"Significant at 1% level: {(df_calibrated['p_value'] < 0.01).sum()} ({(df_calibrated['p_value'] < 0.01).mean()*100:.1f}%)"
        )

        print(
            f"\nMean DM statistic: {df_calibrated['DM'].mean():.4f} (std: {df_calibrated['DM'].std():.4f})"
        )
        print(
            f"Mean p-value: {df_calibrated['p_value'].mean():.4f} (std: {df_calibrated['p_value'].std():.4f})"
        )

        print(f"\nModel performance ranking (calibrated, lower DM = better):")
        for i, (model, score) in enumerate(performance_cal, 1):
            print(f"  {i}. {model}: {score:.4f}")

        print(f"\nModel pair analysis (calibrated):")
        for (alpha, A, B), group in df_calibrated.groupby(["alpha", "A", "B"]):
            wins_A = (group["DM"] < 0).sum()  # Negative DM means A wins
            wins_B = (group["DM"] > 0).sum()  # Positive DM means B wins
            ties = (group["DM"] == 0).sum()
            significant = (group["p_value"] < 0.05).sum()

            print(
                f"  {A} vs {B} (α={alpha}): {wins_A} wins, {wins_B} losses, {ties} ties, {significant} significant"
            )

    # Analyze raw results
    if not df_raw.empty:
        print("\n" + "=" * 60)
        print("RAW MODEL RESULTS")
        print("=" * 60)

        summary_raw = summarize_dm_results(df_raw, "raw")
        comparison_raw = create_model_comparison_table(df_raw, "raw")
        performance_raw = analyze_model_performance(df_raw, "raw")

        # Save raw results
        summary_raw.to_csv(f"{args.output_dir}/dm_summary_n5000_raw.csv", index=False)
        comparison_raw.to_csv(f"{args.output_dir}/dm_model_comparison_n5000_raw.csv")

        # Print raw analysis
        print(f"\nConfiguration breakdown (raw):")
        print(f"  Alphas: {sorted(df_raw['alpha'].unique())}")
        print(f"  Seeds: {len(df_raw['seed'].unique())} unique seeds")
        print(f"  Feature parity: {df_raw['feature_parity'].value_counts().to_dict()}")

        print(f"\nTotal raw DM tests: {len(df_raw)}")
        print(
            f"Significant at 5% level: {(df_raw['p_value'] < 0.05).sum()} ({(df_raw['p_value'] < 0.05).mean()*100:.1f}%)"
        )
        print(
            f"Significant at 1% level: {(df_raw['p_value'] < 0.01).sum()} ({(df_raw['p_value'] < 0.01).mean()*100:.1f}%)"
        )

        print(
            f"\nMean DM statistic: {df_raw['DM'].mean():.4f} (std: {df_raw['DM'].std():.4f})"
        )
        print(
            f"Mean p-value: {df_raw['p_value'].mean():.4f} (std: {df_raw['p_value'].std():.4f})"
        )

        print(f"\nModel performance ranking (raw, lower DM = better):")
        for i, (model, score) in enumerate(performance_raw, 1):
            print(f"  {i}. {model}: {score:.4f}")

        print(f"\nModel pair analysis (raw):")
        for (alpha, A, B), group in df_raw.groupby(["alpha", "A", "B"]):
            wins_A = (group["DM"] < 0).sum()  # Negative DM means A wins
            wins_B = (group["DM"] > 0).sum()  # Positive DM means B wins
            ties = (group["DM"] == 0).sum()
            significant = (group["p_value"] < 0.05).sum()

            print(
                f"  {A} vs {B} (α={alpha}): {wins_A} wins, {wins_B} losses, {ties} ties, {significant} significant"
            )

    # Combined analysis
    if not df_calibrated.empty and not df_raw.empty:
        print("\n" + "=" * 60)
        print("COMPARISON: CALIBRATED VS RAW")
        print("=" * 60)

        # Compare overall significance rates
        cal_sig_5pct = (df_calibrated["p_value"] < 0.05).mean() * 100
        raw_sig_5pct = (df_raw["p_value"] < 0.05).mean() * 100
        cal_sig_1pct = (df_calibrated["p_value"] < 0.01).mean() * 100
        raw_sig_1pct = (df_raw["p_value"] < 0.01).mean() * 100

        print(f"Significance rates:")
        print(f"  Calibrated: {cal_sig_5pct:.1f}% at 5%, {cal_sig_1pct:.1f}% at 1%")
        print(f"  Raw: {raw_sig_5pct:.1f}% at 5%, {raw_sig_1pct:.1f}% at 1%")

        # Compare mean DM statistics
        cal_dm_mean = df_calibrated["DM"].mean()
        raw_dm_mean = df_raw["DM"].mean()

        print(f"\nMean DM statistics:")
        print(f"  Calibrated: {cal_dm_mean:.4f}")
        print(f"  Raw: {raw_dm_mean:.4f}")
        print(f"  Difference: {cal_dm_mean - raw_dm_mean:.4f}")

    print(f"\nResults saved to {args.output_dir}/")
    print("Files created:")
    if not df_calibrated.empty:
        print("  - dm_summary_n5000_calibrated.csv: Calibrated model statistics")
        print(
            "  - dm_model_comparison_n5000_calibrated.csv: Calibrated model comparisons"
        )
    if not df_raw.empty:
        print("  - dm_summary_n5000_raw.csv: Raw model statistics")
        print("  - dm_model_comparison_n5000_raw.csv: Raw model comparisons")


if __name__ == "__main__":
    main()
