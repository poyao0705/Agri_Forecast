#!/usr/bin/env python3
"""
Detailed analysis of Diebold-Mariano test results for n=5000.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_and_analyze_dm_results(base_path="artifacts/dgp=garch11_skt/n=5000"):
    """Load and perform detailed analysis of DM test results."""

    # Load the separated summary data
    try:
        summary_cal = pd.read_csv("results/dm_summary_n5000_calibrated.csv")
        summary_raw = pd.read_csv("results/dm_summary_n5000_raw.csv")

        # Combine for overall analysis
        summary_cal["calibration"] = "calibrated"
        summary_raw["calibration"] = "raw"
        summary_df = pd.concat([summary_cal, summary_raw], ignore_index=True)

    except FileNotFoundError:
        print(
            "Summary files not found. Please run summarize_dm_by_calibration.py first."
        )
        return

    print("=" * 80)
    print("DETAILED DIEBOLD-MARIANO TEST ANALYSIS FOR N=5000")
    print("=" * 80)

    # 1. Overall significance analysis
    print("\n1. OVERALL SIGNIFICANCE ANALYSIS")
    print("-" * 40)

    total_tests = len(summary_df) * 10  # 10 seeds per configuration
    significant_5pct = summary_df["significant_5pct"].sum()
    significant_1pct = summary_df["significant_1pct"].sum()

    print(f"Total DM tests: {total_tests}")
    print(
        f"Significant at 5% level: {significant_5pct} ({significant_5pct/total_tests*100:.1f}%)"
    )
    print(
        f"Significant at 1% level: {significant_1pct} ({significant_1pct/total_tests*100:.1f}%)"
    )

    # 2. Model performance ranking
    print("\n2. MODEL PERFORMANCE RANKING")
    print("-" * 40)

    # Calculate average DM statistics for each model
    model_performance = {}

    for _, row in summary_df.iterrows():
        A, B = row["A"], row["B"]
        dm_mean = row["DM_mean"]

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

    print("Model performance ranking (lower DM = better):")
    for i, (model, score) in enumerate(sorted_models, 1):
        print(f"  {i}. {model}: {score:.4f}")

    # 3. Effect size analysis
    print("\n3. EFFECT SIZE ANALYSIS")
    print("-" * 40)

    print("Effect sizes by model pair (|DM| > 1.0 considered large):")
    for _, row in summary_df.iterrows():
        alpha, A, B = row["alpha"], row["A"], row["B"]
        dm_mean = row["DM_mean"]
        dm_std = row["DM_std"]

        effect_size = abs(dm_mean)
        significance = (
            "LARGE" if effect_size > 1.0 else "MEDIUM" if effect_size > 0.5 else "SMALL"
        )

        print(
            f"  {A} vs {B} (α={alpha}): DM={dm_mean:.4f} ± {dm_std:.4f} ({significance})"
        )

    # 4. Alpha-level analysis
    print("\n4. ALPHA-LEVEL ANALYSIS")
    print("-" * 40)

    alpha_analysis = (
        summary_df.groupby("alpha")
        .agg(
            {
                "DM_mean": ["mean", "std"],
                "significant_5pct": "sum",
                "significant_1pct": "sum",
            }
        )
        .round(4)
    )

    print("Performance by alpha level:")
    for alpha in [0.01, 0.025, 0.05]:
        subset = summary_df[summary_df["alpha"] == alpha]
        avg_dm = subset["DM_mean"].mean()
        std_dm = subset["DM_mean"].std()
        sig_5pct = subset["significant_5pct"].sum()
        sig_1pct = subset["significant_1pct"].sum()

        print(
            f"  α={alpha}: DM={avg_dm:.4f} ± {std_dm:.4f}, {sig_5pct} sig at 5%, {sig_1pct} sig at 1%"
        )

    # 5. Practical significance
    print("\n5. PRACTICAL SIGNIFICANCE")
    print("-" * 40)

    print("Model pairs with consistent performance (wins in >70% of cases):")
    for _, row in summary_df.iterrows():
        alpha, A, B = row["alpha"], row["A"], row["B"]
        dm_mean = row["DM_mean"]

        # Calculate win percentage
        if dm_mean < 0:  # A wins (negative DM)
            wins_A = (dm_mean < 0) * 10  # All 10 seeds
            win_pct = wins_A / 10 * 100
            if win_pct > 70:
                print(f"  {A} consistently beats {B} (α={alpha}): {win_pct:.0f}% wins")
        else:  # B wins (positive DM)
            wins_B = (dm_mean > 0) * 10  # All 10 seeds
            win_pct = wins_B / 10 * 100
            if win_pct > 70:
                print(f"  {B} consistently beats {A} (α={alpha}): {win_pct:.0f}% wins")

    # 6. Statistical power analysis
    print("\n6. STATISTICAL POWER ANALYSIS")
    print("-" * 40)

    print(
        "Power analysis (assuming 80% power requires ~8 significant results out of 10):"
    )
    for _, row in summary_df.iterrows():
        alpha, A, B = row["alpha"], row["A"], row["B"]
        significant = row["significant_5pct"]
        power_estimate = significant / 10 * 100

        if power_estimate >= 80:
            power_level = "HIGH"
        elif power_estimate >= 50:
            power_level = "MEDIUM"
        else:
            power_level = "LOW"

        print(f"  {A} vs {B} (α={alpha}): {power_estimate:.0f}% power ({power_level})")

    # 7. Recommendations
    print("\n7. RECOMMENDATIONS")
    print("-" * 40)

    print("Based on the analysis:")

    # Find best performing model
    best_model = sorted_models[0][0]
    print(f"  • {best_model} appears to be the best performing model overall")

    # Find most significant differences
    significant_pairs = summary_df[
        summary_df["significant_5pct"] >= 2
    ]  # At least 2 significant results
    if not significant_pairs.empty:
        print("  • Statistically significant differences found in:")
        for _, row in significant_pairs.iterrows():
            print(
                f"    - {row['A']} vs {row['B']} (α={row['alpha']}): {row['significant_5pct']} significant"
            )

    # Find large effect sizes
    large_effects = summary_df[abs(summary_df["DM_mean"]) > 1.0]
    if not large_effects.empty:
        print("  • Large effect sizes observed in:")
        for _, row in large_effects.iterrows():
            print(
                f"    - {row['A']} vs {row['B']} (α={row['alpha']}): DM={row['DM_mean']:.3f}"
            )

    print("\n" + "=" * 80)


def create_visualizations():
    """Create visualizations of the DM test results."""

    # Load data
    try:
        summary_cal = pd.read_csv("results/dm_summary_n5000_calibrated.csv")
        summary_raw = pd.read_csv("results/dm_summary_n5000_raw.csv")

    except FileNotFoundError:
        print(
            "Summary files not found. Please run summarize_dm_by_calibration.py first."
        )
        return

    # Create output directory for plots
    os.makedirs("results/figures", exist_ok=True)

    # 1. DM statistics by model pair (calibrated)
    plt.figure(figsize=(12, 8))

    # Create a grouped bar plot for calibrated data
    model_pairs_cal = [f"{row['A']} vs {row['B']}" for _, row in summary_cal.iterrows()]
    alphas_cal = summary_cal["alpha"].values
    dm_means_cal = summary_cal["DM_mean"].values

    x = np.arange(len(model_pairs_cal))
    width = 0.25

    for i, alpha in enumerate([0.01, 0.025, 0.05]):
        mask = alphas_cal == alpha
        plt.bar(
            x[mask] + i * width,
            dm_means_cal[mask],
            width,
            label=f"α={alpha}",
            alpha=0.8,
        )

    plt.xlabel("Model Pair")
    plt.ylabel("Mean DM Statistic (lower = better)")
    plt.title("Diebold-Mariano Test Results - Calibrated Models")
    plt.xticks(x + width, model_pairs_cal, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "results/figures/dm_statistics_calibrated.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. DM statistics by model pair (raw)
    plt.figure(figsize=(12, 8))

    # Create a grouped bar plot for raw data
    model_pairs_raw = [f"{row['A']} vs {row['B']}" for _, row in summary_raw.iterrows()]
    alphas_raw = summary_raw["alpha"].values
    dm_means_raw = summary_raw["DM_mean"].values

    x = np.arange(len(model_pairs_raw))
    width = 0.25

    for i, alpha in enumerate([0.01, 0.025, 0.05]):
        mask = alphas_raw == alpha
        plt.bar(
            x[mask] + i * width,
            dm_means_raw[mask],
            width,
            label=f"α={alpha}",
            alpha=0.8,
        )

    plt.xlabel("Model Pair")
    plt.ylabel("Mean DM Statistic (lower = better)")
    plt.title("Diebold-Mariano Test Results - Raw Models")
    plt.xticks(x + width, model_pairs_raw, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/dm_statistics_raw.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Significance heatmap (calibrated)
    plt.figure(figsize=(10, 6))

    # Create pivot table for significance (calibrated)
    pivot_data_cal = summary_cal.pivot(
        index="alpha", columns=["A", "B"], values="significant_5pct"
    )

    sns.heatmap(
        pivot_data_cal,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Significant Results (5% level)"},
    )
    plt.title("Significance Heatmap: Calibrated Models (5% level)")
    plt.xlabel("Model Pair")
    plt.ylabel("Alpha Level")
    plt.tight_layout()
    plt.savefig(
        "results/figures/dm_significance_heatmap_calibrated.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Significance heatmap (raw)
    plt.figure(figsize=(10, 6))

    # Create pivot table for significance (raw)
    pivot_data_raw = summary_raw.pivot(
        index="alpha", columns=["A", "B"], values="significant_5pct"
    )

    sns.heatmap(
        pivot_data_raw,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Significant Results (5% level)"},
    )
    plt.title("Significance Heatmap: Raw Models (5% level)")
    plt.xlabel("Model Pair")
    plt.ylabel("Alpha Level")
    plt.tight_layout()
    plt.savefig(
        "results/figures/dm_significance_heatmap_raw.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Visualizations saved to results/figures/")


if __name__ == "__main__":
    import os

    load_and_analyze_dm_results()
    create_visualizations()
