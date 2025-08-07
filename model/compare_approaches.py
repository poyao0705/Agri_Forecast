import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_sequence_differences():
    """
    Demonstrate the key differences between overlapping sequences and expanding window approaches.
    """

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    sample_data = np.random.randn(100)

    print("=" * 60)
    print("SEQUENCE CREATION APPROACHES COMPARISON")
    print("=" * 60)

    # Approach 1: Overlapping sequences (Original Transformer)
    print("\n1. OVERLAPPING SEQUENCES APPROACH (Original Transformer)")
    print("-" * 50)

    seq_len = 21
    overlap = 0.3
    step = max(1, int(seq_len * (1 - overlap)))

    overlapping_sequences = []
    for i in range(0, len(sample_data) - seq_len, step):
        sequence = sample_data[i : i + seq_len]
        target = sample_data[i + seq_len]
        overlapping_sequences.append((sequence, target))

    print(f"Total sequences created: {len(overlapping_sequences)}")
    print(f"Sequence length: {seq_len}")
    print(f"Step size: {step}")
    print(f"Overlap: {overlap:.1%}")

    # Show first few sequences
    print("\nFirst 3 sequences:")
    for i, (seq, target) in enumerate(overlapping_sequences[:3]):
        print(
            f"Sequence {i+1}: Data points {i*step}-{i*step+seq_len} -> Target: {target:.4f}"
        )

    # Approach 2: Expanding window (GARCH-like)
    print("\n2. EXPANDING WINDOW APPROACH (GARCH-like)")
    print("-" * 50)

    min_seq_len = 21
    expanding_sequences = []

    for i in range(min_seq_len, len(sample_data)):
        # Use all data from start up to current point
        sequence = sample_data[:i]
        target = sample_data[i]
        expanding_sequences.append((sequence, target))

    print(f"Total sequences created: {len(expanding_sequences)}")
    print(f"Minimum sequence length: {min_seq_len}")
    print(f"Maximum sequence length: {len(sample_data)}")

    # Show first few sequences
    print("\nFirst 3 sequences:")
    for i, (seq, target) in enumerate(expanding_sequences[:3]):
        print(f"Sequence {i+1}: Data points 0-{len(seq)} -> Target: {target:.4f}")

    # Approach 3: GARCH approach (for comparison)
    print("\n3. GARCH APPROACH")
    print("-" * 50)
    print("GARCH uses expanding window but with key differences:")
    print("- Only uses return series (not multiple features)")
    print("- Fits parametric model (GARCH equation) at each step")
    print("- Assumes specific volatility dynamics")
    print("- More interpretable but less flexible")

    # Visual comparison
    plt.figure(figsize=(15, 10))

    # Plot 1: Overlapping sequences
    plt.subplot(2, 2, 1)
    plt.plot(sample_data, "b-", alpha=0.7, label="Data")

    # Highlight first few sequences
    colors = ["red", "green", "orange", "purple"]
    for i, (seq, target) in enumerate(overlapping_sequences[:4]):
        start_idx = i * step
        end_idx = start_idx + seq_len
        plt.plot(
            range(start_idx, end_idx),
            seq,
            color=colors[i],
            linewidth=2,
            label=f"Sequence {i+1}" if i < 3 else None,
        )
        plt.scatter(end_idx, target, color=colors[i], s=50, zorder=5)

    plt.title("Overlapping Sequences Approach")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Expanding window
    plt.subplot(2, 2, 2)
    plt.plot(sample_data, "b-", alpha=0.7, label="Data")

    # Highlight first few sequences
    for i, (seq, target) in enumerate(expanding_sequences[:4]):
        plt.plot(
            range(len(seq)),
            seq,
            color=colors[i],
            linewidth=2,
            label=f"Sequence {i+1}" if i < 3 else None,
        )
        plt.scatter(len(seq), target, color=colors[i], s=50, zorder=5)

    plt.title("Expanding Window Approach")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Sequence length comparison
    plt.subplot(2, 2, 3)
    overlap_lengths = [len(seq) for seq, _ in overlapping_sequences]
    expanding_lengths = [len(seq) for seq, _ in expanding_sequences]

    plt.hist(overlap_lengths, alpha=0.7, label="Overlapping", bins=20)
    plt.hist(expanding_lengths, alpha=0.7, label="Expanding", bins=20)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title("Sequence Length Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Training data comparison
    plt.subplot(2, 2, 4)
    overlap_counts = len(overlapping_sequences)
    expanding_counts = len(expanding_sequences)

    approaches = ["Overlapping\nSequences", "Expanding\nWindow", "GARCH\n(Reference)"]
    counts = [overlap_counts, expanding_counts, len(sample_data) - min_seq_len]
    colors_plot = ["skyblue", "lightcoral", "lightgreen"]

    bars = plt.bar(approaches, counts, color=colors_plot, alpha=0.7)
    plt.ylabel("Number of Training Samples")
    plt.title("Training Data Comparison")

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    comparison_data = {
        "Approach": ["Overlapping Sequences", "Expanding Window", "GARCH"],
        "Training Samples": [
            len(overlapping_sequences),
            len(expanding_sequences),
            len(sample_data) - min_seq_len,
        ],
        "Sequence Length": ["Fixed (21)", "Variable (21-100)", "Variable (21-100)"],
        "Data Usage": [
            "Overlapping windows",
            "All available data",
            "All available data",
        ],
        "Model Type": ["Neural Network", "Neural Network", "Parametric"],
        "Flexibility": ["High", "High", "Low"],
        "Interpretability": ["Low", "Low", "High"],
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    print("\nKey Insights:")
    print(
        "- Overlapping sequences create more training data but may miss long-term patterns"
    )
    print("- Expanding window uses full history but creates less training data")
    print("- GARCH combines expanding window with parametric modeling")
    print("- Your transformer can now use either approach!")


if __name__ == "__main__":
    demonstrate_sequence_differences()
