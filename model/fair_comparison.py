import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm, chi2
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Import transformer functions
exec(open("model/model_v0.1.py").read())


def load_and_preprocess_data():
    """Load and preprocess data consistently for both models"""
    df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")

    df["return"] = df["close"].pct_change()
    df["squared_return"] = df["return"] ** 2
    df["target_return"] = df["return"].shift(-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_ma"] = df["return"].rolling(window=5).mean()
    df["vol_ma"] = df["return"].rolling(window=21).std()

    df.dropna(inplace=True)

    # Use same split for both models
    split = int(len(df) * 0.8)  # 80/20 split like GARCH

    # For GARCH: only returns
    returns = df["target_return"].values
    train_returns = returns[:split]
    test_returns = returns[split:]

    # For Transformer: all features
    features = [
        "return",
        "squared_return",
        "log_return",
        "return_ma",
        "vol_ma",
        "gk_vol_1d",
        "gk_vol_21d",
        "weighted_tavg",
        "weighted_prcp",
        "Fed_Rate",
        "GDP",
        "CPI",
    ]

    X = df[features].values
    y = df["target_return"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train = X_scaled[:split]
    y_train = y[:split]
    X_test = X_scaled[split:]
    y_test = y[split:]

    return (train_returns, test_returns), (X_train, y_train, X_test, y_test), scaler


def run_garch_model(train_returns, test_returns):
    """Run GARCH model with consistent evaluation"""
    print("=" * 60)
    print("GARCH(1,1) MODEL EVALUATION")
    print("=" * 60)

    # Fit GARCH model
    garch_model = arch_model(
        train_returns, vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    model_fit = garch_model.fit(disp="off")

    print("GARCH model fitted successfully")

    # Rolling forecast approach
    predicted_var = []
    predicted_es = []
    mu_list = []
    sigma_list = []

    alpha = 0.05
    z = norm.ppf(alpha)
    c = norm.pdf(z) / alpha

    print("Generating GARCH forecasts...")

    for i in range(len(test_returns)):
        # Use data up to current point for forecasting
        current_data = train_returns[: len(train_returns) + i]

        # Refit model (expanding window)
        temp_model = arch_model(
            current_data, vol="GARCH", p=1, q=1, dist="normal", rescale=False
        )
        temp_fit = temp_model.fit(disp="off")

        # Generate 1-step ahead forecast
        forecast = temp_fit.forecast(horizon=1)

        mu = forecast.mean.iloc[-1, 0]
        sigma = np.sqrt(forecast.variance.iloc[-1, 0])

        mu_list.append(mu)
        sigma_list.append(sigma)

        # Calculate VaR and ES
        var_forecast = mu + sigma * z
        es_forecast = mu - sigma * c

        predicted_var.append(var_forecast)
        predicted_es.append(es_forecast)

    predicted_var = np.array(predicted_var)
    predicted_es = np.array(predicted_es)

    # Evaluate GARCH
    hits = (test_returns <= predicted_var).astype(int)
    hit_rate = hits.mean()
    n = len(test_returns)
    x = hits.sum()

    # Kupiec Test
    p_hat = x / n if x > 0 else 0.001
    if p_hat == 0 or p_hat == 1:
        LR_pof = float("inf")
        p_value = 0.0
    else:
        LR_pof = -2 * (
            np.log((1 - alpha) ** (n - x) * alpha**x)
            - np.log((1 - p_hat) ** (n - x) * p_hat**x)
        )
        p_value = 1 - chi2.cdf(LR_pof, df=1)

    print(f"Hit rate: {hit_rate:.4f} (Expected: {alpha})")
    print(f"Number of breaches: {x} out of {n}")
    print(f"Kupiec Test LR_pof: {LR_pof:.4f}, p-value: {p_value:.4f}")
    print(f"Test Result: {'REJECT' if p_value < 0.05 else 'ACCEPT'} null hypothesis")

    if x > 0:
        actual_es = test_returns[hits == 1].mean()
        predicted_es_breaches = predicted_es[hits == 1].mean()
        print(f"Average Actual Return under VaR: {actual_es:.4f}")
        print(f"Predicted ES (at breach points): {predicted_es_breaches:.4f}")

    return predicted_var, predicted_es, hit_rate, p_value, hits


def run_transformer_model(X_train, y_train, X_test, y_test, scaler):
    """Run Transformer model with same evaluation period as GARCH"""
    print("\n" + "=" * 60)
    print("TRANSFORMER HYBRID MODEL EVALUATION")
    print("=" * 60)

    # Train transformer with overlapping sequences
    model, _, _ = train_with_overlapping_evaluate_with_expanding(
        X_train,
        y_train,
        input_dim=X_train.shape[1],
        max_epochs=100,  # Reduced for quick comparison
    )

    # Evaluate with expanding window (same approach as GARCH)
    predicted_var, predicted_es = evaluate_frozen_model_with_expanding_window(
        model,
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        len(X_train),
    )

    # Calibrate predictions
    print("\nCalibrating Transformer predictions...")

    # Create calibration data
    X_seq_cal, y_seq_cal = create_sequences_with_overlap(
        X_train, y_train, seq_len=21, overlap=0.3
    )
    X_tensor_cal = torch.tensor(X_seq_cal, dtype=torch.float32)
    y_tensor_cal = torch.tensor(y_seq_cal, dtype=torch.float32)

    var_adjustment_factor = calibrate_var_post_training(
        model, X_tensor_cal, y_tensor_cal
    )
    es_calibration_factor = calibrate_es_post_training(
        model, X_tensor_cal, y_tensor_cal
    )

    # Apply calibration
    predicted_var = predicted_var * var_adjustment_factor
    predicted_es = predicted_es * es_calibration_factor

    # Evaluate transformer
    hits = (y_test <= predicted_var).astype(int)
    hit_rate = hits.mean()
    n = len(y_test)
    x = hits.sum()

    # Kupiec Test
    p_hat = x / n if x > 0 else 0.001
    if p_hat == 0 or p_hat == 1:
        LR_pof = float("inf")
        p_value = 0.0
    else:
        LR_pof = -2 * (
            np.log((1 - 0.05) ** (n - x) * 0.05**x)
            - np.log((1 - p_hat) ** (n - x) * p_hat**x)
        )
        p_value = 1 - chi2.cdf(LR_pof, df=1)

    print(f"Hit rate: {hit_rate:.4f} (Expected: 0.05)")
    print(f"Number of breaches: {x} out of {n}")
    print(f"Kupiec Test LR_pof: {LR_pof:.4f}, p-value: {p_value:.4f}")
    print(f"Test Result: {'REJECT' if p_value < 0.05 else 'ACCEPT'} null hypothesis")

    if x > 0:
        actual_es = y_test[hits == 1].mean()
        predicted_es_breaches = predicted_es[hits == 1].mean()
        print(f"Average Actual Return under VaR: {actual_es:.4f}")
        print(f"Predicted ES (at breach points): {predicted_es_breaches:.4f}")

    print(f"VaR Calibration Factor: {var_adjustment_factor:.4f}")
    print(f"ES Calibration Factor: {es_calibration_factor:.4f}")

    return predicted_var, predicted_es, hit_rate, p_value, hits


def compare_models(garch_results, transformer_results, y_test):
    """Compare GARCH and Transformer results"""
    print("\n" + "=" * 60)
    print("FAIR COMPARISON: GARCH vs TRANSFORMER")
    print("=" * 60)

    garch_var, garch_es, garch_hit_rate, garch_p_value, garch_hits = garch_results
    trans_var, trans_es, trans_hit_rate, trans_p_value, trans_hits = transformer_results

    # Create comparison table
    comparison_data = {
        "Metric": [
            "Hit Rate",
            "Expected Hit Rate",
            "Number of Breaches",
            "Total Observations",
            "Kupiec Test p-value",
            "Model Status",
            "Hit Rate Error",
        ],
        "GARCH": [
            f"{garch_hit_rate:.4f}",
            "0.0500",
            f"{garch_hits.sum()}",
            f"{len(y_test)}",
            f"{garch_p_value:.4f}",
            "ACCEPT" if garch_p_value >= 0.05 else "REJECT",
            f"{abs(garch_hit_rate - 0.05):.4f}",
        ],
        "Transformer": [
            f"{trans_hit_rate:.4f}",
            "0.0500",
            f"{trans_hits.sum()}",
            f"{len(y_test)}",
            f"{trans_p_value:.4f}",
            "ACCEPT" if trans_p_value >= 0.05 else "REJECT",
            f"{abs(trans_hit_rate - 0.05):.4f}",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    # Determine winner
    print("\n" + "=" * 60)
    print("WINNER ANALYSIS")
    print("=" * 60)

    # Hit rate comparison (closer to 5% is better)
    garch_error = abs(garch_hit_rate - 0.05)
    trans_error = abs(trans_hit_rate - 0.05)

    if garch_error < trans_error:
        hit_rate_winner = "GARCH"
    elif trans_error < garch_error:
        hit_rate_winner = "Transformer"
    else:
        hit_rate_winner = "Tie"

    # Statistical validity comparison
    if garch_p_value >= 0.05 and trans_p_value >= 0.05:
        validity_winner = "Both Valid"
    elif garch_p_value >= 0.05:
        validity_winner = "GARCH"
    elif trans_p_value >= 0.05:
        validity_winner = "Transformer"
    else:
        validity_winner = "Both Invalid"

    print(f"Hit Rate Accuracy Winner: {hit_rate_winner}")
    print(f"Statistical Validity Winner: {validity_winner}")

    # Overall winner
    if garch_p_value >= 0.05 and trans_p_value >= 0.05:
        if garch_error < trans_error:
            overall_winner = "GARCH"
        elif trans_error < garch_error:
            overall_winner = "Transformer"
        else:
            overall_winner = "Tie"
    elif garch_p_value >= 0.05:
        overall_winner = "GARCH"
    elif trans_p_value >= 0.05:
        overall_winner = "Transformer"
    else:
        overall_winner = "Both Need Improvement"

    print(f"Overall Winner: {overall_winner}")

    # Plot comparison
    plt.figure(figsize=(15, 10))

    # VaR comparison
    plt.subplot(2, 2, 1)
    plt.plot(y_test, label="Actual Returns", alpha=0.7)
    plt.plot(garch_var, label="GARCH VaR", color="red", linewidth=1)
    plt.plot(trans_var, label="Transformer VaR", color="blue", linewidth=1)
    plt.title("VaR Comparison: GARCH vs Transformer")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Hit rate comparison
    plt.subplot(2, 2, 2)
    rolling_garch = pd.Series(garch_hits).rolling(window=20).mean()
    rolling_trans = pd.Series(trans_hits).rolling(window=20).mean()
    plt.plot(rolling_garch, label="GARCH Hit Rate", color="red")
    plt.plot(rolling_trans, label="Transformer Hit Rate", color="blue")
    plt.axhline(y=0.05, color="black", linestyle="--", label="Expected (5%)")
    plt.title("Rolling Hit Rate Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ES comparison
    plt.subplot(2, 2, 3)
    if garch_hits.sum() > 0 and trans_hits.sum() > 0:
        plt.scatter(
            garch_es[garch_hits == 1],
            y_test[garch_hits == 1],
            alpha=0.7,
            label="GARCH ES",
            color="red",
        )
        plt.scatter(
            trans_es[trans_hits == 1],
            y_test[trans_hits == 1],
            alpha=0.7,
            label="Transformer ES",
            color="blue",
        )
        plt.plot(
            [min(garch_es), max(garch_es)],
            [min(garch_es), max(garch_es)],
            "k--",
            label="Perfect Prediction",
        )
        plt.xlabel("Predicted ES")
        plt.ylabel("Actual Returns")
        plt.title("ES Prediction Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Performance summary
    plt.subplot(2, 2, 4)
    models = ["GARCH", "Transformer"]
    hit_rates = [garch_hit_rate, trans_hit_rate]
    p_values = [garch_p_value, trans_p_value]

    x_pos = np.arange(len(models))
    plt.bar(x_pos, hit_rates, alpha=0.7, color=["red", "blue"])
    plt.axhline(y=0.05, color="black", linestyle="--", label="Target (5%)")
    plt.xlabel("Model")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate Comparison")
    plt.xticks(x_pos, models)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return overall_winner


def main():
    """Main function for fair comparison"""
    print("=" * 60)
    print("FAIR COMPARISON: GARCH vs TRANSFORMER")
    print("=" * 60)
    print("Using same data split (80/20) and evaluation period")
    print("Both models use expanding window evaluation")
    print("=" * 60)

    # Load and preprocess data consistently
    garch_data, transformer_data, scaler = load_and_preprocess_data()
    train_returns, test_returns = garch_data
    X_train, y_train, X_test, y_test = transformer_data

    print(f"Training samples: {len(train_returns)}")
    print(f"Test samples: {len(test_returns)}")

    # Run GARCH model
    garch_results = run_garch_model(train_returns, test_returns)

    # Run Transformer model
    transformer_results = run_transformer_model(
        X_train, y_train, X_test, y_test, scaler
    )

    # Compare results
    winner = compare_models(garch_results, transformer_results, y_test)

    print(f"\n🏆 Final Winner: {winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()
