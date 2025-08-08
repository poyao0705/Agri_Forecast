import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import os
import joblib

os.makedirs("model/saved_models", exist_ok=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class BasicVaRTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=32, num_heads=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # More sophisticated output layer
        self.output_layer = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, 2),
        )

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1]  # Use the last time step
        raw_output = self.output_layer(x)

        # Apply clamping to prevent extreme values
        raw_output = torch.clamp(raw_output, -10, 10)

        # Use scale parameter for better calibration
        scale = 0.03  # More conservative scale for transformer
        var = -scale * torch.exp(torch.clamp(raw_output[:, 0], -3, 3))
        es = var - scale * torch.exp(torch.clamp(raw_output[:, 1], -3, 3))

        return torch.stack([var, es], dim=1)


class BasicFZ0Loss(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        var, es = y_pred[:, 0], y_pred[:, 1]
        eps = 1e-8  # To avoid division by zero

        indicator = (y_true <= var).float()
        term1 = -(1 / (self.alpha * es + eps)) * indicator * (var - y_true)
        term2 = (var / (es + eps)) + torch.log(-es + eps) - 1

        return torch.mean(term1 + term2)


def create_sequences_with_overlap(X, y, seq_len=21, overlap=0.5):
    """Create sequences with overlap to increase training data"""
    step = max(1, int(seq_len * (1 - overlap)))
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_len, step):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


# Add ES calibration function
def calibrate_es_post_training(model, X_tensor, y_tensor, alpha=0.05):
    """Post-training ES calibration to improve tail risk estimates"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    y_true = y_tensor.cpu().numpy()
    var_pred = predictions[:, 0]

    # Find actual breaches
    breaches = y_true <= var_pred
    actual_es = y_true[breaches].mean() if breaches.sum() > 0 else var_pred.mean()

    # Calculate empirical ES from historical data
    empirical_var = np.quantile(y_true, alpha)
    empirical_es = y_true[y_true <= empirical_var].mean()

    # ES scaling factor based on historical underestimation
    if actual_es != 0:
        es_scaling_factor = max(1.2, empirical_es / actual_es)  # Minimum 20% adjustment
        print(f"ES calibration factor: {es_scaling_factor:.4f}")
        return es_scaling_factor
    else:
        return 1.0


def calibrate_var_post_training(model, X_tensor, y_tensor, alpha=0.05):
    """Post-training VaR calibration to improve hit rate"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    y_true = y_tensor.cpu().numpy()
    var_pred = predictions[:, 0]

    # Calculate current hit rate
    hits = (y_true <= var_pred).astype(int)
    current_hit_rate = hits.mean()

    print(f"Current hit rate: {current_hit_rate:.4f} (Target: {alpha})")

    # Calculate adjustment factor
    if current_hit_rate > 0:
        # If hit rate is too high, make VaR less negative (less conservative)
        if current_hit_rate > alpha * 1.5:  # If more than 50% above target
            adjustment_factor = 0.8  # Make VaR 20% less negative
            print(
                f"VaR adjustment factor: {adjustment_factor:.4f} (making less conservative)"
            )
        elif current_hit_rate > alpha * 1.2:  # If more than 20% above target
            adjustment_factor = 0.9  # Make VaR 10% less negative
            print(
                f"VaR adjustment factor: {adjustment_factor:.4f} (making less conservative)"
            )
        else:
            adjustment_factor = 1.0
            print(
                f"VaR adjustment factor: {adjustment_factor:.4f} (no adjustment needed)"
            )
    else:
        adjustment_factor = 1.0
        print(
            f"VaR adjustment factor: {adjustment_factor:.4f} (no breaches to calibrate)"
        )

    return adjustment_factor


def comprehensive_evaluation(
    model, X_tensor, y_tensor, threshold_alpha=0.05, es_calibration_factor=1.0
):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
    y_true = y_tensor.cpu().numpy()
    predicted_var = y_pred[:, 0]
    predicted_es = y_pred[:, 1] * es_calibration_factor  # Apply calibration

    # Basic statistics
    hits = (y_true <= predicted_var).astype(int)
    hit_rate = hits.mean()
    n = len(y_true)
    x = hits.sum()

    # Kupiec Test
    p_hat = x / n
    if p_hat == 0 or p_hat == 1:
        LR_pof = float("inf")
        p_value = 0.0
    else:
        LR_pof = -2 * (
            np.log((1 - threshold_alpha) ** (n - x) * threshold_alpha**x)
            - np.log((1 - p_hat) ** (n - x) * p_hat**x)
        )
        p_value = 1 - chi2.cdf(LR_pof, df=1)

    # ES evaluation
    breach_returns = y_true[hits == 1]
    avg_actual_es = breach_returns.mean() if len(breach_returns) > 0 else np.nan
    avg_predicted_es = (
        predicted_es[hits == 1].mean() if len(breach_returns) > 0 else np.nan
    )

    # Additional metrics
    var_quantile = np.quantile(y_true, threshold_alpha)
    es_quantile = y_true[y_true <= var_quantile].mean()

    # Plotting (same as before)
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(y_true, label="Actual Returns", alpha=0.7, linewidth=0.8)
    plt.plot(
        predicted_var,
        label=f"Predicted VaR ({threshold_alpha*100:.0f}%)",
        color="red",
        linewidth=1,
    )
    plt.scatter(
        np.where(hits)[0],
        y_true[hits == 1],
        color="black",
        label="Breaches",
        zorder=5,
        s=10,
    )
    plt.title("VaR Backtesting")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    window = min(20, max(5, len(hits) // 3))  # Adaptive window
    rolling_hit_rate = pd.Series(hits).rolling(window=window).mean()

    plt.plot(rolling_hit_rate, label=f"Rolling Hit Rate ({window} days)")

    plt.axhline(
        y=threshold_alpha,
        color="red",
        linestyle="--",
        label=f"Expected ({threshold_alpha})",
    )
    plt.title("Rolling Hit Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    if len(breach_returns) > 0:
        sorted_breaches = np.sort(breach_returns)
        theoretical_quantiles = np.linspace(0, 1, len(sorted_breaches))
        plt.scatter(theoretical_quantiles, sorted_breaches, alpha=0.7)
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Breach Returns")
        plt.title("Q-Q Plot of Breach Returns")
        plt.grid(True)

    plt.subplot(2, 2, 4)
    if len(breach_returns) > 0:
        plt.scatter(predicted_es[hits == 1], breach_returns, alpha=0.7)
        plt.plot(
            [predicted_es[hits == 1].min(), predicted_es[hits == 1].max()],
            [predicted_es[hits == 1].min(), predicted_es[hits == 1].max()],
            "r--",
            label="Perfect Prediction",
        )
        plt.xlabel("Predicted ES")
        plt.ylabel("Actual Returns")
        plt.title("ES Prediction vs Actual (Calibrated)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print comprehensive results
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Hit rate: {hit_rate:.4f} (Expected: {threshold_alpha})")
    print(f"Number of breaches: {x} out of {n}")
    print(f"Kupiec Test LR_pof: {LR_pof:.4f}, p-value: {p_value:.4f}")
    print(f"Test Result: {'REJECT' if p_value < 0.05 else 'ACCEPT'} null hypothesis")
    print(f"Average Actual Return under VaR: {avg_actual_es:.4f}")
    print(f"Average Predicted ES: {avg_predicted_es:.4f}")
    print(f"Empirical VaR (5%): {var_quantile:.4f}")
    print(f"Empirical ES (5%): {es_quantile:.4f}")
    print(f"ES Constraint Violations: {np.sum(predicted_var < predicted_es)}")
    print(f"ES Calibration Factor Applied: {es_calibration_factor:.4f}")
    print("=" * 60)

    # ES Residual Error Histogram
    if len(breach_returns) > 0:
        residuals = predicted_es[hits == 1] - y_true[hits == 1]
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=15, color="steelblue", edgecolor="black")
        plt.title("Histogram of ES Prediction Errors (Predicted - Actual)")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def multi_alpha_evaluation(model, X_tensor, y_tensor, alphas=[0.01, 0.025, 0.05, 0.1]):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
    y_true = y_tensor.cpu().numpy()

    results = []

    for alpha in alphas:
        predicted_var = y_pred[:, 0]
        hits = (y_true <= predicted_var).astype(int)
        hit_rate = hits.mean()
        n = len(y_true)
        x = hits.sum()

        p_hat = x / n
        if p_hat == 0 or p_hat == 1:
            LR_pof = float("inf")
            p_value = 0.0
        else:
            LR_pof = -2 * (
                np.log((1 - alpha) ** (n - x) * alpha**x)
                - np.log((1 - p_hat) ** (n - x) * p_hat**x)
            )
            p_value = 1 - chi2.cdf(LR_pof, df=1)

        results.append((alpha, hit_rate, x, LR_pof, p_value))

    df_results = pd.DataFrame(
        results, columns=["Alpha", "Hit Rate", "Breaches", "LR_pof", "p-value"]
    )

    print("\nMulti-Alpha VaR Evaluation")
    print(df_results.to_string(index=False))

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(df_results["Alpha"], df_results["Hit Rate"], marker="o", label="Hit Rate")
    plt.plot(
        df_results["Alpha"],
        df_results["Alpha"],
        linestyle="--",
        color="red",
        label="Target",
    )
    plt.xlabel("Alpha Level")
    plt.ylabel("Hit Rate")
    plt.title("VaR Hit Rate vs Alpha")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_with_overlapping_evaluate_with_expanding(
    X, y, input_dim, max_epochs=200, batch_size=64, patience=20, seq_len=21, overlap=0.3
):
    """
    Hybrid approach: Train with overlapping sequences, evaluate with expanding window.
    This combines the benefits of both approaches.
    """
    print("=" * 60)
    print("HYBRID APPROACH: OVERLAPPING TRAINING + EXPANDING EVALUATION")
    print("=" * 60)

    # Step 1: Create overlapping sequences for training
    print("\n1. Creating overlapping sequences for training...")
    X_seq, y_seq = create_sequences_with_overlap(X, y, seq_len=seq_len, overlap=overlap)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    # Split data
    train_size = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    print(f"Training with {len(X_seq)} overlapping sequences...")
    model = BasicVaRTransformer(input_dim=input_dim)
    loss_fn = BasicFZ0Loss(alpha=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.8)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_batch, y_pred)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

        scheduler.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_val_pred = model(X_val)
                val_loss = loss_fn(y_val, y_val_pred)
                if torch.isfinite(val_loss):
                    total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "model/saved_models/var_es_transformer_hybrid.pth")

    # Plot training progress
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress (Hybrid Approach)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("✅ Model trained successfully with overlapping sequences!")

    return model, train_losses, val_losses


def evaluate_frozen_model_with_expanding_window(
    model, X, y, test_start_idx, max_seq_len=100
):
    """
    Evaluate a frozen model using expanding window approach (like GARCH).
    """
    print(f"\n2. Evaluating frozen model with expanding window...")
    print(f"Using expanding window evaluation from index {test_start_idx} to {len(X)}")

    model.eval()
    predicted_var = []
    predicted_es = []

    with torch.no_grad():
        for i in range(test_start_idx, len(X)):
            # Use all data up to current point (expanding window)
            current_data = X[: i + 1]

            # Pad if necessary
            if len(current_data) > max_seq_len:
                current_data = current_data[-max_seq_len:]
            else:
                padding = np.zeros(
                    (max_seq_len - len(current_data), current_data.shape[1])
                )
                current_data = np.vstack([padding, current_data])

            # Make prediction with frozen model
            X_tensor = torch.tensor(
                current_data.reshape(1, -1, current_data.shape[1]), dtype=torch.float32
            )
            prediction = model(X_tensor)

            predicted_var.append(prediction[0, 0].item())
            predicted_es.append(prediction[0, 1].item())

    print(f"✅ Generated {len(predicted_var)} predictions using expanding window")
    return np.array(predicted_var), np.array(predicted_es)


def hybrid_approach_main():
    """
    Main function for the hybrid approach: train with overlapping, evaluate with expanding.
    """
    print("=" * 60)
    print("HYBRID APPROACH IMPLEMENTATION")
    print("=" * 60)

    # Load and preprocess data
    df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")

    df["return"] = df["close"].pct_change()
    df["squared_return"] = df["return"] ** 2
    df["target_return"] = df["return"].shift(-1)
    # df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    # df["return_ma"] = df["return"].rolling(window=5).mean()
    # df["vol_ma"] = df["return"].rolling(window=21).std()

    df.dropna(inplace=True)

    features = [
        "return",
        "squared_return",
        "gk_vol_1d",
        "gk_vol_21d",
    ]

    X = df[features].values
    y = df["target_return"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "model/saved_models/scaler.pkl")

    # Step 1: Train with overlapping sequences
    model, train_losses, val_losses = train_with_overlapping_evaluate_with_expanding(
        X_scaled, y, input_dim=X_scaled.shape[1]
    )

    # Step 2: Evaluate with expanding window
    test_start_idx = int(0.85 * len(X_scaled))
    predicted_var, predicted_es = evaluate_frozen_model_with_expanding_window(
        model, X_scaled, y, test_start_idx
    )

    y_test = y[test_start_idx:]

    # Step 3: Calibrate predictions
    print("\n3. Calibrating predictions...")

    # Create calibration data from training set
    X_train_cal = X_scaled[:test_start_idx]
    y_train_cal = y[:test_start_idx]

    # Create overlapping sequences for calibration
    X_seq_cal, y_seq_cal = create_sequences_with_overlap(
        X_train_cal, y_train_cal, seq_len=21, overlap=0.3
    )
    X_tensor_cal = torch.tensor(X_seq_cal, dtype=torch.float32)
    y_tensor_cal = torch.tensor(y_seq_cal, dtype=torch.float32)

    # # Apply calibration
    var_adjustment_factor = calibrate_var_post_training(
        model, X_tensor_cal, y_tensor_cal
    )
    es_calibration_factor = calibrate_es_post_training(
        model, X_tensor_cal, y_tensor_cal
    )

    # Apply calibration factors
    predicted_var = predicted_var * var_adjustment_factor
    predicted_es = predicted_es * es_calibration_factor

    # Step 4: Evaluate results
    print("\n4. Evaluating hybrid approach results...")
    print("=" * 60)
    print("HYBRID APPROACH EVALUATION")
    print("=" * 60)

    # Calculate metrics
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
        print(f"Average Predicted ES: {predicted_es_breaches:.4f}")

    print(f"VaR Calibration Factor: {var_adjustment_factor:.4f}")
    print(f"ES Calibration Factor: {es_calibration_factor:.4f}")
    print("=" * 60)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(y_test, label="Actual Returns", alpha=0.7)
    plt.plot(predicted_var, label="Hybrid VaR (5%)", color="red", linewidth=1)
    if x > 0:
        breach_indices = np.where(hits == 1)[0]
        plt.scatter(
            breach_indices,
            y_test[breach_indices],
            color="black",
            label=f"Breaches ({x})",
            s=30,
            zorder=5,
        )
    plt.legend()
    plt.title("Hybrid Approach: VaR Backtesting")
    plt.grid(True, alpha=0.3)

    # Rolling hit rate
    plt.subplot(2, 2, 2)
    rolling_hit = pd.Series(hits).rolling(window=20).mean()
    plt.plot(rolling_hit, label="Rolling Hit Rate (20 days)")
    plt.axhline(y=0.05, color="red", linestyle="--", label="Expected (5%)")
    plt.title("Rolling Hit Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Q-Q plot of breach returns
    plt.subplot(2, 2, 3)
    if x > 0:
        breach_returns = y_test[hits == 1]
        sorted_breaches = np.sort(breach_returns)
        quantiles = np.linspace(0, 1, len(sorted_breaches))
        plt.scatter(quantiles, sorted_breaches, alpha=0.7)
        plt.title("Q-Q Plot of Breach Returns")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Actual Returns")
        plt.grid(True, alpha=0.3)

    # ES prediction vs actual
    plt.subplot(2, 2, 4)
    if x > 0:
        plt.scatter(predicted_es[hits == 1], breach_returns, alpha=0.7)
        plt.plot(
            [predicted_es[hits == 1].min(), predicted_es[hits == 1].max()],
            [predicted_es[hits == 1].min(), predicted_es[hits == 1].max()],
            "r--",
            label="Perfect Prediction",
        )
        plt.xlabel("Predicted ES")
        plt.ylabel("Actual Returns")
        plt.title("ES Prediction vs Actual (Calibrated)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary comparison
    print("\nHYBRID APPROACH SUMMARY:")
    print("-" * 30)
    print("✅ Training: Overlapping sequences (more data, better learning)")
    print("✅ Evaluation: Expanding window (like GARCH)")
    print("✅ Calibration: Automatic VaR and ES adjustment")
    print(f"✅ Hit Rate: {hit_rate:.4f} (Target: 0.05)")
    print(f"✅ Model Performance: {'GOOD' if p_value >= 0.05 else 'NEEDS IMPROVEMENT'}")

    return model, predicted_var, predicted_es, hit_rate, p_value


def main():

    # Uncomment the next line to run hybrid approach
    hybrid_approach_main()


if __name__ == "__main__":
    main()
