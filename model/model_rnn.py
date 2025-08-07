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


class RNNVaRES(nn.Module):
    def __init__(
        self, input_dim, target_std=0.02, hidden_dim=64, num_layers=2, dropout=0.2
    ):
        super().__init__()
        self.register_buffer("scale", torch.tensor(target_std).float())

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.output(h_n[-1])

        # Improved VaR and ES calculation for better numerical stability
        var = -torch.exp(
            torch.clamp(out[:, 0], min=-5.0, max=5.0)
        )  # Clamp to prevent extreme values
        es_offset = (
            torch.exp(torch.clamp(out[:, 1], min=-5.0, max=5.0)) + 0.01
        )  # Ensure positive offset
        es = var - es_offset  # Ensure ES is more negative than VaR

        return torch.stack([var, es], dim=1)


class FZ0Loss(nn.Module):
    def __init__(self, alpha=0.05, lambda_reg=0.2, es_penalty_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.es_penalty_weight = es_penalty_weight

    def forward(self, y_true, y_pred):
        var, es = y_pred[:, 0], y_pred[:, 1]
        eps = 1e-6  # Increased epsilon for better numerical stability

        # Base FZ0 terms with improved numerical stability
        indicator = (y_true <= var).float()

        # Avoid division by zero and handle edge cases
        safe_es = torch.clamp(
            es, min=-10.0, max=-0.001
        )  # Ensure ES is negative and not too extreme

        term1 = -(1 / (self.alpha * safe_es + eps)) * indicator * (var - y_true)
        term2 = (var / (safe_es + eps)) + torch.log(-safe_es + eps) - 1
        fz0_loss = torch.mean(term1 + term2)

        # Stronger ES < VaR constraint
        constraint_penalty = torch.mean(torch.relu(var - es)) * 5.0  # Increased penalty

        # Enhanced ES penalty for breaches
        breach_mask = indicator.bool()
        if breach_mask.sum() > 0:
            actual_breach_severity = y_true[breach_mask]
            predicted_es_breach = es[breach_mask]
            # Penalty when ES is not extreme enough during breaches
            es_underestimation_penalty = torch.mean(
                torch.relu(predicted_es_breach - actual_breach_severity)
            )
        else:
            es_underestimation_penalty = torch.tensor(0.0)

        # Coverage rate penalty to encourage proper VaR coverage
        hit_rate = torch.mean(indicator)
        coverage_penalty = torch.abs(hit_rate - self.alpha) * 10.0

        total_loss = (
            fz0_loss
            + self.lambda_reg * constraint_penalty
            + coverage_penalty
            + self.es_penalty_weight * es_underestimation_penalty
        )

        return total_loss


class ImprovedVaRLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        var, es = y_pred[:, 0], y_pred[:, 1]

        # FIX 2: Improved quantile loss that encourages proper coverage
        residual = y_true - var
        var_loss = torch.mean(
            torch.max(self.alpha * residual, (self.alpha - 1) * residual)
        )

        # FIX 3: Stronger penalty for incorrect coverage rate
        breaches = (y_true <= var).float()
        actual_coverage = torch.mean(breaches)
        coverage_penalty = 100.0 * torch.abs(
            actual_coverage - self.alpha
        )  # Increased from 50.0

        # ES loss - only apply to actual breaches
        breach_severity = torch.abs(y_true - var) * breaches
        es_target = var - breach_severity
        es_loss = (
            torch.mean(torch.abs(es - es_target) * breaches)
            if torch.sum(breaches) > 0
            else 0.0
        )

        # Ensure ES is more negative than VaR
        constraint_penalty = torch.mean(torch.relu(var - es)) * 20.0

        total_loss = var_loss + es_loss + constraint_penalty + coverage_penalty
        return total_loss


def debug_data_distribution(y):
    print(f"Return statistics:")
    print(f"Mean: {np.mean(y):.6f}")
    print(f"Std: {np.std(y):.6f}")
    print(f"Min: {np.min(y):.6f}")
    print(f"Max: {np.max(y):.6f}")
    print(f"5% quantile: {np.quantile(y, 0.05):.6f}")
    print(f"95% quantile: {np.quantile(y, 0.95):.6f}")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=50, alpha=0.7)
    plt.title("Return Distribution")
    plt.xlabel("Returns")
    plt.subplot(1, 2, 2)
    plt.plot(y)
    plt.title("Return Time Series")
    plt.xlabel("Time")
    plt.show()


def train_with_debugging(X_seq, y_seq, input_dim, max_epochs=150):
    debug_data_distribution(y_seq)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    train_size = int(0.7 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    target_std = y_tensor.std().item()
    model = RNNVaRES(input_dim=input_dim, target_std=target_std)

    # Optional: initialize model biases closer to empirical VaR/ES
    empirical_var = np.quantile(y_seq, 0.05)
    empirical_es = y_seq[y_seq <= empirical_var].mean()

    # Better initialization with bounds checking
    if np.isfinite(empirical_var) and np.isfinite(empirical_es) and empirical_var < 0:
        target_var_output = np.log(-empirical_var)
        target_es_output = np.log(
            max(0.01, empirical_var - empirical_es)
        )  # Ensure positive
    else:
        # Fallback initialization
        target_var_output = -2.0  # Roughly -0.135 VaR
        target_es_output = 0.5  # Roughly 0.65 offset

    with torch.no_grad():
        model.output[-1].bias[0] = torch.tensor(target_var_output, dtype=torch.float32)
        model.output[-1].bias[1] = torch.tensor(target_es_output, dtype=torch.float32)
        model.output[-1].weight.data *= 0.1  # Start with small weights

    print(
        f"Initialized model to target VaR: {empirical_var:.4f}, ES: {empirical_es:.4f}"
    )

    loss_fn = FZ0Loss(alpha=0.05, lambda_reg=0.2, es_penalty_weight=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )  # Better optimizer
    scheduler = StepLR(optimizer, step_size=30, gamma=0.8)  # Slower decay

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

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

        if epoch % 10 == 0 or epoch < 5:
            with torch.no_grad():
                sample_pred = model(X_tensor[:100])
                sample_var = sample_pred[:, 0].detach().numpy()
                sample_y = y_tensor[:100].numpy()
                coverage_rate = np.mean(sample_y <= sample_var)

            print(
                f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )
            print(f"Coverage Rate: {coverage_rate:.4f} (Target: 0.05)")

            if epoch % 20 == 0:
                print(f"Sample VaR predictions: {sample_var[:5]}")
                print(f"Sample ES predictions: {sample_pred[:5, 1].detach().numpy()}")
                print(f"Actual 5% quantile: {np.quantile(sample_y, 0.05):.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 25:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model, X_tensor, y_tensor


def check_model_sanity(model, X_sample, y_sample):
    """Check if model predictions are reasonable"""
    model.eval()
    with torch.no_grad():
        pred = model(X_sample[:10]).numpy()

    var_pred = pred[:, 0]
    es_pred = pred[:, 1]

    print(f"Sanity Check:")
    print(f"VaR range: [{var_pred.min():.4f}, {var_pred.max():.4f}]")
    print(f"ES range: [{es_pred.min():.4f}, {es_pred.max():.4f}]")
    print(f"Data range: [{y_sample.min():.4f}, {y_sample.max():.4f}]")
    print(f"Data 5% quantile: {np.quantile(y_sample, 0.05):.4f}")

    # Check if predictions are in reasonable range
    data_5pct = np.quantile(y_sample, 0.05)
    if (
        var_pred.mean() > data_5pct * 0.5 and var_pred.mean() < data_5pct * 2.0
    ):  # More reasonable range
        print("✓ VaR predictions seem reasonable")
    else:
        print("✗ VaR predictions are not in reasonable range")
        print(f"  VaR mean: {var_pred.mean():.4f}, Data 5%: {data_5pct:.4f}")

    if np.all(es_pred <= var_pred):
        print("✓ ES constraint satisfied")
    else:
        print("✗ ES constraint violated")


def create_sequences_with_overlap(X, y, seq_len=21, overlap=0.5):
    step = max(1, int(seq_len * (1 - overlap)))
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_len, step):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])  # FIXED: predict next timestep, not current
    return np.array(X_seq), np.array(y_seq)


def main():
    # Data preparation (from your original code)
    df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")

    df["return"] = df["close"].pct_change()
    df["squared_return"] = df["return"] ** 2
    df["target_return"] = df["return"].shift(-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_ma"] = df["return"].rolling(window=5).mean()
    df["vol_ma"] = df["return"].rolling(window=21).std()

    df.dropna(inplace=True)

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

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "model/saved_models/scaler.pkl")

    print("Data preparation complete. Checking target variable...")
    debug_data_distribution(y)

    # Create sequences
    X_seq, y_seq = create_sequences_with_overlap(X_scaled, y, seq_len=21, overlap=0.3)

    # Use more data for testing
    split_idx = int(0.8 * len(X_seq))
    X_train_val, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train_val, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"Training samples: {len(X_train_val)}, Test samples: {len(X_test)}")

    # Train model
    model, _, _ = train_with_debugging(
        X_train_val, y_train_val, input_dim=X_seq.shape[2], max_epochs=150
    )

    # Check model before evaluation
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    check_model_sanity(model, X_test_tensor, y_test)

    # Save model
    torch.save(model.state_dict(), "model/saved_models/var_es_rnn_weights.pth")

    # Apply ES calibration (simplified version)
    es_calibration_factor = calibrate_es_post_training(
        model, X_test_tensor, y_test_tensor
    )

    # Comprehensive evaluation
    comprehensive_evaluation(
        model, X_test_tensor, y_test_tensor, es_calibration_factor=es_calibration_factor
    )

    # Multi-alpha evaluation
    multi_alpha_evaluation(model, X_test_tensor, y_test_tensor)


# Add the calibration functions (simplified versions)
def calibrate_es_post_training(model, X_tensor, y_tensor, alpha=0.05):
    """Simplified ES calibration"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    y_true = y_tensor.cpu().numpy()
    var_pred = predictions[:, 0]

    # Find actual breaches
    breaches = y_true <= var_pred
    if breaches.sum() > 0:
        actual_es = y_true[breaches].mean()
        # Calculate empirical ES from historical data
        empirical_var = np.quantile(y_true, alpha)
        empirical_es = y_true[y_true <= empirical_var].mean()

        if actual_es != 0:
            es_scaling_factor = max(1.0, empirical_es / actual_es)
            print(f"ES calibration factor: {es_scaling_factor:.4f}")
            return es_scaling_factor

    return 1.0


def comprehensive_evaluation(
    model, X_tensor, y_tensor, threshold_alpha=0.05, es_calibration_factor=1.0
):
    """Simplified evaluation function"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
    y_true = y_tensor.cpu().numpy()
    predicted_var = y_pred[:, 0]
    predicted_es = y_pred[:, 1] * es_calibration_factor

    # Basic statistics
    hits = (y_true <= predicted_var).astype(int)
    hit_rate = hits.mean()
    n = len(y_true)
    x = hits.sum()

    # Kupiec Test
    p_hat = x / n if x > 0 else 0.001
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

    # Print results
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Hit rate: {hit_rate:.4f} (Expected: {threshold_alpha})")
    print(f"Number of breaches: {x} out of {n}")
    print(f"Kupiec Test LR_pof: {LR_pof:.4f}, p-value: {p_value:.4f}")
    print(f"Test Result: {'REJECT' if p_value < 0.05 else 'ACCEPT'} null hypothesis")
    print(f"Average Actual Return under VaR: {avg_actual_es:.4f}")
    print(f"Average Predicted ES: {avg_predicted_es:.4f}")
    print(f"Empirical VaR (5%): {np.quantile(y_true, threshold_alpha):.4f}")
    print(
        f"Empirical ES (5%): {y_true[y_true <= np.quantile(y_true, threshold_alpha)].mean():.4f}"
    )
    print(f"ES Constraint Violations: {np.sum(predicted_var < predicted_es)}")
    print(f"ES Calibration Factor Applied: {es_calibration_factor:.4f}")
    print("=" * 60)

    # Simple visualization
    plt.figure(figsize=(16, 10))

    # 1. VaR Backtesting
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

    # 2. Rolling Hit Rate (with adaptive window)
    plt.subplot(2, 2, 2)
    adaptive_window = min(20, max(5, len(hits) // 3))  # At least 5, at most 20
    rolling_hit_rate = pd.Series(hits).rolling(window=adaptive_window).mean()
    plt.plot(rolling_hit_rate, label=f"Rolling Hit Rate ({adaptive_window} days)")
    plt.axhline(
        y=threshold_alpha,
        color="red",
        linestyle="--",
        label=f"Expected ({threshold_alpha})",
    )
    plt.title("Rolling Hit Rate")
    plt.legend()
    plt.grid(True)

    # 3. Q-Q Plot
    plt.subplot(2, 2, 3)
    if len(breach_returns) > 0:
        sorted_breaches = np.sort(breach_returns)
        theoretical_quantiles = np.linspace(0, 1, len(sorted_breaches))
        plt.scatter(theoretical_quantiles, sorted_breaches, alpha=0.7)
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Breach Returns")
        plt.title("Q-Q Plot of Breach Returns")
        plt.grid(True)

    # 4. ES Prediction vs Actual
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


def multi_alpha_evaluation(model, X_tensor, y_tensor, alphas=[0.01, 0.025, 0.05, 0.1]):
    """Simplified multi-alpha evaluation"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
    y_true = y_tensor.cpu().numpy()

    results = []
    predicted_var = y_pred[:, 0]

    for alpha in alphas:
        hits = (y_true <= predicted_var).astype(int)
        hit_rate = hits.mean()
        n = len(y_true)
        x = hits.sum()

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

        results.append((alpha, hit_rate, x, LR_pof, p_value))

    df_results = pd.DataFrame(
        results, columns=["Alpha", "Hit Rate", "Breaches", "LR_pof", "p-value"]
    )

    print("\nMulti-Alpha VaR Evaluation")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
