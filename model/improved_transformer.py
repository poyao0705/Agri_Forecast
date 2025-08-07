import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Import base functions
exec(open("model/model_v0.1.py").read())


class ImprovedVaRTransformerV2(nn.Module):
    """
    Improved Transformer model with better architecture for VaR/ES prediction
    """

    def __init__(
        self, input_dim, model_dim=256, num_heads=8, num_layers=4, dropout=0.1
    ):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # More sophisticated output head
        self.output_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 4, 2),
        )

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x[:, -1])  # Use last timestep
        out = self.output_head(x)

        # Better VaR/ES formulation
        var = -torch.exp(out[:, 0])  # VaR is negative
        es_offset = 1.5 * torch.exp(out[:, 1]) + 0.01  # More conservative ES
        es = var - es_offset  # ES is more negative than VaR

        return torch.stack([var, es], dim=1)


class ImprovedFZ0LossV2(nn.Module):
    """
    Improved loss function with better calibration for VaR/ES
    """

    def __init__(self, alpha=0.05, lambda_reg=0.05, es_penalty_weight=1.5):
        super().__init__()
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.es_penalty_weight = es_penalty_weight

    def forward(self, y_true, y_pred):
        var, es = y_pred[:, 0], y_pred[:, 1]
        eps = 1e-8

        # ES < VaR constraint
        constraint_penalty = torch.mean(torch.relu(var - es))

        # Main FZ0 loss
        indicator = (y_true <= var).float()
        term1 = -(1 / (self.alpha * es + eps)) * indicator * (var - y_true)
        term2 = (var / (es + eps)) + torch.log(-es + eps) - 1

        fz0_loss = torch.mean(term1 + term2)

        # Hit rate penalty with target closer to alpha
        hit_rate = torch.mean(indicator)
        target_hit_rate = self.alpha * 1.02  # Very close to target
        hit_rate_penalty = (
            torch.abs(hit_rate - target_hit_rate) * 5.0
        )  # Stronger penalty

        # Penalty for overly conservative predictions
        conservative_penalty = torch.mean(torch.relu(-var - 0.05))  # Less penalty

        # ES underestimation penalty
        breach_mask = indicator.bool()
        if breach_mask.sum() > 0:
            actual_breach_severity = y_true[breach_mask]
            predicted_es_breach = es[breach_mask]
            es_underestimation_penalty = torch.mean(
                torch.relu(predicted_es_breach - actual_breach_severity)
            )
        else:
            es_underestimation_penalty = torch.tensor(0.0)

        total_loss = (
            fz0_loss
            + self.lambda_reg * constraint_penalty
            + hit_rate_penalty
            + 0.05 * conservative_penalty  # Smaller penalty
            + self.es_penalty_weight * es_underestimation_penalty
        )

        return total_loss


def train_improved_transformer(
    X,
    y,
    input_dim,
    max_epochs=300,
    batch_size=128,
    patience=25,
    seq_len=21,
    overlap=0.5,
):
    """
    Train improved transformer with better hyperparameters
    """
    print("=" * 60)
    print("IMPROVED TRANSFORMER TRAINING")
    print("=" * 60)

    # Create overlapping sequences with more overlap
    X_seq, y_seq = create_sequences_with_overlap(X, y, seq_len=seq_len, overlap=overlap)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    # Split data
    train_size = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Use improved model and loss
    model = ImprovedVaRTransformerV2(input_dim=input_dim)
    loss_fn = ImprovedFZ0LossV2(alpha=0.05, lambda_reg=0.05, es_penalty_weight=1.5)

    # Better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=0.5
                )  # Smaller gradient clipping
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

        if epoch % 20 == 0:
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
    torch.save(
        model.state_dict(), "model/saved_models/improved_transformer_weights.pth"
    )

    # Plot training progress
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Improved Transformer Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("✅ Improved Transformer trained successfully!")

    return model, train_losses, val_losses


def evaluate_improved_transformer(model, X, y, test_start_idx, max_seq_len=100):
    """
    Evaluate improved transformer with expanding window
    """
    print(f"\nEvaluating improved transformer with expanding window...")
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

            # Make prediction
            X_tensor = torch.tensor(
                current_data.reshape(1, -1, current_data.shape[1]), dtype=torch.float32
            )
            prediction = model(X_tensor)

            predicted_var.append(prediction[0, 0].item())
            predicted_es.append(prediction[0, 1].item())

    print(f"✅ Generated {len(predicted_var)} predictions")
    return np.array(predicted_var), np.array(predicted_es)


def improved_calibration(model, X_train, y_train):
    """
    Improved calibration approach
    """
    print("\nApplying improved calibration...")

    # Create calibration data
    X_seq_cal, y_seq_cal = create_sequences_with_overlap(
        X_train, y_train, seq_len=21, overlap=0.5
    )
    X_tensor_cal = torch.tensor(X_seq_cal, dtype=torch.float32)
    y_tensor_cal = torch.tensor(y_seq_cal, dtype=torch.float32)

    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor_cal).cpu().numpy()

    y_true = y_tensor_cal.cpu().numpy()
    var_pred = predictions[:, 0]

    # Calculate current hit rate
    hits = (y_true <= var_pred).astype(int)
    current_hit_rate = hits.mean()

    print(f"Current hit rate: {current_hit_rate:.4f} (Target: 0.05)")

    # More conservative calibration
    if current_hit_rate > 0.06:  # If more than 6%
        var_adjustment_factor = 0.85  # Make VaR 15% less negative
        print(
            f"VaR adjustment factor: {var_adjustment_factor:.4f} (making less conservative)"
        )
    elif current_hit_rate > 0.055:  # If more than 5.5%
        var_adjustment_factor = 0.9  # Make VaR 10% less negative
        print(
            f"VaR adjustment factor: {var_adjustment_factor:.4f} (making less conservative)"
        )
    else:
        var_adjustment_factor = 1.0
        print(
            f"VaR adjustment factor: {var_adjustment_factor:.4f} (no adjustment needed)"
        )

    # ES calibration
    breaches = y_true <= var_pred
    actual_es = y_true[breaches].mean() if breaches.sum() > 0 else var_pred.mean()
    empirical_var = np.quantile(y_true, 0.05)
    empirical_es = y_true[y_true <= empirical_var].mean()

    if actual_es != 0:
        es_scaling_factor = max(1.1, empirical_es / actual_es)  # More conservative
        print(f"ES calibration factor: {es_scaling_factor:.4f}")
    else:
        es_scaling_factor = 1.0
        print(f"ES calibration factor: {es_scaling_factor:.4f}")

    return var_adjustment_factor, es_scaling_factor


def run_improved_transformer_comparison():
    """
    Run improved transformer and compare with GARCH
    """
    print("=" * 60)
    print("IMPROVED TRANSFORMER vs GARCH COMPARISON")
    print("=" * 60)

    # Load and preprocess data
    df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")

    df["return"] = df["close"].pct_change()
    df["squared_return"] = df["return"] ** 2
    df["target_return"] = df["return"].shift(-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_ma"] = df["return"].rolling(window=5).mean()
    df["vol_ma"] = df["return"].rolling(window=21).std()

    df.dropna(inplace=True)

    # Use same split as GARCH
    split = int(len(df) * 0.8)  # 80/20 split

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

    print(f"Training samples: {len(train_returns)}")
    print(f"Test samples: {len(test_returns)}")

    # Run GARCH for comparison
    print("\n" + "=" * 60)
    print("GARCH(1,1) MODEL")
    print("=" * 60)

    garch_model = arch_model(
        train_returns, vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    model_fit = garch_model.fit(disp="off")

    # GARCH predictions
    predicted_var_garch = []
    predicted_es_garch = []

    alpha = 0.05
    z = norm.ppf(alpha)
    c = norm.pdf(z) / alpha

    for i in range(len(test_returns)):
        current_data = train_returns[: len(train_returns) + i]
        temp_model = arch_model(
            current_data, vol="GARCH", p=1, q=1, dist="normal", rescale=False
        )
        temp_fit = temp_model.fit(disp="off")
        forecast = temp_fit.forecast(horizon=1)

        mu = forecast.mean.iloc[-1, 0]
        sigma = np.sqrt(forecast.variance.iloc[-1, 0])

        var_forecast = mu + sigma * z
        es_forecast = mu - sigma * c

        predicted_var_garch.append(var_forecast)
        predicted_es_garch.append(es_forecast)

    predicted_var_garch = np.array(predicted_var_garch)
    predicted_es_garch = np.array(predicted_es_garch)

    # Evaluate GARCH
    hits_garch = (test_returns <= predicted_var_garch).astype(int)
    hit_rate_garch = hits_garch.mean()
    n_garch = len(test_returns)
    x_garch = hits_garch.sum()

    p_hat_garch = x_garch / n_garch if x_garch > 0 else 0.001
    if p_hat_garch == 0 or p_hat_garch == 1:
        LR_pof_garch = float("inf")
        p_value_garch = 0.0
    else:
        LR_pof_garch = -2 * (
            np.log((1 - alpha) ** (n_garch - x_garch) * alpha**x_garch)
            - np.log((1 - p_hat_garch) ** (n_garch - x_garch) * p_hat_garch**x_garch)
        )
        p_value_garch = 1 - chi2.cdf(LR_pof_garch, df=1)

    print(f"GARCH Hit rate: {hit_rate_garch:.4f} (Expected: {alpha})")
    print(f"GARCH Kupiec Test p-value: {p_value_garch:.4f}")
    print(f"GARCH Test Result: {'ACCEPT' if p_value_garch >= 0.05 else 'REJECT'}")

    # Run Improved Transformer
    print("\n" + "=" * 60)
    print("IMPROVED TRANSFORMER MODEL")
    print("=" * 60)

    # Train improved transformer
    model, _, _ = train_improved_transformer(
        X_train, y_train, input_dim=X_train.shape[1]
    )

    # Evaluate with expanding window
    predicted_var_trans, predicted_es_trans = evaluate_improved_transformer(
        model,
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        len(X_train),
    )

    # Apply improved calibration
    var_adjustment_factor, es_calibration_factor = improved_calibration(
        model, X_train, y_train
    )

    # Apply calibration
    predicted_var_trans = predicted_var_trans * var_adjustment_factor
    predicted_es_trans = predicted_es_trans * es_calibration_factor

    # Evaluate Transformer
    hits_trans = (y_test <= predicted_var_trans).astype(int)
    hit_rate_trans = hits_trans.mean()
    n_trans = len(y_test)
    x_trans = hits_trans.sum()

    p_hat_trans = x_trans / n_trans if x_trans > 0 else 0.001
    if p_hat_trans == 0 or p_hat_trans == 1:
        LR_pof_trans = float("inf")
        p_value_trans = 0.0
    else:
        LR_pof_trans = -2 * (
            np.log((1 - 0.05) ** (n_trans - x_trans) * 0.05**x_trans)
            - np.log((1 - p_hat_trans) ** (n_trans - x_trans) * p_hat_trans**x_trans)
        )
        p_value_trans = 1 - chi2.cdf(LR_pof_trans, df=1)

    print(f"Transformer Hit rate: {hit_rate_trans:.4f} (Expected: 0.05)")
    print(f"Transformer Kupiec Test p-value: {p_value_trans:.4f}")
    print(f"Transformer Test Result: {'ACCEPT' if p_value_trans >= 0.05 else 'REJECT'}")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

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
            f"{hit_rate_garch:.4f}",
            "0.0500",
            f"{x_garch}",
            f"{n_garch}",
            f"{p_value_garch:.4f}",
            "ACCEPT" if p_value_garch >= 0.05 else "REJECT",
            f"{abs(hit_rate_garch - 0.05):.4f}",
        ],
        "Improved Transformer": [
            f"{hit_rate_trans:.4f}",
            "0.0500",
            f"{x_trans}",
            f"{n_trans}",
            f"{p_value_trans:.4f}",
            "ACCEPT" if p_value_trans >= 0.05 else "REJECT",
            f"{abs(hit_rate_trans - 0.05):.4f}",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    # Determine winner
    garch_error = abs(hit_rate_garch - 0.05)
    trans_error = abs(hit_rate_trans - 0.05)

    if garch_error < trans_error:
        hit_rate_winner = "GARCH"
    elif trans_error < garch_error:
        hit_rate_winner = "Improved Transformer"
    else:
        hit_rate_winner = "Tie"

    if p_value_garch >= 0.05 and p_value_trans >= 0.05:
        validity_winner = "Both Valid"
    elif p_value_garch >= 0.05:
        validity_winner = "GARCH"
    elif p_value_trans >= 0.05:
        validity_winner = "Improved Transformer"
    else:
        validity_winner = "Both Invalid"

    print(f"\nHit Rate Accuracy Winner: {hit_rate_winner}")
    print(f"Statistical Validity Winner: {validity_winner}")

    # Overall winner
    if p_value_garch >= 0.05 and p_value_trans >= 0.05:
        if garch_error < trans_error:
            overall_winner = "GARCH"
        elif trans_error < garch_error:
            overall_winner = "Improved Transformer"
        else:
            overall_winner = "Tie"
    elif p_value_garch >= 0.05:
        overall_winner = "GARCH"
    elif p_value_trans >= 0.05:
        overall_winner = "Improved Transformer"
    else:
        overall_winner = "Both Need Improvement"

    print(f"Overall Winner: {overall_winner}")

    # Plot comparison
    plt.figure(figsize=(15, 10))

    # VaR comparison
    plt.subplot(2, 2, 1)
    plt.plot(y_test, label="Actual Returns", alpha=0.7)
    plt.plot(predicted_var_garch, label="GARCH VaR", color="red", linewidth=1)
    plt.plot(
        predicted_var_trans, label="Improved Transformer VaR", color="blue", linewidth=1
    )
    plt.title("VaR Comparison: GARCH vs Improved Transformer")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Hit rate comparison
    plt.subplot(2, 2, 2)
    rolling_garch = pd.Series(hits_garch).rolling(window=20).mean()
    rolling_trans = pd.Series(hits_trans).rolling(window=20).mean()
    plt.plot(rolling_garch, label="GARCH Hit Rate", color="red")
    plt.plot(rolling_trans, label="Improved Transformer Hit Rate", color="blue")
    plt.axhline(y=0.05, color="black", linestyle="--", label="Expected (5%)")
    plt.title("Rolling Hit Rate Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Performance summary
    plt.subplot(2, 2, 3)
    models = ["GARCH", "Improved Transformer"]
    hit_rates = [hit_rate_garch, hit_rate_trans]
    p_values = [p_value_garch, p_value_trans]

    x_pos = np.arange(len(models))
    plt.bar(x_pos, hit_rates, alpha=0.7, color=["red", "blue"])
    plt.axhline(y=0.05, color="black", linestyle="--", label="Target (5%)")
    plt.xlabel("Model")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate Comparison")
    plt.xticks(x_pos, models)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Error comparison
    plt.subplot(2, 2, 4)
    errors = [garch_error, trans_error]
    plt.bar(x_pos, errors, alpha=0.7, color=["red", "blue"])
    plt.xlabel("Model")
    plt.ylabel("Hit Rate Error")
    plt.title("Hit Rate Error Comparison")
    plt.xticks(x_pos, models)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n🏆 Final Winner: {overall_winner}")
    print("=" * 60)

    return overall_winner


if __name__ == "__main__":
    run_improved_transformer_comparison()
