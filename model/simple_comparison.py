import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


class RobustRNNModel:
    """Robust RNN model with proper VaR constraints"""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, sequence_length=21):
        """Build RNN model with proper output scaling"""
        model = Sequential(
            [
                SimpleRNN(
                    units=16,
                    activation="tanh",
                    dropout=0.1,
                    return_sequences=False,
                    input_shape=(sequence_length, 1),
                ),
                Dense(8, activation="relu"),
                Dropout(0.1),
                Dense(1, activation="linear"),  # Single VaR output
            ]
        )

        def robust_quantile_loss(y_true, y_pred):
            # Simple quantile loss without ES complications
            var_pred = y_pred[:, 0]

            # Don't scale here - let the model learn the right scale
            # var_pred should be negative for losses

            # Quantile loss
            indicator = tf.cast(y_true <= var_pred, tf.float32)
            loss = tf.reduce_mean((self.alpha - indicator) * (var_pred - y_true))

            return loss

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001
            ),  # Lower learning rate
            loss=robust_quantile_loss,
        )

        # Initialize the final layer bias to empirical quantile
        model.layers[-1].set_weights(
            [
                model.layers[-1].get_weights()[0],  # Keep weights
                np.array([-0.03]),  # Set bias to empirical 5% quantile
            ]
        )

        self.model = model
        return model

    def prepare_data(self, returns, sequence_length=21):
        """Prepare data using paper's method"""
        # Use squared returns as input
        squared_returns = returns**2
        squared_returns_scaled = self.scaler.fit_transform(
            squared_returns.reshape(-1, 1)
        ).flatten()

        X, y = [], []
        for i in range(len(squared_returns_scaled) - sequence_length):
            X.append(squared_returns_scaled[i : i + sequence_length])
            y.append(returns[i + sequence_length])

        return np.array(X), np.array(y)

    def train(self, returns, epochs=100, batch_size=32, sequence_length=21):
        """Train the model"""
        X, y = self.prepare_data(returns, sequence_length)
        X = X.reshape(-1, sequence_length, 1)

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.build_model(sequence_length)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        print(f"📊 RNN Training: {X_train.shape} -> {y_train.shape}")

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0,
        )

        return history

    def predict(self, returns, sequence_length=21):
        """Make predictions"""
        squared_returns = (returns**2).reshape(-1, 1)
        squared_returns_scaled = self.scaler.transform(squared_returns).flatten()

        X_test = []
        for i in range(sequence_length, len(squared_returns_scaled)):
            X_test.append(squared_returns_scaled[i - sequence_length : i])

        X_test = np.array(X_test).reshape(-1, sequence_length, 1)

        predictions = self.model.predict(X_test, verbose=0)
        var_pred = predictions[:, 0]

        # Don't apply artificial scaling - let model learn natural scale

        return var_pred


class RobustTransformerModel(nn.Module):
    """Simplified Transformer with proper constraints"""

    def __init__(
        self, input_dim=1, model_dim=32, num_heads=2, num_layers=1, dropout=0.1
    ):
        super().__init__()

        self.input_linear = nn.Linear(input_dim, model_dim)

        # Simple positional encoding
        self.register_buffer(
            "pos_encoding", self._get_positional_encoding(model_dim, 100)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(model_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),  # Single VaR output - no artificial scaling
        )

        # Initialize the final layer to predict around the empirical 5% quantile
        with torch.no_grad():
            self.output_layer[-1].weight.normal_(0, 0.01)
            self.output_layer[-1].bias.fill_(-0.03)  # Start near empirical 5% VaR

    def _get_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_linear(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len]

        x = self.transformer(x)
        x = x[:, -1]  # Last timestep

        out = self.output_layer(x)

        # Let the model learn the natural scale
        return out


class SimpleQuantileLoss(nn.Module):
    """Simple, stable quantile loss"""

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        var_pred = y_pred.squeeze()

        indicator = (y_true <= var_pred).float()
        loss = torch.mean((self.alpha - indicator) * (var_pred - y_true))

        return loss


def train_robust_transformer(returns, epochs=100, batch_size=32, sequence_length=21):
    """Train transformer with robust setup"""

    # Prepare data
    scaler = StandardScaler()
    squared_returns = returns**2
    squared_returns_scaled = scaler.fit_transform(
        squared_returns.reshape(-1, 1)
    ).flatten()

    X, y = [], []
    for i in range(len(squared_returns_scaled) - sequence_length):
        X.append(squared_returns_scaled[i : i + sequence_length])
        y.append(returns[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = RobustTransformerModel()
    criterion = SimpleQuantileLoss(alpha=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

    print(f"📊 Transformer Training: {X_train.shape} -> {y_train.shape}")

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_batch, y_pred)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_batch, y_pred)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    return model, scaler, train_losses, val_losses


def predict_robust_transformer(model, scaler, returns, sequence_length=21):
    """Make predictions with transformer"""
    squared_returns = (returns**2).reshape(-1, 1)
    squared_returns_scaled = scaler.transform(squared_returns).flatten()

    X_test = []
    for i in range(sequence_length, len(squared_returns_scaled)):
        X_test.append(squared_returns_scaled[i - sequence_length : i])

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy().squeeze()

    return predictions


def robust_comparison():
    """Robust comparison between RNN and Transformer"""

    print("🚀 ROBUST VaR MODEL COMPARISON")
    print("=" * 50)

    # Load data
    df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")
    df["return"] = df["close"].pct_change()
    df.dropna(inplace=True)
    returns = df["return"].values

    print(f"📊 Data: {len(returns)} observations")
    print(f"📊 Return statistics:")
    print(f"   Mean: {returns.mean():.4f}")
    print(f"   Std:  {returns.std():.4f}")
    print(f"   5% quantile: {np.quantile(returns, 0.05):.4f}")

    # Train models
    print("\n🤖 Training RNN...")
    rnn_model = RobustRNNModel(alpha=0.05)
    rnn_history = rnn_model.train(returns, epochs=50)

    print("\n🤖 Training Transformer...")
    transformer_model, transformer_scaler, trans_train_losses, trans_val_losses = (
        train_robust_transformer(returns, epochs=50)
    )

    # Make predictions
    print("\n📈 Making predictions...")

    var_pred_rnn = rnn_model.predict(returns)
    var_pred_trans = predict_robust_transformer(
        transformer_model, transformer_scaler, returns
    )

    # Align data
    sequence_length = 21
    min_len = min(len(var_pred_rnn), len(var_pred_trans))
    returns_test = returns[sequence_length : sequence_length + min_len]
    var_pred_rnn = var_pred_rnn[:min_len]
    var_pred_trans = var_pred_trans[:min_len]

    print(f"📊 Evaluation: {len(returns_test)} observations")
    print(f"   RNN VaR range: [{var_pred_rnn.min():.4f}, {var_pred_rnn.max():.4f}]")
    print(
        f"   Transformer VaR range: [{var_pred_trans.min():.4f}, {var_pred_trans.max():.4f}]"
    )

    # Calculate hit rates
    hits_rnn = (returns_test <= var_pred_rnn).astype(int)
    hits_trans = (returns_test <= var_pred_trans).astype(int)

    hit_rate_rnn = hits_rnn.mean()
    hit_rate_trans = hits_trans.mean()

    print(f"\n🎯 RESULTS:")
    print("=" * 30)
    print(f"RNN Hit Rate:         {hit_rate_rnn:.1%}")
    print(f"Transformer Hit Rate: {hit_rate_trans:.1%}")
    print(f"Target Hit Rate:      5.0%")
    print(f"RNN Breaches:         {hits_rnn.sum()}/{len(hits_rnn)}")
    print(f"Transformer Breaches: {hits_trans.sum()}/{len(hits_trans)}")

    # Check if models are working
    empirical_var = np.quantile(returns_test, 0.05)
    print(f"\n📊 Reference:")
    print(f"Empirical 5% VaR: {empirical_var:.4f}")
    print(f"RNN Mean VaR:     {var_pred_rnn.mean():.4f}")
    print(f"Transformer Mean VaR: {var_pred_trans.mean():.4f}")

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot 1: RNN Results
    plt.subplot(2, 3, 1)
    plt.plot(returns_test[:500], label="Actual Returns", alpha=0.7, linewidth=0.8)
    plt.plot(var_pred_rnn[:500], label="RNN VaR (5%)", color="red", linewidth=1)
    plt.axhline(
        y=empirical_var, color="green", linestyle="--", label="Empirical VaR", alpha=0.7
    )
    if hits_rnn.sum() > 0:
        breach_idx = np.where(hits_rnn[:500] == 1)[0]
        plt.scatter(
            breach_idx,
            returns_test[breach_idx],
            color="black",
            s=15,
            label="Breaches",
            zorder=5,
        )
    plt.title(f"RNN: Hit Rate = {hit_rate_rnn:.1%}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Transformer Results
    plt.subplot(2, 3, 2)
    plt.plot(returns_test[:500], label="Actual Returns", alpha=0.7, linewidth=0.8)
    plt.plot(
        var_pred_trans[:500], label="Transformer VaR (5%)", color="blue", linewidth=1
    )
    plt.axhline(
        y=empirical_var, color="green", linestyle="--", label="Empirical VaR", alpha=0.7
    )
    if hits_trans.sum() > 0:
        breach_idx = np.where(hits_trans[:500] == 1)[0]
        plt.scatter(
            breach_idx,
            returns_test[breach_idx],
            color="black",
            s=15,
            label="Breaches",
            zorder=5,
        )
    plt.title(f"Transformer: Hit Rate = {hit_rate_trans:.1%}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Hit Rate Comparison
    plt.subplot(2, 3, 3)
    models = ["RNN", "Transformer", "Target"]
    hit_rates = [hit_rate_rnn, hit_rate_trans, 0.05]
    colors = ["red", "blue", "green"]

    bars = plt.bar(models, hit_rates, color=colors, alpha=0.7)
    plt.axhline(y=0.05, color="green", linestyle="--", alpha=0.5)
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate Comparison")
    plt.grid(True, alpha=0.3)

    for bar, rate in zip(bars, hit_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
        )

    # Plot 4: Training curves
    plt.subplot(2, 3, 4)
    plt.plot(
        rnn_history.history["loss"], label="RNN Train Loss", color="red", alpha=0.7
    )
    plt.plot(
        rnn_history.history["val_loss"],
        label="RNN Val Loss",
        color="red",
        linestyle="--",
    )
    plt.plot(
        trans_train_losses, label="Transformer Train Loss", color="blue", alpha=0.7
    )
    plt.plot(
        trans_val_losses, label="Transformer Val Loss", color="blue", linestyle="--"
    )
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: VaR distributions
    plt.subplot(2, 3, 5)
    plt.hist(
        var_pred_rnn, bins=30, alpha=0.5, label="RNN VaR", color="red", density=True
    )
    plt.hist(
        var_pred_trans,
        bins=30,
        alpha=0.5,
        label="Transformer VaR",
        color="blue",
        density=True,
    )
    plt.axvline(x=empirical_var, color="green", linestyle="--", label="Empirical VaR")
    plt.title("VaR Prediction Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Scatter plot
    plt.subplot(2, 3, 6)
    plt.scatter(var_pred_rnn, var_pred_trans, alpha=0.5, s=5)
    plt.plot(
        [var_pred_rnn.min(), var_pred_rnn.max()],
        [var_pred_rnn.min(), var_pred_rnn.max()],
        "r--",
        alpha=0.5,
    )
    plt.xlabel("RNN VaR")
    plt.ylabel("Transformer VaR")
    plt.title("RNN vs Transformer Predictions")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Analysis
    print(f"\n🔍 ANALYSIS:")
    print("=" * 30)

    def evaluate_model(hit_rate, name):
        if 0.03 <= hit_rate <= 0.07:
            return f"✅ {name} performs well"
        elif hit_rate < 0.01:
            return f"❌ {name} too conservative (predicting extreme losses)"
        elif hit_rate > 0.10:
            return f"❌ {name} not conservative enough (missing tail risk)"
        else:
            return f"⚠️  {name} needs tuning"

    print(evaluate_model(hit_rate_rnn, "RNN"))
    print(evaluate_model(hit_rate_trans, "Transformer"))

    # Winner
    rnn_error = abs(hit_rate_rnn - 0.05)
    trans_error = abs(hit_rate_trans - 0.05)

    if rnn_error < trans_error:
        print(f"\n🏆 WINNER: RNN (closer to 5% target)")
    elif trans_error < rnn_error:
        print(f"\n🏆 WINNER: Transformer (closer to 5% target)")
    else:
        print(f"\n🏆 RESULT: Tie")

    return rnn_model, transformer_model, hit_rate_rnn, hit_rate_trans


if __name__ == "__main__":
    rnn_model, transformer_model, hit_rate_rnn, hit_rate_trans = robust_comparison()
