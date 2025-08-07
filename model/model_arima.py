import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2

# Load your data (must include 'target_return' column)
df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")
df["return"] = df["close"].pct_change()
df["target_return"] = df["return"].shift(-1)
df.dropna(inplace=True)

returns = df["target_return"].values

# Split train/test
split = int(len(returns) * 0.8)
train, test = returns[:split], returns[split:]

# Fit ARIMA on training data (tune order as needed)
model = ARIMA(train, order=(1, 0, 1))  # p=1, d=0, q=1 here; try (1,1,1) too
model_fit = model.fit()

# Forecast next-step returns
forecast = model_fit.forecast(steps=len(test))

# Calculate 5% empirical VaR and ES from forecast errors
residuals = model_fit.resid
alpha = 0.05
empirical_var = np.quantile(residuals, alpha)
empirical_es = residuals[residuals <= empirical_var].mean()

# Apply to forecasted mean to get point VaR/ES predictions
predicted_var = forecast + empirical_var
predicted_es = forecast + empirical_es

# Evaluate
actual = test
hits = (actual <= predicted_var).astype(int)
hit_rate = hits.mean()
n = len(test)
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

# ES evaluation
actual_es = actual[hits == 1].mean()

# Print results
print("=" * 60)
print("ARIMA-BASED VaR/ES EVALUATION")
print("=" * 60)
print(f"Hit rate: {hit_rate:.4f} (Expected: {alpha})")
print(f"Number of breaches: {x} out of {n}")
print(f"Kupiec Test LR_pof: {LR_pof:.4f}, p-value: {p_value:.4f}")
print(f"Test Result: {'REJECT' if p_value < 0.05 else 'ACCEPT'} null hypothesis")
print(f"Average Actual Return under VaR: {actual_es:.4f}")
print(f"Predicted ES: {predicted_es[hits == 1].mean():.4f}")
print(f"Empirical Residual VaR (5%): {empirical_var:.4f}")
print(f"Empirical Residual ES (5%): {empirical_es:.4f}")
print("=" * 60)

# Optional: plot
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual Returns")
plt.plot(predicted_var, label="ARIMA VaR (5%)", color="red")
plt.scatter(np.where(hits)[0], actual[hits == 1], color="black", label="Breaches")
plt.legend()
plt.title("ARIMA VaR Forecast vs Actual")
plt.grid(True)
plt.show()
plt.figure(figsize=(16, 10))

# 1. VaR Backtesting
plt.subplot(2, 2, 1)
plt.plot(actual, label="Actual Returns", alpha=0.7)
plt.plot(predicted_var, label="Predicted VaR (5%)", color="red", linewidth=1)
plt.scatter(np.where(hits)[0], actual[hits == 1], color="black", label="Breaches", s=10)
plt.title("VaR Backtesting")
plt.legend()
plt.grid(True)

# 2. Rolling Hit Rate
plt.subplot(2, 2, 2)
rolling_window = min(20, max(5, len(hits) // 3))
rolling_hit_rate = pd.Series(hits).rolling(window=rolling_window).mean()
plt.plot(rolling_hit_rate, label=f"Rolling Hit Rate ({rolling_window} days)")
plt.axhline(y=alpha, color="red", linestyle="--", label=f"Expected ({alpha})")
plt.title("Rolling Hit Rate")
plt.legend()
plt.grid(True)

# 3. Q-Q Plot of Breach Returns
plt.subplot(2, 2, 3)
breach_returns = actual[hits == 1]
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
    min_val = min(predicted_es[hits == 1].min(), breach_returns.min())
    max_val = max(predicted_es[hits == 1].max(), breach_returns.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("Predicted ES")
    plt.ylabel("Actual Returns")
    plt.title("ES Prediction vs Actual (Calibrated)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
