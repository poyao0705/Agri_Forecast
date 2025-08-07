import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, t, chi2

# Load and preprocess data
df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")
df["return"] = df["close"].pct_change()
df["target_return"] = df["return"].shift(-1)
df.dropna(inplace=True)

returns = df["target_return"].values

# Train/test split
split = int(len(returns) * 0.8)
train, test = returns[:split], returns[split:]

print(f"Training samples: {len(train)}, Test samples: {len(test)}")

# Fit GARCH(1,1) model (Normal dist; for t-distribution use: dist="t")
garch_model = arch_model(train, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
model_fit = garch_model.fit(disp="off")

print("GARCH model fitted successfully")
print(model_fit.summary())

# Rolling forecast approach
predicted_var = []
predicted_es = []
mu_list = []
sigma_list = []

alpha = 0.05
z = norm.ppf(alpha)
c = norm.pdf(z) / alpha

print("Generating forecasts...")

for i in range(len(test)):
    # Use data up to current point for forecasting
    current_data = returns[: split + i]

    # Refit model (or use expanding window)
    temp_model = arch_model(
        current_data, vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    temp_fit = temp_model.fit(disp="off")

    # Generate 1-step ahead forecast
    forecast = temp_fit.forecast(horizon=1)

    mu = forecast.mean.iloc[-1, 0]  # Get the forecast mean
    sigma = np.sqrt(forecast.variance.iloc[-1, 0])  # Get the forecast variance

    mu_list.append(mu)
    sigma_list.append(sigma)

    # Calculate VaR and ES
    var_forecast = mu + sigma * z
    es_forecast = mu - sigma * c

    predicted_var.append(var_forecast)
    predicted_es.append(es_forecast)

# Convert to numpy arrays
predicted_var = np.array(predicted_var)
predicted_es = np.array(predicted_es)
mu_array = np.array(mu_list)
sigma_array = np.array(sigma_list)

print(f"Generated {len(predicted_var)} forecasts")

# Evaluate predictions
actual = test
hits = (actual <= predicted_var).astype(int)
hit_rate = hits.mean()
n = len(test)
x = hits.sum()
breach_returns = actual[hits == 1]


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

# ES evaluation - only compute when there are breaches
if x > 0:
    actual_es = actual[hits == 1].mean()
    predicted_es_breaches = predicted_es[hits == 1].mean()
else:
    actual_es = np.nan
    predicted_es_breaches = np.nan

# Print results
print("=" * 60)
print("GARCH(1,1)-BASED VaR/ES EVALUATION")
print("=" * 60)
print(f"Hit rate: {hit_rate:.4f} (Expected: {alpha})")
print(f"Number of breaches: {x} out of {n}")
print(f"Kupiec Test LR_pof: {LR_pof:.4f}, p-value: {p_value:.4f}")
print(f"Test Result: {'REJECT' if p_value < 0.05 else 'ACCEPT'} null hypothesis")
if not np.isnan(actual_es):
    print(f"Average Actual Return under VaR: {actual_es:.4f}")
    print(f"Predicted ES (at breach points): {predicted_es_breaches:.4f}")
else:
    print("No VaR breaches occurred - cannot compute ES")
print("=" * 60)

# Additional statistics
print(f"Mean predicted volatility: {sigma_array.mean():.4f}")
print(f"Mean predicted return: {mu_array.mean():.4f}")
print(f"Actual test return mean: {actual.mean():.4f}")
print(f"Actual test return std: {actual.std():.4f}")

# Plotting
plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 1, 1)
plt.plot(actual, label="Actual Returns", alpha=0.7)
plt.plot(predicted_var, label="GARCH VaR (5%)", color="red", linewidth=1)
if x > 0:
    breach_indices = np.where(hits == 1)[0]
    plt.scatter(
        breach_indices,
        actual[breach_indices],
        color="black",
        label=f"Breaches ({x})",
        s=30,
        zorder=5,
    )
plt.legend()
plt.title("GARCH VaR Forecast vs Actual Returns")
plt.grid(True, alpha=0.3)
plt.ylabel("Returns")

# Volatility plot
plt.subplot(2, 1, 2)
plt.plot(sigma_array, label="Predicted Volatility", color="blue")
plt.title("GARCH Predicted Volatility")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 1. VaR Backtesting (zoomed or detailed)
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(actual, label="Actual Returns", alpha=0.6)
plt.plot(predicted_var, label="GARCH VaR (5%)", color="red")
plt.scatter(
    np.where(hits)[0],
    breach_returns,
    color="black",
    label="Breaches",
    s=20,
    zorder=5,
)
plt.title("VaR Backtesting")
plt.legend()
plt.grid(True)

# 2. Rolling Hit Rate
plt.subplot(2, 2, 2)
rolling_hit = pd.Series(hits).rolling(window=20).mean()
plt.plot(rolling_hit, label="Rolling Hit Rate (20 days)")
plt.axhline(y=alpha, color="red", linestyle="--", label="Expected (5%)")
plt.title("Rolling Hit Rate")
plt.legend()
plt.grid(True)

# 3. Q-Q Plot of Breach Returns
plt.subplot(2, 2, 3)
sorted_breaches = np.sort(breach_returns)
quantiles = np.linspace(0, 1, len(sorted_breaches))
plt.scatter(quantiles, sorted_breaches, alpha=0.7)
plt.title("Q-Q Plot of Breach Returns")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Actual Returns")

# 4. ES Prediction vs Actual
plt.subplot(2, 2, 4)
plt.scatter(predicted_es[hits == 1], breach_returns, alpha=0.6)
plt.plot(
    [min(predicted_es), max(predicted_es)],
    [min(predicted_es), max(predicted_es)],
    "r--",
    label="Perfect Fit",
)
plt.xlabel("Predicted ES")
plt.ylabel("Actual Returns")
plt.title("ES Prediction vs Actual")
plt.legend()

plt.tight_layout()
plt.show()

# Summary statistics
print("\nSummary Statistics:")
print(f"VaR Coverage: {hit_rate:.2%} (Target: {alpha:.2%})")
print(f"Kupiec test p-value: {p_value:.4f}")
# Empirical VaR and ES
empirical_var = np.quantile(actual, alpha)
empirical_es = actual[actual <= empirical_var].mean()

# Constraint violations (always 0 in GARCH)
es_constraint_violations = np.sum(predicted_var < predicted_es)

print(f"Empirical VaR (5%): {empirical_var:.4f}")
print(f"Empirical ES (5%): {empirical_es:.4f}")
print(f"ES Constraint Violations: {es_constraint_violations}")
print(f"ES Calibration Factor Applied: 1.0000")

if p_value < 0.05:
    print("⚠️  VaR model is rejected at 5% significance level")
else:
    print("✅ VaR model is not rejected at 5% significance level")
