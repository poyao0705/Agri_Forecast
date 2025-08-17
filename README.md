# Agri-Forecast: Agricultural Commodity Price Forecasting

This project implements and evaluates various models for forecasting Value-at-Risk (VaR) and Expected Shortfall (ES) for agricultural commodity prices using transformer neural networks and classical methods.

## 🚀 Quick Start

### For New Users
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run hybrid prediction (recommended for first-time users)
python hybrid_live_prediction.py --mode auto

# 3. Or use specific modes for different needs
python hybrid_live_prediction.py --mode retrain  # Full retraining
python hybrid_live_prediction.py --mode calibrate  # Fast calibration
```

### For Researchers
```bash
# Run experiments with different configurations
python scripts/run_sim_models.py --dgp garch11_skt --alpha 0.01 --calibrate --seed 42
```

## 📁 Project Structure

```
agri-forecast/
├─ README.md                    # This file
├─ requirements.txt             # Python dependencies
├─ .gitignore                   # Git ignore patterns
├─ hybrid_live_prediction.py    # 🆕 Smart hybrid prediction (main script)
├─ QUICK_START_GUIDE.md         # 🆕 New user guide
├─ PREDICTION_COMPARISON.md     # 🆕 Script comparison guide
├─ data/
│  ├─ raw/                      # Original CSVs (never overwritten)
│  ├─ interim/                  # Simulated CSVs (by seed) or partially processed
│  └─ processed/                # Aligned & feature-built data for training
├─ src/
│  ├─ models/                   # Neural network models
│  │  ├─ transformer_var_es_paper_exact.py  # Main transformer model
│  │  └─ srnn_ve1_paper_exact.py
│  ├─ baselines/                # Classical baseline models
│  │  └─ baseline_classic_var_es.py
│  ├─ utils/                    # Evaluation utilities
│  │  └─ eval_tools.py
│  └─ dgp/                      # Data generating processes (simulators)
│     ├─ __init__.py            # DGP registry and imports
│     ├─ gaussian.py            # IID Gaussian processes
│     ├─ garch.py               # GARCH(1,1) models
│     ├─ stochastic_volatility.py # Stochastic volatility
│     └─ neural_like.py         # Neural network-like processes
├─ scripts/
│  ├─ run_sim_models.py         # Main CLI for running experiments
│  ├─ aggregate_results.py      # Post-run aggregation and analysis
│  └─ reorganize_legacy.py      # (Optional) migrator for old outputs
├─ artifacts/                   # ONE folder per (dgp, alpha, seed, cal, features)
├─ hybrid_predictions/          # 🆕 Prediction outputs
├─ prediction_figures/          # 🆕 Diagnostic plots
├─ results/
│  ├─ tables/                   # Aggregated across many runs
│  └─ figures/                  # Publication figures (nice file names)
└─ reports/
    ├─ thesis/                  # LaTeX/Word documents
    └─ slides/                  # Presentation materials
```

## 🎯 What This Model Does

Your transformer model predicts **next-day risk measures** for financial time series:

- **VaR (Value at Risk)**: Maximum expected loss for the next day
- **ES (Expected Shortfall)**: Average loss when VaR is exceeded

**Example Output:**
```
TOMORROW'S RISK PREDICTION
============================================================
Confidence Level: 99.0%
VaR (Value at Risk): -2.34%
ES (Expected Shortfall): -3.45%
Risk Level: MODERATE
```

## 📊 Prediction Scripts

### **Hybrid Prediction** (`hybrid_live_prediction.py`)
**Purpose**: Smart balance between retraining and calibration for daily predictions
- ✅ **Auto mode**: Retrains weekly, calibrates daily (persistent across sessions)
- ✅ **Retrain mode**: Full retraining every time (10-30 minutes)
- ✅ **Calibrate mode**: Uses calibration only (1-2 seconds)
- ✅ Perfect for all use cases - trading, research, and production
- ✅ **Smart memory**: Remembers training history even after restarting

```bash
# Auto mode (recommended for new users)
python hybrid_live_prediction.py --mode auto

# Force retraining (for maximum accuracy)
python hybrid_live_prediction.py --mode retrain

# Use calibration only (for fast daily predictions)
python hybrid_live_prediction.py --mode calibrate
```

## 🔬 Research Usage

### Running Experiments

The main script for running experiments is `scripts/run_sim_models.py`:

```bash
# Run a single experiment
python scripts/run_sim_models.py --dgp garch11_skt --alpha 0.05 --seed 42

# Run multiple seeds
python scripts/run_sim_models.py --dgp garch11_skt --alpha 0.05 --seeds "1-10"

# Run with calibration
python scripts/run_sim_models.py --dgp garch11_skt --alpha 0.05 --calibrate --seed 42
```

### Available DGPs

- `garch11_skt`: GARCH(1,1) with skew-t innovations
- `garch11_t`: GARCH(1,1) with Student-t innovations
- `iid_gaussian`: IID Gaussian returns
- `sv`: Stochastic volatility model
- `srnn_like`: SRNN-like data generating process

**DGP Organization:**
- **`src/dgp/`**: Clean, modular DGP implementations
- **`src/dgp/__init__.py`**: Central registry and presets
- **Individual modules**: Each DGP type in its own file

### Available Models

- **Transformer**: Attention-based neural network
- **SRNN**: State recurrent neural network
- **Baseline**: Classical methods (GARCH-t, RiskMetrics, Random Walk)

## 📈 Model Details

### Transformer Model
- **Architecture**: Multi-head attention with positional encoding
- **Features**: Configurable feature set (parity vs full)
- **Calibration**: Optional exact-factor calibration
- **Loss**: FZ0 loss function (Patton 2019)
- **Context Length**: 64 days
- **Output**: VaR and ES predictions

### Key Features
- **Sliding Window**: Uses 64-day context for each prediction
- **Calibration**: Adjusts predictions based on recent performance
- **Multiple Modes**: Evaluation, live prediction, and hybrid approaches

## 📋 Data Requirements

Your CSV file needs at least one column:
```csv
date,close
2020-01-01,100.50
2020-01-02,101.20
2020-01-03,99.80
...
```

**Required:**
- `close`: Daily closing prices

**Optional:**
- `date`: Date column (for reference)

## 🎯 Use Cases

### For Traders
- **Daily risk management**
- **Position sizing decisions**
- **Stop-loss placement**
- **Portfolio stress testing**

### For Researchers
- **Model comparison studies**
- **Backtesting strategies**
- **Risk measure evaluation**
- **Academic research**

### For Risk Managers
- **Regulatory reporting**
- **Stress testing**
- **Risk limit monitoring**
- **Portfolio optimization**

## 📊 Evaluation Metrics

- **FZ0 Loss**: Fissler-Ziegel loss function (lower is better)
- **Hit Rate**: Proportion of VaR violations (should be close to α)
- **Kupiec Test**: Unconditional coverage test
- **Christoffersen Tests**: Independence and conditional coverage tests
- **Diebold-Mariano**: Pairwise model comparison tests

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agri-forecast
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START_GUIDE.md)**: For new users
- **[Prediction Comparison](PREDICTION_COMPARISON.md)**: Script differences
- **[Model Evaluation](README.md#research-usage)**: For researchers

## 🤝 Contributing

1. Follow the established directory structure
2. Use the new artifacts organization for all outputs
3. Update the aggregation script if adding new metrics
4. Document any new DGPs or models

## 📄 License

[Add your license information here]
