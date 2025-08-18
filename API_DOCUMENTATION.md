# API Documentation

## Overview

This document provides detailed API documentation for all prediction scripts and functions in the Agri-Forecast project.

## 📊 Prediction Scripts

### 1. Direct Model Execution

**Purpose**: Run models directly with command line arguments for maximum flexibility.

#### Transformer Model
```bash
# Basic usage
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py

# With custom parameters
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py --alpha 0.05 --calibrate --run-tag "experiment1"
```

#### GARCH Model
```bash
# Basic usage
PYTHONPATH=. python src/models/garch.py

# With custom parameters
PYTHONPATH=. python src/models/garch.py --alpha 0.05 --calibrate --no-feature-parity --run-tag "experiment2"
```

#### Command Line Arguments (Both Models)
- `--csv`: Path to CSV file (default: `data/merged_data_with_realised_volatility.csv`)
- `--alpha`: VaR/ES confidence level (default: 0.01)
- `--calibrate`: Apply calibration (default: False)
- `--no-feature-parity`: Use full features instead of parity (default: True)
- `--out-dir`: Output directory for results (default: `saved_models`)
- `--fig-dir`: Output directory for figures (default: `figures`)
- `--run-tag`: Optional run tag for file naming

### 2. Organized Runner (`run_individual_models.py`)

**Purpose**: Run models with organized artifact structure and consistent interface.

#### Usage
```bash
# Run transformer model
python run_individual_models.py --model transformer --csv data/raw/merged_data_with_realised_volatility.csv

# Run GARCH model
python run_individual_models.py --model garch --csv data/raw/merged_data_with_realised_volatility.csv --calibrate
```

### 3. Live Prediction (`live_prediction.py`)

**Purpose**: Make actual predictions for tomorrow's trading using all historical data.

#### Usage
```bash
python live_prediction.py
```

#### Key Functions

##### `prepare_live_data(csv_path)`
Prepares data for live prediction using ALL historical data.

**Parameters:**
- `csv_path` (str): Path to CSV file with price data

**Returns:**
- `X_scaled` (np.ndarray): Standardized feature matrix
- `y` (np.ndarray): Target return values
- `scaler` (StandardScaler): Fitted scaler object
- `feature_cols` (list): List of feature column names

##### `train_live_model(X, y, alpha=0.01)`
Trains the transformer model on all historical data.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Target values
- `alpha` (float): VaR/ES confidence level (default: 0.01)

**Returns:**
- `model` (BasicVaRTransformer): Trained model
- `training_time` (float): Training time in seconds

##### `predict_tomorrow(model, X_all, scaler, feature_cols, alpha=0.01)`
Makes prediction for tomorrow using the most recent CONTEXT_LEN days.

**Parameters:**
- `model` (BasicVaRTransformer): Trained model
- `X_all` (np.ndarray): All feature data
- `scaler` (StandardScaler): Fitted scaler
- `feature_cols` (list): Feature column names
- `alpha` (float): VaR/ES confidence level

**Returns:**
- `var_pred` (float): VaR prediction
- `es_pred` (float): ES prediction

#### Output Format
```json
{
  "prediction_date": "2024-01-15",
  "target_date": "2024-01-16",
  "var": -0.0234,
  "es": -0.0345,
  "var_pct": -2.34,
  "es_pct": -3.45,
  "risk_level": "MODERATE",
  "confidence": 99.0
}
```

---

### 2. Hybrid Prediction (`hybrid_live_prediction.py`)

**Purpose**: Smart balance between retraining and calibration for daily predictions.

#### Usage
```bash
# Auto mode (recommended)
python hybrid_live_prediction.py --mode auto

# Force retraining
python hybrid_live_prediction.py --mode retrain

# Use calibration only
python hybrid_live_prediction.py --mode calibrate
```

#### Command Line Arguments
- `--mode`: Prediction mode (`retrain`, `calibrate`, `auto`)
- `--alpha`: VaR/ES confidence level (default: 0.01)
- `--csv`: Path to CSV file (default: `data/merged_data_with_realised_volatility.csv`)

#### Class: `HybridPredictor`

##### Constructor
```python
HybridPredictor(csv_path, alpha=0.01)
```

**Parameters:**
- `csv_path` (str): Path to CSV file
- `alpha` (float): VaR/ES confidence level

##### Methods

###### `prepare_data(use_all_data=True)`
Prepares data for training or prediction.

**Parameters:**
- `use_all_data` (bool): Whether to use all data or exclude today

**Returns:**
- `X_scaled` (np.ndarray): Standardized features
- `y` (np.ndarray): Target values
- `scaler` (StandardScaler): Fitted scaler
- `feature_cols` (list): Feature names

###### `train_model(X, y)`
Trains the model from scratch.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Target values

**Returns:**
- `model` (BasicVaRTransformer): Trained model
- `training_time` (float): Training time

###### `update_calibration(X, y)`
Updates calibration factors using recent data.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Target values

**Returns:**
- `calibration_time` (float): Calibration time

###### `predict_tomorrow(mode='auto')`
Makes prediction for tomorrow with smart mode selection.

**Parameters:**
- `mode` (str): `retrain`, `calibrate`, or `auto`

**Returns:**
- `results` (dict): Prediction results

#### Auto Mode Logic
- **First run**: Full retraining
- **Daily runs**: Calibration only  
- **Every 7 days**: Full retraining again
- **Persistent**: Remembers last training date across sessions
- **Smart**: Automatically loads training history from `hybrid_predictions/last_training_info.json`

---

### 3. Comparison Analysis (`retraining_vs_calibration.py`)

**Purpose**: Compare retraining vs calibration approaches.

#### Usage
```bash
python retraining_vs_calibration.py
```

#### Key Functions

##### `approach_1_retrain_daily(csv_path, alpha=0.01, days_to_simulate=30)`
Simulates daily retraining approach.

**Parameters:**
- `csv_path` (str): Path to CSV file
- `alpha` (float): VaR/ES confidence level
- `days_to_simulate` (int): Number of days to simulate

**Returns:**
- `predictions` (list): List of prediction results

##### `approach_2_calibration_daily(csv_path, alpha=0.01, days_to_simulate=30)`
Simulates daily calibration approach.

**Parameters:**
- `csv_path` (str): Path to CSV file
- `alpha` (float): VaR/ES confidence level
- `days_to_simulate` (int): Number of days to simulate

**Returns:**
- `predictions` (list): List of prediction results

##### `compare_approaches(csv_path)`
Compares both approaches side by side.

**Parameters:**
- `csv_path` (str): Path to CSV file

---

## 🔧 Core Model Functions

### Transformer Model (`src/models/transformer_var_es_paper_exact.py`)

#### `pipeline(csv_path, alpha=0.01, feature_parity=True, calibrate=False, run_tag=None, out_dir="saved_models", fig_dir="figures")`

Main pipeline function for model evaluation.

**Parameters:**
- `csv_path` (str): Path to CSV file
- `alpha` (float): VaR/ES confidence level
- `feature_parity` (bool): Use simplified features if True
- `calibrate` (bool): Apply calibration if True
- `run_tag` (str): Tag for output files
- `out_dir` (str): Output directory
- `fig_dir` (str): Figures directory

**Returns:**
- `model` (BasicVaRTransformer): Trained model
- `metrics` (dict): Performance metrics
- `(v_eval, e_eval, y_aligned, fz0)` (tuple): Predictions and actual values

#### `train_with_stride(X_train, y_train, input_dim, alpha=0.01, seq_len=64, train_stride=1)`

Trains the transformer model with sliding windows.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training targets
- `input_dim` (int): Number of input features
- `alpha` (float): VaR/ES confidence level
- `seq_len` (int): Context length
- `train_stride` (int): Stride for training windows

**Returns:**
- `model` (BasicVaRTransformer): Trained model

#### `evaluate_with_sliding_batch(model, X_all, y_all, start_idx, seq_len=64, batch_size=64)`

Evaluates model using sliding batch approach.

**Parameters:**
- `model` (BasicVaRTransformer): Trained model
- `X_all` (np.ndarray): All feature data
- `y_all` (np.ndarray): All target data
- `start_idx` (int): Starting index for evaluation
- `seq_len` (int): Context length
- `batch_size` (int): Batch size for evaluation

**Returns:**
- `v` (np.ndarray): VaR predictions
- `e` (np.ndarray): ES predictions

---

## 📊 Data Processing Functions

### `build_inputs_from_prices(df)`

Builds features from price data.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with 'close' column

**Returns:**
- `df` (pd.DataFrame): DataFrame with additional features

**Features Created:**
- `log_ret`: Log returns
- `r2`: Squared returns
- `neg_ret`: Negative return indicator
- `ewma94_var`: EWMA variance (λ=0.94)
- `ewma97_var`: EWMA variance (λ=0.97)
- `ewma94`: EWMA volatility (λ=0.94)
- `ewma97`: EWMA volatility (λ=0.97)
- `target_return`: Next-day return

### `split_and_make_features(df, feature_parity=True, train_frac=0.5)`

Splits data and creates features.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `feature_parity` (bool): Use simplified features if True
- `train_frac` (float): Training fraction

**Returns:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training targets
- `X_test` (np.ndarray): Test features
- `y_test` (np.ndarray): Test targets
- `meta` (dict): Metadata including scaler info

---

## 🎯 Calibration Functions

### `rolling_online_factors(y, v, e, alpha, W=None)`

Computes rolling calibration factors.

**Parameters:**
- `y` (np.ndarray): Actual returns
- `v` (np.ndarray): VaR predictions
- `e` (np.ndarray): ES predictions
- `alpha` (float): VaR/ES confidence level
- `W` (int): Window size (auto-determined if None)

**Returns:**
- `c_v` (np.ndarray): VaR calibration factors
- `c_e` (np.ndarray): ES calibration factors

### `expanding_online_factors(y, v, e, alpha, warmup=None)`

Computes expanding calibration factors.

**Parameters:**
- `y` (np.ndarray): Actual returns
- `v` (np.ndarray): VaR predictions
- `e` (np.ndarray): ES predictions
- `alpha` (float): VaR/ES confidence level
- `warmup` (int): Warmup period (auto-determined if None)

**Returns:**
- `c_v` (np.ndarray): VaR calibration factors
- `c_e` (np.ndarray): ES calibration factors

---

## 📈 Model Architecture

### `BasicVaRTransformer`

Transformer model for VaR/ES prediction.

#### Constructor
```python
BasicVaRTransformer(input_dim, model_dim=32, num_heads=2, num_layers=1, dropout=0.2)
```

**Parameters:**
- `input_dim` (int): Number of input features
- `model_dim` (int): Model dimension (default: 32)
- `num_heads` (int): Number of attention heads (default: 2)
- `num_layers` (int): Number of transformer layers (default: 1)
- `dropout` (float): Dropout rate (default: 0.2)

#### Forward Pass
```python
def forward(self, x)
```

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

**Returns:**
- `torch.Tensor`: Output tensor of shape (batch_size, 2) containing [VaR, ES]

---

## 🔍 Evaluation Metrics

### `fz0_per_step(y_true, var_pred, es_pred, alpha)`

Computes FZ0 loss for each prediction.

**Parameters:**
- `y_true` (np.ndarray): Actual returns
- `var_pred` (np.ndarray): VaR predictions
- `es_pred` (np.ndarray): ES predictions
- `alpha` (float): VaR/ES confidence level

**Returns:**
- `fz0` (np.ndarray): FZ0 loss values

### `kupiec_pof(hits, alpha)`

Kupiec test for unconditional coverage.

**Parameters:**
- `hits` (np.ndarray): Binary hit sequence
- `alpha` (float): VaR/ES confidence level

**Returns:**
- `LR` (float): Likelihood ratio statistic
- `p_value` (float): P-value
- `hit_rate` (float): Observed hit rate
- `expected_hits` (float): Expected number of hits

### `christoffersen_independence(hits)`

Christoffersen test for independence.

**Parameters:**
- `hits` (np.ndarray): Binary hit sequence

**Returns:**
- `LR` (float): Likelihood ratio statistic
- `p_value` (float): P-value

---

## 📁 Output Files

### Prediction Outputs

#### Live Predictions (`live_predictions/`)
```
prediction_20240115_143022.json
```

#### Hybrid Predictions (`hybrid_predictions/`)
```
prediction_20240115_143022.json
```

#### Evaluation Outputs (`predictions_output/`)
```
transformer_next_day_prediction_calibrated.npz
transformer_next_day_prediction_calibrated.json
```

### JSON Output Format
```json
{
  "prediction_date": "2024-01-15",
  "target_date": "2024-01-16",
  "mode": "retrain",
  "var_raw": -0.0234,
  "es_raw": -0.0345,
  "var_calibrated": -0.0256,
  "es_calibrated": -0.0378,
  "var_pct": -2.56,
  "es_pct": -3.78,
  "calibration_factors": {
    "c_v": 1.094,
    "c_e": 1.096
  },
  "processing_time": 1250.45,
  "last_training_date": "2024-01-15"
}
```

### NPZ Output Format
Contains numpy arrays:
- `y`: Actual returns
- `var`: VaR predictions
- `es`: ES predictions
- `fz0`: FZ0 loss values
- `hits`: Binary hit sequence
- `features`: Feature column names
- `feature_parity`: Boolean feature flag
- `c_v`: VaR calibration factor
- `c_e`: ES calibration factor

---

## 🚨 Error Handling

### Common Errors and Solutions

#### Data File Not Found
```
❌ Error: Data file not found at data/merged_data_with_realised_volatility.csv
```
**Solution**: Ensure your CSV file exists in the `data/` directory.

#### Insufficient Data
```
❌ Error: Not enough data for training
```
**Solution**: Ensure you have at least 200+ days of price data.

#### Model Training Issues
```
❌ Error: Model failed to converge
```
**Solution**: Try different alpha values or use `feature_parity=True`.

#### Memory Issues
```
❌ Error: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU training.

---

## 📚 Examples

### Basic Usage
```python
from src.models.transformer_var_es_paper_exact import pipeline

# Run evaluation
model, metrics, (v_eval, e_eval, y_aligned, fz0) = pipeline(
    csv_path="data/your_data.csv",
    alpha=0.01,
    calibrate=True
)

print(f"Hit rate: {metrics['hit_rate']:.4f}")
print(f"FZ0 loss: {metrics['avg_fz0']:.6f}")
```

### Custom Prediction
```python
from hybrid_live_prediction import HybridPredictor

# Create predictor
predictor = HybridPredictor("data/your_data.csv", alpha=0.01)

# Make prediction
results = predictor.predict_tomorrow(mode="auto")

print(f"Tomorrow's VaR: {results['var_pct']:.2f}%")
print(f"Tomorrow's ES: {results['es_pct']:.2f}%")
```
