# Standalone Transformer Model for VaR/ES Prediction

This is a simplified, standalone version of the transformer model that others can use with minimal dependencies.

## 📁 Required Files

You only need these 3 files:
1. `src/models/transformer_var_es_paper_exact.py` - The transformer model
2. `src/utils/eval_tools.py` - Evaluation metrics and tools
3. `standalone_transformer.py` - This script

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_standalone.txt
```

### 2. Prepare Your Data
Your CSV file needs a `close` column with daily closing prices:
```csv
date,close
2020-01-01,100.50
2020-01-02,101.20
2020-01-03,99.80
...
```

### 3. Run the Model
```bash
# Basic usage
python standalone_transformer.py --csv your_data.csv

# With custom alpha (VaR confidence level)
python standalone_transformer.py --csv your_data.csv --alpha 0.05

# Use feature parity mode (faster, simpler features)
python standalone_transformer.py --csv your_data.csv --feature_parity

# Make next day prediction
python standalone_transformer.py --csv your_data.csv --predict_next

# Save results to files
python standalone_transformer.py --csv your_data.csv --save_results

# List available trained models
python standalone_transformer.py --list_models

# Force retraining (ignore existing model)
python standalone_transformer.py --csv your_data.csv --force_retrain
```

## 📊 Output

The model will output:
- **Hit rate**: Proportion of VaR violations (should be close to alpha)
- **Statistical tests**: Kupiec, Christoffersen independence and conditional coverage tests
- **FZ0 loss**: Fissler-Ziegel loss function (lower is better)
- **Next day prediction**: VaR and ES predictions (if `--predict_next` is used)

## 🔧 Command Line Options

- `--csv`: Path to your CSV file (required)
- `--alpha`: VaR/ES confidence level (default: 0.01)
- `--feature_parity`: Use simplified features (x_cov only)
- `--train_frac`: Training fraction (default: 0.5)
- `--predict_next`: Make prediction for next day
- `--save_results`: Save results to files
- `--model_dir`: Directory to save/load models (default: models)
- `--force_retrain`: Force retraining even if model exists
- `--list_models`: List available trained models and exit

## 📈 Example Output

```
==================================================
TRAINING TRANSFORMER MODEL
==================================================
Using feature parity mode (x_cov only)
Training model...
Training completed!

==================================================
EVALUATING MODEL
==================================================
Test Results:
  Hit rate: 0.0120 (Target: 0.0100)
  Kupiec test: LR=0.1044, p=0.7466
  Independence test: LR=0.0325, p=0.8569
  Conditional coverage: LR=0.1370, p=0.9338
  Average FZ0 loss: -2.069198

==================================================
NEXT DAY PREDICTION
==================================================
VaR prediction: -0.0256 (-2.56%)
ES prediction: -0.0378 (-3.78%)
Risk level: MODERATE
```

## 🎯 Use Cases

- **Risk Management**: Daily VaR/ES predictions
- **Model Evaluation**: Backtesting and statistical validation
- **Research**: Academic studies and model comparison
- **Trading**: Position sizing and risk assessment

## 📋 Data Requirements

- **Minimum**: 200+ days of price data
- **Recommended**: 1000+ days for better performance
- **Format**: CSV with `close` column
- **Frequency**: Daily data (can be adapted for other frequencies)

## 🔍 Model Details

- **Architecture**: Transformer with multi-head attention
- **Context Length**: 64 days
- **Features**: Configurable (parity vs full)
- **Loss Function**: FZ0 loss (Patton 2019)
- **Calibration**: Optional exact-factor calibration

## 📁 Output Files (if --save_results)

- `results/transformer_predictions_YYYYMMDD_HHMMSS.npz`: Predictions and metrics
- `results/transformer_metrics_YYYYMMDD_HHMMSS.json`: Statistical test results

## 🧠 Model Management

### Automatic Model Saving/Loading
The script automatically saves trained models and reuses them for efficiency:

- **Model Naming**: `transformer_a{alpha}_{parity/full}_train{frac}_{data_hash}.pth`
- **Example**: `transformer_a010_parity_train50_b5f4d221.pth`
  - `a010`: alpha = 0.01
  - `parity`: feature parity mode
  - `train50`: 50% training fraction
  - `b5f4d221`: data hash (identifies dataset)

### Model Operations
```bash
# List all available models
python standalone_transformer.py --list_models

# Use existing model (fast)
python standalone_transformer.py --csv your_data.csv --alpha 0.01

# Force retraining
python standalone_transformer.py --csv your_data.csv --alpha 0.01 --force_retrain

# Use custom model directory
python standalone_transformer.py --csv your_data.csv --model_dir my_models
```

### Benefits
- **Speed**: Loading existing models is ~100x faster than training
- **Consistency**: Same parameters always use the same model
- **Flexibility**: Force retraining when needed
- **Organization**: Models organized by parameters and data

## 🆘 Troubleshooting

### Common Issues

1. **"No module named 'src'"**: Make sure you're in the correct directory
2. **"CSV must contain a 'close' column"**: Check your data format
3. **"Not enough data"**: Ensure you have at least 200+ days of data
4. **CUDA errors**: The model will automatically use CPU if GPU is not available

### Performance Tips

- Use `--feature_parity` for faster training
- Use smaller alpha values (0.01, 0.025) for better performance
- Ensure sufficient training data (1000+ days recommended)

## 📚 Advanced Usage

### Custom Training Parameters
You can modify the model parameters in `src/models/transformer_var_es_paper_exact.py`:
- `CONTEXT_LEN`: Context window size (default: 64)
- `MAX_EPOCHS`: Training epochs (default: 200)
- `BATCH_SIZE`: Batch size (default: 64)
- `LR`: Learning rate (default: 1e-4)

### Integration with Other Tools
The saved `.npz` files can be loaded in Python:
```python
import numpy as np
data = np.load('results/transformer_predictions_20240101_120000.npz')
print(data.files)  # ['y', 'var', 'es', 'fz0', 'hits']
```

## 🤝 Contributing

This standalone version is designed to be simple and self-contained. If you need additional features, consider using the full simulation framework.
