# Standalone Transformer Model - Summary

## 📁 Required Files
- `standalone_transformer_class.py` - Main script (class-based version)
- `src/models/transformer_var_es_paper_exact.py` - Transformer model implementation
- `src/utils/eval_tools.py` - Evaluation tools and metrics
- `requirements_standalone.txt` - Python dependencies

## 🚀 How to Use

### Basic Usage
```bash
# Standard evaluation mode
python standalone_transformer_class.py --csv your_data.csv --alpha 0.01 --feature_parity

# With calibration and next-day prediction
python standalone_transformer_class.py --csv your_data.csv --alpha 0.01 --feature_parity --calibrate --predict_next

# Live prediction mode (hybrid retraining/calibration)
python standalone_transformer_class.py --csv your_data.csv --alpha 0.01 --feature_parity --live_mode --mode auto
```

### Live Mode Options
```bash
# Force retraining
python standalone_transformer_class.py --csv your_data.csv --live_mode --mode retrain

# Use calibration only
python standalone_transformer_class.py --csv your_data.csv --live_mode --mode calibrate

# Auto mode (retrain every 7 days)
python standalone_transformer_class.py --csv your_data.csv --live_mode --mode auto
```

## 📊 What Output is Generated

### Standard Mode
- **Model Training/Loading**: Trains new model or loads existing one
- **Evaluation Results**: Hit rate, statistical tests (Kupiec, Independence, Conditional Coverage), FZ0 loss
- **Next Day Prediction** (if `--predict_next`): VaR and ES predictions with risk level assessment

### Live Mode
- **Hybrid Workflow**: Automatic retraining or calibration based on mode
- **Tomorrow's Prediction**: VaR/ES predictions with interpretation
- **Saved Results**: JSON file with prediction details and metadata
- **Training Tracking**: Persistent storage of last training date

## ✨ Key Features

### 🏗️ **Class-Based Architecture**
- **`StandalonePredictor`** class encapsulates all functionality
- **State Management**: Model, data, calibration factors, training dates
- **Clean Interface**: Easy to use programmatically or via command line

### 🔄 **Hybrid Live Prediction**
- **Auto-Retraining**: Configurable retraining schedule (default: 7 days)
- **Smart Calibration**: Uses rolling online calibration from transformer script
- **Production Ready**: Optimized for daily predictions

### 💾 **Model Persistence**
- **Data Hashing**: Unique model identification based on data characteristics
- **Automatic Loading**: Loads existing models when available
- **Training History**: Tracks and persists training dates

### 🎯 **Flexible Modes**
- **Standard Mode**: Research/analysis with full evaluation
- **Live Mode**: Production-ready daily predictions
- **Calibration**: Uses exact same method as main transformer script

## 📈 Example Output

### Standard Mode
```
Test Results (calibrated):
  Hit rate: 0.0106 (Target: 0.0100)
  Kupiec test: LR=0.1089, p=0.7414
  Independence test: LR=0.8269, p=0.3632
  Conditional coverage: LR=0.9358, p=0.6263
  Average FZ0 loss: -1.993531

VaR prediction: -0.0404 (-4.04%)
ES prediction: -0.0498 (-4.98%)
Risk level: HIGH
```

### Live Mode
```
============================================================
TOMORROW'S RISK PREDICTION
============================================================
Mode: CALIBRATE
Confidence Level: 99.0%
VaR (Value at Risk): -4.04%
ES (Expected Shortfall): -4.98%

Raw vs Calibrated:
  Raw VaR: -4.04%
  Calibrated VaR: -4.04%

Risk Level: HIGH
Processing Time: 0.02 seconds
```

## 🔧 Technical Details

### Class Structure
```python
class StandalonePredictor:
    def __init__(self, csv_path, alpha=0.01, feature_parity=True, ...)
    def load_and_prepare_data(self)
    def train_model(self, force_retrain=False)
    def evaluate_model(self, calibrate=False)
    def predict_next_day(self)
    def run_live_mode(self, mode="auto", retrain_days=7, ...)
    def run_standard_mode(self, calibrate=False, predict_next=False, ...)
```

### Calibration Method
- **Same as Transformer Script**: Uses `rolling_online_factors` from `transformer_var_es_paper_exact.py`
- **No Look-Ahead Bias**: Proper implementation for live predictions
- **Adaptive Factors**: Calibration factors update based on recent performance

### Model Management
- **Data-Driven Naming**: Model files include data hash for uniqueness
- **Automatic Detection**: Checks for existing models before training
- **Persistent State**: Saves training dates and calibration factors

## 🛠️ Troubleshooting

### Common Issues
1. **"No trained model found"**: Run with `--mode retrain` first
2. **Data hash mismatch**: Different data will create different model files
3. **Import errors**: Ensure `src/` directory is in Python path

### Performance
- **Fast Calibration**: ~0.02 seconds for calibration-only mode
- **Model Loading**: ~1-2 seconds for existing models
- **Full Training**: ~3-4 minutes for new model training

## 🔗 Integration Examples

### Programmatic Usage
```python
from standalone_transformer_class import StandalonePredictor

# Create predictor
predictor = StandalonePredictor(
    csv_path="data/prices.csv",
    alpha=0.01,
    feature_parity=True
)

# Run live prediction
results = predictor.run_live_mode(mode="auto")

# Run standard evaluation
results = predictor.run_standard_mode(calibrate=True, predict_next=True)
```

### Command Line Integration
```bash
# Daily cron job for live predictions
0 9 * * * cd /path/to/project && python standalone_transformer_class.py --csv data/daily_prices.csv --live_mode --mode auto

# Weekly retraining
0 8 * * 1 cd /path/to/project && python standalone_transformer_class.py --csv data/weekly_prices.csv --live_mode --mode retrain
```

## 🎉 Benefits of Class-Based Approach

1. **🎯 Clean Architecture**: Organized, maintainable code structure
2. **🔄 State Management**: Persistent model and calibration state
3. **⚡ Production Ready**: Optimized for live daily predictions
4. **🔧 Flexible**: Easy to extend and customize
5. **📊 Consistent**: Same calibration method as main transformer script
6. **💾 Efficient**: Smart model loading and data hashing
7. **🎛️ Configurable**: Multiple modes for different use cases

The class-based approach makes the standalone transformer much more organized, maintainable, and production-ready while preserving all the functionality of the original script!
