# Prediction Scripts Comparison

## Current Available Scripts

### **Hybrid Prediction** (`hybrid_live_prediction.py`)
**Purpose**: Smart balance between retraining and calibration for daily predictions

**What it does:**
- **Auto mode**: Retrains weekly, calibrates daily (persistent across sessions)
- **Retrain mode**: Full retraining every time
- **Calibrate mode**: Uses calibration only
- Makes ONE prediction for tomorrow
- Saves prediction with timestamp
- **Smart memory**: Remembers last training date even after restarting

**Use case**: Real trading, risk management, daily decision making

**Output**:
```
TOMORROW'S RISK PREDICTION
============================================================
Mode: RETRAIN
Confidence Level: 99.0%
VaR (Value at Risk): -2.34%
ES (Expected Shortfall): -3.45%
Risk Level: MODERATE
Processing Time: 1250.45 seconds
```

---

## When to Use Which Mode

### Use `--mode retrain` when:
- ✅ You want the most up-to-date model
- ✅ You have time for full training (10-30 minutes)
- ✅ You're doing weekly/monthly updates
- ✅ You want maximum accuracy
- ✅ You have sufficient computational resources

### Use `--mode calibrate` when:
- ✅ You need fast daily predictions (1-2 seconds)
- ✅ You already have a trained model
- ✅ You're doing daily risk management
- ✅ You want stable predictions
- ✅ You're in a production environment

### Use `--mode auto` when:
- ✅ You want the best of both worlds
- ✅ You're setting up a production system
- ✅ You want automatic decision making
- ✅ You want to balance speed and accuracy
- ✅ You're new to the system (recommended)
- ✅ You want persistent memory across sessions
- ✅ You're running daily automated predictions

---

## Key Differences

| Aspect | Retrain Mode | Calibrate Mode | Auto Mode |
|--------|-------------|----------------|-----------|
| **Time per run** | 10-30 minutes | 1-2 seconds | Variable |
| **Data Usage** | 100% for training | Uses existing model | Smart choice |
| **Accuracy** | Highest | High | Balanced |
| **Stability** | Low (new model each time) | High (same model) | Balanced |
| **Use Case** | Weekly updates | Daily trading | Production system |

---

## Retraining vs Calibration Analysis

### **Why This Matters:**
When using models daily, you have two approaches:

#### **Approach 1: Retrain Every Day**
```python
# Every day:
1. Get new data point
2. Retrain entire model from scratch (10-30 minutes)
3. Make prediction for tomorrow
```

#### **Approach 2: Calibration**
```python
# Train once, then calibrate daily:
1. Train model once on historical data
2. Each day: apply calibration factors (1-2 seconds)
3. Make prediction for tomorrow
```

### **Performance Comparison:**

| Metric | Retraining | Calibration |
|--------|------------|-------------|
| **Time per day** | 10-30 minutes | 1-2 seconds |
| **Speedup** | 1x | 300-900x faster |
| **Stability** | Low (parameters change daily) | High (structure stays same) |
| **Accuracy** | High | Medium-High |
| **Cost** | High ($10K+ annually) | Low ($100 annually) |

### **Real-World Examples:**

**Tier 1 Banks (JPM, GS, MS):**
- **Frequency**: Retrain monthly, calibrate daily
- **Reason**: Regulatory requirements + risk management

**Quant Hedge Funds (Renaissance, Two Sigma):**
- **Frequency**: Retrain weekly, calibrate hourly
- **Reason**: Alpha decay + market efficiency

**Asset Managers (BlackRock, Vanguard):**
- **Frequency**: Retrain quarterly, calibrate daily
- **Reason**: Long-term focus + cost efficiency

### **Recommendation:**
- **For research/development**: Use retraining to understand model behavior
- **For live trading**: Use calibration for speed and stability
- **For production systems**: Use hybrid approach (retrain weekly, calibrate daily)

---

## Example Workflow

### For Daily Trading:
```bash
# First time setup
python hybrid_live_prediction.py --mode retrain

# Daily predictions (fast)
python hybrid_live_prediction.py --mode calibrate

# Or use auto mode (recommended)
python hybrid_live_prediction.py --mode auto
```

### For Research/Development:
```bash
# Test different configurations
python hybrid_live_prediction.py --mode retrain --alpha 0.01
python hybrid_live_prediction.py --mode retrain --alpha 0.05
python hybrid_live_prediction.py --mode retrain --alpha 0.10
```

### For Production Systems:
```bash
# Set up automated daily predictions
# Use auto mode for smart decision making
python hybrid_live_prediction.py --mode auto
```

---

## File Structure

```
your_project/
├── hybrid_live_prediction.py    # Main prediction script
├── hybrid_predictions/          # Prediction results
│   ├── last_training_info.json  # Training history (auto mode)
│   └── prediction_*.json        # Individual predictions
├── data/
│   └── your_prices.csv         # Your price data
└── src/
    └── models/
        └── transformer_var_es_paper_exact.py  # Core model
```

---

## Quick Start

**For new users (recommended):**
```bash
python hybrid_live_prediction.py --mode auto
```

**For daily trading:**
```bash
python hybrid_live_prediction.py --mode calibrate
```

**For weekly updates:**
```bash
python hybrid_live_prediction.py --mode retrain
```

The script will automatically handle data loading, model training, and result saving. Just make sure your CSV file is in the `data/` folder with a `close` column.

---

## 🧠 Smart Memory Feature

### **How Auto Mode Remembers Across Sessions:**

The auto mode now has **persistent memory** - it remembers when you last trained the model, even if you restart your computer or run the script on different days.

**What gets saved:**
```json
{
  "last_training_date": "2024-12-01T14:30:22.123456",
  "alpha": 0.01,
  "csv_path": "data/merged_data_with_realised_volatility.csv"
}
```

**How it works:**
1. **First run**: Retrains and saves training date
2. **Next day**: Loads training date, uses calibration (fast)
3. **After 7 days**: Loads training date, retrains (accurate)
4. **New session**: Still remembers when you last trained!

**Benefits:**
- ✅ **No manual tracking**: Script remembers for you
- ✅ **Works across sessions**: Restart computer, still works
- ✅ **Automatic decisions**: Smart retrain vs calibrate choice
- ✅ **Production ready**: Perfect for automated daily runs

---

## Decision Matrix

### **Choose Retrain Mode if:**
- [ ] You want maximum accuracy
- [ ] You have time for full training (10-30 minutes)
- [ ] You're doing weekly/monthly updates
- [ ] You have sufficient computational resources
- [ ] You're testing different configurations

### **Choose Calibrate Mode if:**
- [ ] You need fast daily predictions (1-2 seconds)
- [ ] You already have a trained model
- [ ] You're doing daily risk management
- [ ] You want stable predictions
- [ ] You're in a production environment

### **Choose Auto Mode if:**
- [ ] You want the best of both worlds
- [ ] You're setting up a production system
- [ ] You want automatic decision making
- [ ] You're new to the system
- [ ] You want to balance speed and accuracy

---

## Command Line Options

```bash
python hybrid_live_prediction.py [OPTIONS]

Options:
  --mode {retrain,calibrate,auto}  Prediction mode (default: auto)
  --alpha FLOAT                    VaR/ES confidence level (default: 0.01)
  --csv PATH                       Path to CSV file (default: data/merged_data_with_realised_volatility.csv)
```

### **Examples:**
```bash
# Basic usage
python hybrid_live_prediction.py

# Force retraining
python hybrid_live_prediction.py --mode retrain

# Use calibration only
python hybrid_live_prediction.py --mode calibrate

# Different confidence level
python hybrid_live_prediction.py --alpha 0.05

# Custom data file
python hybrid_live_prediction.py --csv data/my_prices.csv
```
