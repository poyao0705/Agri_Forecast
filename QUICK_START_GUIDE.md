# Quick Start Guide: Next-Day Return Prediction

## For New Users - How to Use the Transformer Model

### What This Model Does
This transformer model predicts **next-day risk measures** for financial time series (like commodity prices):
- **VaR (Value at Risk)**: Maximum expected loss for the next day
- **ES (Expected Shortfall)**: Average loss when VaR is exceeded

### Prerequisites
1. Python 3.7+ installed
2. Required packages: `torch`, `pandas`, `numpy`, `scikit-learn`
3. Your data in CSV format

### Your Data Format
Your CSV file needs at least one column:
```
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

### Step 1: Prepare Your Data
Place your CSV file in the `data/` folder. For example:
```
data/
└── your_prices.csv
```

### Step 2: Run the Prediction

**Option A: Direct Model Execution (Simplest)**
```bash
# Run transformer model
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py --csv data/your_prices.csv

# Run GARCH model
PYTHONPATH=. python src/models/garch.py --csv data/your_prices.csv
```

**Option B: Organized Runner**
```bash
python run_individual_models.py --model transformer --csv data/your_prices.csv
python run_individual_models.py --model garch --csv data/your_prices.csv
```

### Step 3: Get Your Results
The models will:
1. **Train the model** on the first 50% of your data
2. **Generate predictions** for the second 50% of your data
3. **Save results** to:
   - `saved_models/` - Raw predictions and metrics (.npz, .json files)
   - `figures/` - Diagnostic plots (.png files)

### What You Get
For each day in your test period, you'll get:
- **VaR prediction**: The maximum expected loss (negative number)
- **ES prediction**: The average loss when VaR is exceeded (negative number)
- **Actual return**: What actually happened the next day
- **Hit rate**: How often the model correctly predicted extreme losses

### Example Output
```
TRANSFORMER MODEL: Next-Day Return Prediction
============================================================
Step 1: Training model and generating predictions...
Step 2: Prediction Results
----------------------------------------
Number of predictions: 2296
Date range: 2296 days of predictions
VaR range: [-0.0452, -0.0012]
ES range: [-0.0678, -0.0021]
Hit rate: 0.0104 (target: 0.0100)
Average FZ0 loss: 0.023456

Recent Predictions (last 10 days):
Day | VaR    | ES     | Actual | Hit?
----------------------------------------
2287 | -0.0234 | -0.0345 | -0.0156 | ✗
2288 | -0.0211 | -0.0312 | -0.0289 | ✓
2289 | -0.0198 | -0.0298 | -0.0123 | ✗
...
```

### Understanding the Results

**VaR (Value at Risk):**
- Negative number (e.g., -0.0234)
- Means: "We expect the maximum loss tomorrow to be 2.34%"
- If actual return is worse than -2.34%, VaR was "hit"

**ES (Expected Shortfall):**
- Negative number, always worse than VaR (e.g., -0.0345)
- Means: "If VaR is exceeded, we expect to lose 3.45% on average"

**Hit Rate:**
- Should be close to your alpha level (e.g., 0.01 for 1% VaR)
- 0.0104 means VaR was exceeded 1.04% of the time (close to target 1%)

### Customization Options

**Change confidence level:**
```bash
# Command line option
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py --alpha 0.05
PYTHONPATH=. python src/models/garch.py --alpha 0.05
```

**Enable calibration:**
```bash
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py --calibrate
PYTHONPATH=. python src/models/garch.py --calibrate
```

**Use full features instead of parity:**
```bash
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py --no-feature-parity
PYTHONPATH=. python src/models/garch.py --no-feature-parity
```

**Custom output directories:**
```bash
PYTHONPATH=. python src/models/transformer_var_es_paper_exact.py --out-dir "my_results" --fig-dir "my_plots"
```

**Use full features (more complex):**
```python
# In the pipeline call, change:
feature_parity=False  # Instead of True
```

**Disable calibration:**
```python
# In the pipeline call, change:
calibrate=False  # Instead of True
```

### Troubleshooting

**"Data file not found" error:**
- Make sure your CSV file is in the `data/` folder
- Check the filename matches what's in the script

**"ValueError" during training:**
- Ensure your data has enough rows (at least 200+ days)
- Check that your 'close' column contains numeric values

**Poor performance:**
- Try different alpha values (0.01, 0.05, 0.10)
- Use `feature_parity=False` for more complex features
- Ensure your data has enough volatility for meaningful predictions

### Next Steps
1. Run the basic prediction
2. Check the diagnostic plots in `prediction_figures/`
3. Analyze the metrics in `predictions_output/`
4. Adjust parameters based on your needs

### Files Created
```
predictions_output/
├── transformer_next_day_prediction_calibrated.npz  # Raw predictions
├── transformer_next_day_prediction_calibrated.json # Model metrics
└── ...

prediction_figures/
├── transformer_next_day_prediction_calibrated_var_es_plot.png
├── transformer_next_day_prediction_calibrated_hit_sequence.png
└── ...
```

That's it! You now have next-day risk predictions for your financial time series.
