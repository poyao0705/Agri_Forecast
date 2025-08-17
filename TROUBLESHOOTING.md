# Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. Data Issues

#### ❌ "Data file not found" Error
```
❌ Error: Data file not found at data/merged_data_with_realised_volatility.csv
```

**Causes:**
- CSV file doesn't exist in the `data/` directory
- Wrong file path specified
- File permissions issue

**Solutions:**
1. Check if your CSV file exists:
   ```bash
   ls -la data/
   ```

2. Ensure your CSV has the required `close` column:
   ```bash
   head -5 data/your_file.csv
   ```

3. Update the file path in your script:
   ```python
   csv_path = "data/your_actual_file.csv"
   ```

#### ❌ "Not enough data for training" Error
```
❌ Error: Not enough data for training
```

**Causes:**
- Less than 200 days of price data
- Too many NaN values after preprocessing
- Data preprocessing removed too many rows

**Solutions:**
1. Check your data length:
   ```bash
   wc -l data/your_file.csv
   ```

2. Ensure you have at least 200+ days of data

3. Check for missing values:
   ```python
   import pandas as pd
   df = pd.read_csv("data/your_file.csv")
   print(f"Total rows: {len(df)}")
   print(f"Missing values: {df.isnull().sum()}")
   ```

#### ❌ "Invalid data format" Error
```
❌ Error: Could not convert string to float
```

**Causes:**
- Non-numeric values in price column
- Date format issues
- Extra characters in data

**Solutions:**
1. Clean your data:
   ```python
   import pandas as pd
   df = pd.read_csv("data/your_file.csv")
   
   # Convert close to numeric, removing any non-numeric values
   df['close'] = pd.to_numeric(df['close'], errors='coerce')
   
   # Remove rows with NaN values
   df = df.dropna()
   
   # Save cleaned data
   df.to_csv("data/cleaned_data.csv", index=False)
   ```

2. Check for common issues:
   ```python
   # Check for currency symbols
   print(df['close'].head())
   
   # Check for commas in numbers
   df['close'] = df['close'].str.replace(',', '')
   ```

### 2. Model Training Issues

#### ❌ "Model failed to converge" Error
```
❌ Error: Model training did not converge
```

**Causes:**
- Learning rate too high
- Insufficient training data
- Poor data quality
- Model architecture issues

**Solutions:**
1. Use simplified features:
   ```python
   # In your script, set:
   feature_parity = True  # Use simplified features
   ```

2. Try different alpha values:
   ```python
   alpha = 0.05  # Instead of 0.01
   ```

3. Increase training data:
   - Ensure you have at least 1000+ days of data
   - Use more recent data if available

4. Check data quality:
   ```python
   # Check for extreme values
   print(f"Price range: {df['close'].min()} to {df['close'].max()}")
   print(f"Return range: {df['log_ret'].min():.4f} to {df['log_ret'].max():.4f}")
   ```

#### ❌ "CUDA out of memory" Error
```
❌ Error: CUDA out of memory
```

**Causes:**
- GPU memory insufficient
- Batch size too large
- Model too large

**Solutions:**
1. Use CPU training:
   ```python
   # Add this to your script
   import torch
   torch.set_default_tensor_type('torch.FloatTensor')
   ```

2. Reduce batch size:
   ```python
   # In the model configuration
   BATCH_SIZE = 32  # Instead of 64
   ```

3. Use smaller model:
   ```python
   # In BasicVaRTransformer
   model_dim = 16  # Instead of 32
   num_heads = 1   # Instead of 2
   ```

### 3. Prediction Issues

#### ❌ "No trained model found" Error
```
❌ No trained model found. Please run with mode='retrain' first.
```

**Causes:**
- Running calibration mode without a trained model
- Model file corrupted or missing

**Solutions:**
1. Run with retraining first:
   ```bash
   python hybrid_live_prediction.py --mode retrain
   ```

2. Then use calibration:
   ```bash
   python hybrid_live_prediction.py --mode calibrate
   ```

#### ❌ "Prediction values are NaN" Error
```
❌ Error: Prediction contains NaN values
```

**Causes:**
- Input data contains NaN values
- Model weights became NaN during training
- Calibration factors are invalid

**Solutions:**
1. Check input data:
   ```python
   # Check for NaN in features
   print(f"NaN in features: {np.isnan(X).sum()}")
   ```

2. Check model weights:
   ```python
   # Check model parameters
   for name, param in model.named_parameters():
       if torch.isnan(param).any():
           print(f"NaN in {name}")
   ```

3. Check calibration factors:
   ```python
   print(f"Calibration factors: {calibration_factors}")
   ```

### 4. Performance Issues

#### ❌ "Training is too slow" Issue
```
Training time: 1800.45 seconds  # 30 minutes is too long
```

**Solutions:**
1. Use simplified features:
   ```python
   feature_parity = True  # Much faster
   ```

2. Reduce training epochs:
   ```python
   MAX_EPOCHS = 100  # Instead of 200
   ```

3. Use smaller context length:
   ```python
   CONTEXT_LEN = 32  # Instead of 64
   ```

4. Use hybrid approach:
   ```bash
   python hybrid_live_prediction.py --mode calibrate
   ```

#### ❌ "Poor prediction accuracy" Issue
```
Hit rate: 0.0050  # Should be close to alpha (0.01)
```

**Solutions:**
1. Enable calibration:
   ```python
   calibrate = True
   ```

2. Use more features:
   ```python
   feature_parity = False  # Use full feature set
   ```

3. Try different alpha values:
   ```python
   alpha = 0.05  # More conservative
   ```

4. Check data quality and recency

### 5. File and Directory Issues

#### ❌ "Permission denied" Error
```
❌ Error: Permission denied when creating directory
```

**Solutions:**
1. Check directory permissions:
   ```bash
   ls -la
   ```

2. Create directories manually:
   ```bash
   mkdir -p live_predictions hybrid_predictions
   ```

3. Fix permissions:
   ```bash
   chmod 755 live_predictions hybrid_predictions
   ```

#### ❌ "Output files not created" Issue
```
❌ No output files found
```

**Solutions:**
1. Check if script completed successfully
2. Look for error messages in output
3. Check disk space:
   ```bash
   df -h
   ```

### 6. Environment Issues

#### ❌ "Module not found" Error
```
❌ ModuleNotFoundError: No module named 'torch'
```

**Solutions:**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Activate virtual environment:
   ```bash
   source venv/bin/activate  # On Unix/Mac
   venv\Scripts\activate     # On Windows
   ```

3. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

#### ❌ "Version compatibility" Error
```
❌ Error: Incompatible versions
```

**Solutions:**
1. Check versions:
   ```bash
   python --version
   pip list | grep torch
   pip list | grep pandas
   ```

2. Update packages:
   ```bash
   pip install --upgrade torch pandas numpy
   ```

3. Use specific versions:
   ```bash
   pip install torch==1.12.0 pandas==1.5.0
   ```

## 🔧 Debugging Tips

### 1. Enable Verbose Output
```python
# Add debug prints
print(f"Data shape: {X.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Training loss: {loss.item():.6f}")
```

### 2. Check Intermediate Results
```python
# Check predictions at each step
print(f"Raw VaR: {var_raw:.6f}")
print(f"Calibrated VaR: {var_calibrated:.6f}")
print(f"Calibration factor: {c_v:.3f}")
```

### 3. Validate Data Pipeline
```python
# Test data preprocessing
df = pd.read_csv("data/your_file.csv")
df_processed = build_inputs_from_prices(df)
print(f"Original rows: {len(df)}")
print(f"Processed rows: {len(df_processed)}")
print(f"Features: {df_processed.columns.tolist()}")
```

### 4. Monitor Training Progress
```python
# Add training progress monitoring
for epoch in range(MAX_EPOCHS):
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
```

## 📞 Getting Help

### 1. Check Logs
Look for error messages in:
- Terminal output
- Log files in `logs/` directory
- Python error traceback

### 2. Common Debugging Commands
```bash
# Check data file
head -10 data/your_file.csv

# Check Python environment
python -c "import torch; print(torch.__version__)"

# Check disk space
df -h

# Check memory usage
top -p $(pgrep python)
```

### 3. Minimal Test Case
Create a minimal test to isolate the issue:
```python
# test_minimal.py
import pandas as pd
import numpy as np

# Create minimal test data
dates = pd.date_range('2020-01-01', periods=500, freq='D')
prices = 100 + np.cumsum(np.random.randn(500) * 0.01)
df = pd.DataFrame({'date': dates, 'close': prices})
df.to_csv('test_data.csv', index=False)

# Test basic functionality
from src.models.transformer_var_es_paper_exact import build_inputs_from_prices
df_processed = build_inputs_from_prices(df)
print(f"Test successful: {len(df_processed)} rows processed")
```

### 4. Contact Information
If you're still having issues:
1. Check the [API Documentation](API_DOCUMENTATION.md)
2. Review the [Quick Start Guide](QUICK_START_GUIDE.md)
3. Look at example outputs in the documentation
4. Create a minimal reproducible example

## 🎯 Performance Optimization

### 1. Speed Up Training
- Use `feature_parity=True` (simplified features)
- Reduce `MAX_EPOCHS` to 100
- Use smaller `CONTEXT_LEN` (32 instead of 64)
- Use GPU if available

### 2. Improve Accuracy
- Use `feature_parity=False` (full features)
- Enable calibration (`calibrate=True`)
- Use more training data
- Try different alpha values

### 3. Memory Optimization
- Use CPU training for large datasets
- Reduce batch size
- Use smaller model dimensions
- Process data in chunks

## 📊 Expected Performance

### Typical Results
- **Training time**: 10-30 minutes (full retraining)
- **Calibration time**: 1-2 seconds
- **Hit rate**: Should be close to alpha (e.g., 0.01 for 1% VaR)
- **FZ0 loss**: Typically 0.02-0.05

### Performance Benchmarks
| Dataset Size | Training Time | Calibration Time | Hit Rate |
|--------------|---------------|------------------|----------|
| 1000 days    | 5-10 min      | 0.5-1 sec        | 0.008-0.012 |
| 2000 days    | 10-20 min     | 1-2 sec          | 0.008-0.012 |
| 5000 days    | 20-40 min     | 2-3 sec          | 0.008-0.012 |

### Troubleshooting Checklist
- [ ] Data file exists and has correct format
- [ ] At least 200+ days of price data
- [ ] No NaN values in price column
- [ ] Dependencies installed correctly
- [ ] Virtual environment activated
- [ ] Sufficient disk space
- [ ] Adequate memory available
