import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Import both models
from model_rnn import RNNVaRES, FZ0Loss, create_sequences_with_overlap, train_with_debugging
from model_v0_1 import ImprovedVaRTransformer, ImprovedFZ0Loss, train_with_advanced_features

def load_and_prepare_data():
    """Load and prepare data consistently for both models"""
    df = pd.read_csv("model/data/merged_data_with_realised_volatility.csv")

    df["return"] = df["close"].pct_change()
    df["squared_return"] = df["return"] ** 2
    df["target_return"] = df["return"].shift(-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_ma"] = df["return"].rolling(window=5).mean()
    df["vol_ma"] = df["return"].rolling(window=21).std()

    df.dropna(inplace=True)

    features = [
        "return", "squared_return", "log_return", "return_ma", "vol_ma",
        "gk_vol_1d", "gk_vol_21d", "weighted_tavg", "weighted_prcp",
        "Fed_Rate", "GDP", "CPI",
    ]

    X = df[features].values
    y = df["target_return"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "model/saved_models/scaler.pkl")

    # Create sequences
    X_seq, y_seq = create_sequences_with_overlap(X_scaled, y, seq_len=21, overlap=0.3)

    # Split data consistently
    split_idx = int(0.8 * len(X_seq))
    X_train_val, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train_val, y_test = y_seq[:split_idx], y_seq[split_idx:]

    return X_train_val, y_train_val, X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance with consistent metrics"""
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
    
    y_true = y_test
    predicted_var = y_pred[:, 0]
    predicted_es = y_pred[:, 1]

    # Basic statistics
    hits = (y_true <= predicted_var).astype(int)
    hit_rate = hits.mean()
    n = len(y_true)
    x = hits.sum()

    # Kupiec Test
    p_hat = x / n if x > 0 else 0.001
    if p_hat == 0 or p_hat == 1:
        LR_pof = float("inf")
        p_value = 0.0
    else:
        LR_pof = -2 * (
            np.log((1 - 0.05) ** (n - x) * 0.05**x)
            - np.log((1 - p_hat) ** (n - x) * p_hat**x)
        )
        p_value = 1 - chi2.cdf(LR_pof, df=1)

    # ES evaluation
    breach_returns = y_true[hits == 1]
    avg_actual_es = breach_returns.mean() if len(breach_returns) > 0 else np.nan
    avg_predicted_es = predicted_es[hits == 1].mean() if len(breach_returns) > 0 else np.nan

    # Additional metrics
    var_quantile = np.quantile(y_true, 0.05)
    es_quantile = y_true[y_true <= var_quantile].mean()
    constraint_violations = np.sum(predicted_var < predicted_es)

    results = {
        'model': model_name,
        'hit_rate': hit_rate,
        'breaches': x,
        'total_samples': n,
        'LR_pof': LR_pof,
        'p_value': p_value,
        'avg_actual_es': avg_actual_es,
        'avg_predicted_es': avg_predicted_es,
        'empirical_var': var_quantile,
        'empirical_es': es_quantile,
        'constraint_violations': constraint_violations,
        'var_mean': predicted_var.mean(),
        'var_std': predicted_var.std(),
        'es_mean': predicted_es.mean(),
        'es_std': predicted_es.std()
    }

    return results, predicted_var, predicted_es, hits

def plot_comparison(results_rnn, results_transformer, var_rnn, var_transformer, hits_rnn, hits_transformer):
    """Plot comparison between models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. VaR predictions comparison
    axes[0, 0].plot(var_rnn, label='RNN VaR', alpha=0.7)
    axes[0, 0].plot(var_transformer, label='Transformer VaR', alpha=0.7)
    axes[0, 0].set_title('VaR Predictions Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Hit rate comparison
    window = min(20, max(5, len(hits_rnn) // 3))
    rolling_hit_rnn = pd.Series(hits_rnn).rolling(window=window).mean()
    rolling_hit_transformer = pd.Series(hits_transformer).rolling(window=window).mean()
    
    axes[0, 1].plot(rolling_hit_rnn, label=f'RNN ({window} days)', alpha=0.7)
    axes[0, 1].plot(rolling_hit_transformer, label=f'Transformer ({window} days)', alpha=0.7)
    axes[0, 1].axhline(y=0.05, color='red', linestyle='--', label='Expected (0.05)')
    axes[0, 1].set_title('Rolling Hit Rate Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. VaR distribution comparison
    axes[0, 2].hist(var_rnn, bins=30, alpha=0.7, label='RNN', density=True)
    axes[0, 2].hist(var_transformer, bins=30, alpha=0.7, label='Transformer', density=True)
    axes[0, 2].set_title('VaR Distribution Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Performance metrics comparison
    metrics = ['hit_rate', 'LR_pof', 'p_value']
    rnn_metrics = [results_rnn[m] for m in metrics]
    transformer_metrics = [results_transformer[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, rnn_metrics, width, label='RNN', alpha=0.7)
    axes[1, 0].bar(x + width/2, transformer_metrics, width, label='Transformer', alpha=0.7)
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Performance Metrics Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 5. ES comparison
    breach_mask_rnn = hits_rnn == 1
    breach_mask_transformer = hits_transformer == 1
    
    if breach_mask_rnn.sum() > 0 and breach_mask_transformer.sum() > 0:
        axes[1, 1].scatter(var_rnn[breach_mask_rnn], y_test[breach_mask_rnn], 
                           alpha=0.7, label='RNN Breaches', s=20)
        axes[1, 1].scatter(var_transformer[breach_mask_transformer], y_test[breach_mask_transformer], 
                           alpha=0.7, label='Transformer Breaches', s=20)
        axes[1, 1].set_xlabel('Predicted VaR')
        axes[1, 1].set_ylabel('Actual Returns')
        axes[1, 1].set_title('VaR vs Actual Returns (Breaches)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # 6. Summary table
    axes[1, 2].axis('off')
    summary_text = f"""
    Model Comparison Summary
    
    RNN Model:
    - Hit Rate: {results_rnn['hit_rate']:.4f}
    - Breaches: {results_rnn['breaches']}/{results_rnn['total_samples']}
    - LR_pof: {results_rnn['LR_pof']:.4f}
    - P-value: {results_rnn['p_value']:.4f}
    - Constraint Violations: {results_rnn['constraint_violations']}
    
    Transformer Model:
    - Hit Rate: {results_transformer['hit_rate']:.4f}
    - Breaches: {results_transformer['breaches']}/{results_transformer['total_samples']}
    - LR_pof: {results_transformer['LR_pof']:.4f}
    - P-value: {results_transformer['p_value']:.4f}
    - Constraint Violations: {results_transformer['constraint_violations']}
    """
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

def main():
    """Main comparison function"""
    print("Loading and preparing data...")
    X_train_val, y_train_val, X_test, y_test = load_and_prepare_data()
    
    print(f"Training samples: {len(X_train_val)}, Test samples: {len(X_test)}")
    
    # Train RNN model
    print("\n" + "="*50)
    print("TRAINING RNN MODEL")
    print("="*50)
    model_rnn, _, _ = train_with_debugging(
        X_train_val, y_train_val, input_dim=X_train_val.shape[2], max_epochs=150
    )
    
    # Train Transformer model
    print("\n" + "="*50)
    print("TRAINING TRANSFORMER MODEL")
    print("="*50)
    model_transformer, _, _ = train_with_advanced_features(
        X_train_val, y_train_val, input_dim=X_train_val.shape[2], max_epochs=150
    )
    
    # Evaluate both models
    print("\n" + "="*50)
    print("EVALUATING MODELS")
    print("="*50)
    
    results_rnn, var_rnn, es_rnn, hits_rnn = evaluate_model(
        model_rnn, X_test, y_test, "RNN"
    )
    
    results_transformer, var_transformer, es_transformer, hits_transformer = evaluate_model(
        model_transformer, X_test, y_test, "Transformer"
    )
    
    # Print results
    print("\nRNN Model Results:")
    print(f"Hit Rate: {results_rnn['hit_rate']:.4f} (Expected: 0.05)")
    print(f"Breaches: {results_rnn['breaches']} out of {results_rnn['total_samples']}")
    print(f"Kupiec Test: LR_pof={results_rnn['LR_pof']:.4f}, p-value={results_rnn['p_value']:.4f}")
    print(f"Test Result: {'REJECT' if results_rnn['p_value'] < 0.05 else 'ACCEPT'} null hypothesis")
    print(f"ES Constraint Violations: {results_rnn['constraint_violations']}")
    
    print("\nTransformer Model Results:")
    print(f"Hit Rate: {results_transformer['hit_rate']:.4f} (Expected: 0.05)")
    print(f"Breaches: {results_transformer['breaches']} out of {results_transformer['total_samples']}")
    print(f"Kupiec Test: LR_pof={results_transformer['LR_pof']:.4f}, p-value={results_transformer['p_value']:.4f}")
    print(f"Test Result: {'REJECT' if results_transformer['p_value'] < 0.05 else 'ACCEPT'} null hypothesis")
    print(f"ES Constraint Violations: {results_transformer['constraint_violations']}")
    
    # Plot comparison
    plot_comparison(results_rnn, results_transformer, var_rnn, var_transformer, hits_rnn, hits_transformer)
    
    # Determine winner
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    # Compare hit rates (closer to 0.05 is better)
    rnn_hit_error = abs(results_rnn['hit_rate'] - 0.05)
    transformer_hit_error = abs(results_transformer['hit_rate'] - 0.05)
    
    print(f"Hit Rate Error (closer to 0.05 is better):")
    print(f"  RNN: {rnn_hit_error:.4f}")
    print(f"  Transformer: {transformer_hit_error:.4f}")
    
    # Compare p-values (higher is better for accepting null hypothesis)
    print(f"\nP-value (higher is better):")
    print(f"  RNN: {results_rnn['p_value']:.4f}")
    print(f"  Transformer: {results_transformer['p_value']:.4f}")
    
    # Compare constraint violations (lower is better)
    print(f"\nES Constraint Violations (lower is better):")
    print(f"  RNN: {results_rnn['constraint_violations']}")
    print(f"  Transformer: {results_transformer['constraint_violations']}")
    
    # Overall winner
    rnn_score = (1 - rnn_hit_error) + results_rnn['p_value'] + (1 / (1 + results_rnn['constraint_violations']))
    transformer_score = (1 - transformer_hit_error) + results_transformer['p_value'] + (1 / (1 + results_transformer['constraint_violations']))
    
    print(f"\nOverall Score (higher is better):")
    print(f"  RNN: {rnn_score:.4f}")
    print(f"  Transformer: {transformer_score:.4f}")
    
    if rnn_score > transformer_score:
        print("\n🏆 RNN Model performs better overall!")
    elif transformer_score > rnn_score:
        print("\n🏆 Transformer Model performs better overall!")
    else:
        print("\n🤝 Both models perform similarly!")

if __name__ == "__main__":
    main() 