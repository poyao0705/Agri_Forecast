#!/usr/bin/env python3
"""
Standalone Transformer Model for VaR/ES Prediction (Class-based)

This is a simplified, standalone version that others can use with just:
1. src/models/transformer_var_es_paper_exact.py
2. src/utils/eval_tools.py
3. This script

Usage:
    python standalone_transformer_class.py --csv your_data.csv --alpha 0.01
"""

import sys
import os
import argparse
import time
import json
import hashlib
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.transformer_var_es_paper_exact import (
    build_inputs_from_prices,
    split_and_make_features,
    train_with_stride,
    evaluate_with_sliding_batch,
    BasicVaRTransformer,
    CONTEXT_LEN,
    TRAIN_STRIDE,
    ALPHA,
)
from utils.eval_tools import (
    fz0_per_step,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    plot_var_es_diagnostics,
    _choose_window_for_alpha,
)
from src.models.transformer_var_es_paper_exact import rolling_online_factors


class StandalonePredictor:
    """Standalone Transformer Predictor for VaR/ES Prediction."""

    def __init__(
        self,
        csv_path,
        alpha=0.01,
        feature_parity=True,
        train_frac=0.5,
        model_dir="models",
        output_dir="predictions",
    ):
        self.csv_path = csv_path
        self.alpha = alpha
        self.feature_parity = feature_parity
        self.train_frac = train_frac
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Model and data state
        self.model = None
        self.df = None
        self.data_hash = None
        self.feature_cols = None
        self.last_training_date = None
        self.calibration_factors = {"c_v": 1.0, "c_e": 1.0}

        # Load last training date if it exists
        self.load_training_info()

    def load_training_info(self):
        """Load last training date from persistent storage."""
        try:
            training_info_file = os.path.join(
                self.output_dir, "last_training_info.json"
            )
            if os.path.exists(training_info_file):
                with open(training_info_file, "r") as f:
                    info = json.load(f)
                    self.last_training_date = datetime.fromisoformat(
                        info["last_training_date"]
                    )
                    print(
                        f"✓ Loaded last training date: {self.last_training_date.strftime('%Y-%m-%d')}"
                    )
        except Exception as e:
            print(f"Note: Could not load last training date: {e}")
            self.last_training_date = None

    def save_training_info(self):
        """Save last training date to persistent storage."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            training_info_file = os.path.join(
                self.output_dir, "last_training_info.json"
            )

            info = {
                "last_training_date": datetime.now().isoformat(),
                "alpha": self.alpha,
                "csv_path": self.csv_path,
            }

            with open(training_info_file, "w") as f:
                json.dump(info, f, indent=2)

            print(f"✓ Saved training info to: {training_info_file}")
        except Exception as e:
            print(f"Warning: Could not save training info: {e}")

    def load_and_prepare_data(self):
        """Load and prepare data for the transformer model."""
        print(f"Loading data from: {self.csv_path}")

        # Load data
        df = pd.read_csv(self.csv_path)

        # Ensure we have the required 'close' column (case insensitive)
        close_col = None
        for col in df.columns:
            if col.lower() == "close":
                close_col = col
                break

        if close_col is None:
            raise ValueError("CSV must contain a 'close' column with price data")

        # Rename to lowercase for consistency
        df = df.rename(columns={close_col: "close"})

        # Keep only numeric columns and 'close' for processing
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "close" not in numeric_cols:
            numeric_cols.append("close")

        df = df[numeric_cols]

        # Build inputs (creates features from prices)
        df_processed = build_inputs_from_prices(df)

        print(f"Data loaded: {len(df_processed)} rows")
        print(f"Date range: {df_processed.index[0]} to {df_processed.index[-1]}")

        self.df = df_processed
        self.data_hash = self.compute_data_hash(df_processed)
        print(f"Data hash: {self.data_hash}")

        return df_processed

    def compute_data_hash(self, df):
        """Compute a hash of the data for model identification."""
        # Create a hash based on data characteristics
        hash_input = f"{len(df)}_{df['close'].iloc[-10:].sum():.2f}_{df['close'].iloc[:10].sum():.2f}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def split_data(self):
        """Split data into train and test sets using the proper pipeline."""
        # Use the proper feature creation and splitting pipeline
        X_train, y_train, X_test, y_test, meta = split_and_make_features(
            self.df, feature_parity=self.feature_parity, train_frac=self.train_frac
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(y_test)} samples")
        print(f"Features: {meta['feature_cols']}")

        self.feature_cols = meta["feature_cols"]

        return X_train, y_train, X_test, y_test

    def get_model_filename(self):
        """Generate model filename based on parameters and data hash."""
        feat_suffix = "parity" if self.feature_parity else "full"
        train_suffix = f"train{int(self.train_frac*100)}"
        alpha_suffix = f"a{int(self.alpha*1000):03d}"

        return f"transformer_{alpha_suffix}_{feat_suffix}_{train_suffix}_{self.data_hash}.pth"

    def train_model(self, force_retrain=False):
        """Train the transformer model or load existing model."""
        print("\n" + "=" * 50)
        print("MODEL TRAINING/LOADING")
        print("=" * 50)

        # Generate model filename
        model_filename = self.get_model_filename()
        model_path = os.path.join(self.model_dir, model_filename)

        # Check if model exists and we're not forcing retrain
        if os.path.exists(model_path) and not force_retrain:
            print(f"Loading existing model from: {model_path}")
            # We need to know the input dimension, so get it from feature_cols
            if self.feature_cols is None:
                # If feature_cols is not set, we need to split data first
                self.split_data()
            model = BasicVaRTransformer(input_dim=len(self.feature_cols))
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print("Model loaded successfully!")
            return model

        # Train new model
        print("Training new model...")
        X_train, y_train, X_test, y_test = self.split_data()

        input_dim = X_train.shape[1]
        model = train_with_stride(
            X_train=X_train,
            y_train=y_train,
            input_dim=input_dim,
            alpha=self.alpha,
            seq_len=CONTEXT_LEN,
            train_stride=TRAIN_STRIDE,
        )

        # Save model
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        print("Training completed!")

        return model

    def evaluate_model(self, calibrate=False):
        """Evaluate the model on test data."""
        print("\n" + "=" * 50)
        print("EVALUATING MODEL")
        print("=" * 50)

        X_train, y_train, X_test, y_test = self.split_data()

        # Get predictions
        # Combine train and test data for evaluation
        X_all = np.concatenate([X_train, X_test]).astype(np.float32)
        y_all = np.concatenate([y_train, y_test]).astype(np.float32)
        split_idx = len(X_train)

        v_pred, e_pred = evaluate_with_sliding_batch(
            self.model, X_all, y_all, split_idx, CONTEXT_LEN, 64
        )

        # Align predictions with actual returns
        y_aligned = y_all[split_idx : split_idx + len(v_pred)]

        # Apply calibration if requested
        if calibrate:
            print("Applying rolling online calibration...")
            # Use the exact same calibration method as the transformer script
            c_v, c_e = rolling_online_factors(y_aligned, v_pred, e_pred, self.alpha)
            v_eval = v_pred * c_v
            # Keep coherence: ES < VaR
            e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)
            print(
                f"Calibration factors - c_v: {np.mean(c_v):.4f}, c_e: {np.mean(c_e):.4f}"
            )
        else:
            v_eval, e_eval = v_pred, e_pred
            c_v, c_e = np.ones_like(v_pred), np.ones_like(e_pred)

        # Calculate metrics
        hits = (y_aligned <= v_eval).astype(int)
        hit_rate = hits.mean()

        # Statistical tests
        LR_pof, p_pof, _, _ = kupiec_pof(hits, self.alpha)
        LR_ind, p_ind = christoffersen_independence(hits)
        LR_cc, p_cc = christoffersen_cc(hits, self.alpha)
        fz0 = fz0_per_step(y_aligned, v_eval, e_eval, self.alpha)

        # Display results
        print(f"Test Results ({'calibrated' if calibrate else 'raw'}):")
        print(f"  Hit rate: {hit_rate:.4f} (Target: {self.alpha:.4f})")
        print(f"  Kupiec test: LR={LR_pof:.4f}, p={p_pof:.4f}")
        print(f"  Independence test: LR={LR_ind:.4f}, p={p_ind:.4f}")
        print(f"  Conditional coverage: LR={LR_cc:.4f}, p={p_cc:.4f}")
        print(f"  Average FZ0 loss: {fz0.mean():.6f}")

        return {
            "hit_rate": hit_rate,
            "kupiec_p": p_pof,
            "independence_p": p_ind,
            "conditional_p": p_cc,
            "fz0_loss": fz0.mean(),
            "v_pred": v_eval,
            "e_pred": e_eval,
            "y_true": y_aligned,
        }

    def predict_next_day(self):
        """Make prediction for the next day."""
        print("\n" + "=" * 50)
        print("NEXT DAY PREDICTION")
        print("=" * 50)

        # Prepare features for prediction
        df_processed = build_inputs_from_prices(self.df)
        mu_train = (
            df_processed["log_ret"]
            .iloc[: int(len(df_processed) * self.train_frac)]
            .mean()
        )
        df_processed["x_cov"] = (df_processed["log_ret"] - mu_train) ** 2

        # Add neg_xcov feature if needed
        if "neg_xcov" in self.feature_cols:
            df_processed["neg_xcov"] = df_processed["neg_ret"] * df_processed["x_cov"]

        # Get the most recent features
        recent_features = (
            df_processed[self.feature_cols].tail(CONTEXT_LEN).values.astype(np.float32)
        )
        x_input = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x_input)
            var_pred = float(prediction[0, 0].cpu())
            es_pred = float(prediction[0, 1].cpu())

        # Apply calibration factors if available
        var_calibrated = var_pred * self.calibration_factors["c_v"]
        es_calibrated = es_pred * self.calibration_factors["c_e"]

        # Ensure ES < VaR
        es_calibrated = min(es_calibrated, var_calibrated - 1e-8)

        # Display results
        print(f"VaR prediction: {var_calibrated:.4f} ({var_calibrated*100:.2f}%)")
        print(f"ES prediction: {es_calibrated:.4f} ({es_calibrated*100:.2f}%)")

        # Risk level assessment
        var_abs = abs(var_calibrated * 100)
        if var_abs < 1:
            risk_level = "LOW"
        elif var_abs < 3:
            risk_level = "MODERATE"
        elif var_abs < 5:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"

        print(f"Risk level: {risk_level}")

        return {"var": var_calibrated, "es": es_calibrated, "risk_level": risk_level}

    def should_retrain(self, retrain_days, mode):
        """Determine if model should be retrained based on mode and last training date."""
        if mode == "retrain":
            return True
        elif mode == "calibrate":
            return False
        elif mode == "auto":
            if self.last_training_date is None:
                return True
            days_since_training = (datetime.now() - self.last_training_date).days
            return days_since_training >= retrain_days
        return False

    def update_calibration_factors(self, calibration_window=100):
        """Update calibration factors using recent data."""
        print("Updating calibration factors...")
        start_time = time.time()

        # Use last calibration_window days for calibration
        df_recent = self.df.tail(calibration_window)

        # Prepare features for recent data
        df_processed = build_inputs_from_prices(df_recent)
        mu_train = df_processed["log_ret"].mean()
        df_processed["x_cov"] = (df_processed["log_ret"] - mu_train) ** 2

        # Generate raw predictions for recent data
        self.model.eval()
        v_raw, e_raw = [], []

        with torch.no_grad():
            for i in range(len(df_processed) - CONTEXT_LEN):
                features = (
                    df_processed[["x_cov"]]
                    .iloc[i : i + CONTEXT_LEN]
                    .values.astype(np.float32)
                )
                x_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                prediction = self.model(x_input)
                v_raw.append(float(prediction[0, 0].cpu()))
                e_raw.append(float(prediction[0, 1].cpu()))

        v_raw = np.array(v_raw)
        e_raw = np.array(e_raw)
        y_recent = df_processed["target_return"].iloc[CONTEXT_LEN:].values

        # Calculate calibration factors using the same method as transformer script
        c_v, c_e = rolling_online_factors(y_recent, v_raw, e_raw, self.alpha)

        # Use the most recent factors
        self.calibration_factors = {
            "c_v": float(c_v[-1]) if len(c_v) > 0 else 1.0,
            "c_e": float(c_e[-1]) if len(c_e) > 0 else 1.0,
        }

        calibration_time = time.time() - start_time
        print(f"✓ Calibration updated in {calibration_time:.2f} seconds")
        print(
            f"  Factors: c_v={self.calibration_factors['c_v']:.3f}, c_e={self.calibration_factors['c_e']:.3f}"
        )

        return calibration_time

    def make_live_prediction(self):
        """Make prediction for tomorrow using the model and calibration factors."""
        print("Making prediction for tomorrow...")

        # Prepare features for the most recent data
        df_processed = build_inputs_from_prices(self.df)
        mu_train = df_processed["log_ret"].iloc[:-1].mean()  # Exclude today
        df_processed["x_cov"] = (df_processed["log_ret"] - mu_train) ** 2

        # Get the most recent features
        recent_features = (
            df_processed[["x_cov"]].tail(CONTEXT_LEN).values.astype(np.float32)
        )
        x_input = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x_input)
            var_raw = float(prediction[0, 0].cpu())
            es_raw = float(prediction[0, 1].cpu())

        # Apply calibration
        var_calibrated = var_raw * self.calibration_factors["c_v"]
        es_calibrated = es_raw * self.calibration_factors["c_e"]

        # Ensure ES < VaR
        es_calibrated = min(es_calibrated, var_calibrated - 1e-8)

        return {
            "var_raw": var_raw,
            "es_raw": es_raw,
            "var_calibrated": var_calibrated,
            "es_calibrated": es_calibrated,
            "var_pct": var_calibrated * 100,
            "es_pct": es_calibrated * 100,
        }

    def display_live_results(self, results, mode, processing_time):
        """Display live prediction results."""
        print("\n" + "=" * 60)
        print("TOMORROW'S RISK PREDICTION")
        print("=" * 60)

        print(f"Mode: {mode.upper()}")
        print(f"Confidence Level: {(1-self.alpha)*100:.1f}%")
        print(f"VaR (Value at Risk): {results['var_pct']:.2f}%")
        print(f"ES (Expected Shortfall): {results['es_pct']:.2f}%")

        print(f"\nRaw vs Calibrated:")
        print(f"  Raw VaR: {results['var_raw']*100:.2f}%")
        print(f"  Calibrated VaR: {results['var_pct']:.2f}%")

        print(f"\nInterpretation:")
        print(
            f"• VaR: We expect the maximum loss tomorrow to be {abs(results['var_pct']):.2f}%"
        )
        print(
            f"• ES: If tomorrow is really bad, we expect to lose {abs(results['es_pct']):.2f}% on average"
        )

        # Risk level assessment
        var_abs = abs(results["var_pct"])
        if var_abs < 1:
            risk_level = "LOW"
        elif var_abs < 3:
            risk_level = "MODERATE"
        elif var_abs < 5:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"

        print(f"\nRisk Level: {risk_level}")
        print(f"Processing Time: {processing_time:.2f} seconds")

    def save_live_results(self, results, mode, processing_time):
        """Save live prediction results."""
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(self.output_dir, f"prediction_{timestamp}.json")

        full_results = {
            "prediction_date": datetime.now().strftime("%Y-%m-%d"),
            "target_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "mode": mode,
            "alpha": self.alpha,
            "processing_time": processing_time,
            **results,
        }

        with open(prediction_file, "w") as f:
            json.dump(full_results, f, indent=2)

        print(f"\n✓ Prediction saved to: {prediction_file}")
        return prediction_file

    def run_live_mode(self, mode="auto", retrain_days=7, calibration_window=100):
        """Run the predictor in live mode with hybrid retraining/calibration."""
        print("=" * 60)
        print("HYBRID LIVE PREDICTION MODE")
        print("=" * 60)

        # Determine if we should retrain
        should_retrain_model = self.should_retrain(retrain_days, mode)

        if should_retrain_model:
            print(f"Mode: FULL RETRAINING (using all data)")
            start_time = time.time()

            # Load data and train model
            self.load_and_prepare_data()
            self.model = self.train_model(force_retrain=True)

            # Save training info
            self.save_training_info()

            training_time = time.time() - start_time

            # Update calibration factors
            cal_time = self.update_calibration_factors(calibration_window)

            total_time = training_time + cal_time

        else:
            last_trained_str = (
                self.last_training_date.strftime("%Y-%m-%d")
                if self.last_training_date
                else "Never"
            )
            print(f"Mode: CALIBRATION (last trained: {last_trained_str})")

            # Load data
            self.load_and_prepare_data()

            # Try to load existing model
            model_filename = self.get_model_filename()
            model_path = os.path.join(self.model_dir, model_filename)

            if not os.path.exists(model_path):
                print(
                    "❌ No trained model found. Please run with mode='retrain' first."
                )
                return None

            # Load model
            self.model = self.train_model(force_retrain=False)

            # Update calibration factors
            cal_time = self.update_calibration_factors(calibration_window)

            total_time = cal_time

        # Make live prediction
        results = self.make_live_prediction()

        # Display and save results
        self.display_live_results(results, mode, total_time)
        self.save_live_results(results, mode, total_time)

        print("\n" + "=" * 60)
        print("SUCCESS! Tomorrow's risk prediction is ready.")
        print("=" * 60)

        return results

    def run_standard_mode(
        self, calibrate=False, predict_next=False, save_results=False
    ):
        """Run the predictor in standard evaluation mode."""
        # Load data
        self.load_and_prepare_data()

        # Train or load model
        self.model = self.train_model(force_retrain=False)

        # Evaluate model
        results = self.evaluate_model(calibrate=calibrate)

        # Make next day prediction if requested
        if predict_next:
            prediction = self.predict_next_day()

        # Save results if requested
        if save_results:
            # Implementation for saving evaluation results
            pass

        print("\n" + "=" * 50)
        print("COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        return results


def list_available_models(model_dir):
    """List available trained models."""
    if not os.path.exists(model_dir):
        print(f"No model directory found: {model_dir}")
        return

    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not models:
        print(f"No trained models found in: {model_dir}")
        return

    print(f"Available models in {model_dir}:")
    for model in sorted(models):
        print(f"  - {model}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Transformer Model for VaR/ES Prediction"
    )
    parser.add_argument(
        "--csv", required=False, help="Path to CSV file with 'close' column"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="VaR/ES confidence level (default: 0.01)",
    )
    parser.add_argument(
        "--feature_parity",
        action="store_true",
        help="Use feature parity mode (x_cov only)",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.5, help="Training fraction (default: 0.5)"
    )
    parser.add_argument(
        "--predict_next", action="store_true", help="Make prediction for next day"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to files"
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory to save/load models (default: models)",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if model exists",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available trained models and exit",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply calibration to improve hit rate accuracy",
    )

    # Hybrid live prediction options
    parser.add_argument(
        "--live_mode",
        action="store_true",
        help="Enable live prediction mode with auto-retraining",
    )
    parser.add_argument(
        "--mode",
        choices=["retrain", "calibrate", "auto"],
        default="auto",
        help="Live mode: retrain, calibrate, or auto (default: auto)",
    )
    parser.add_argument(
        "--retrain_days",
        type=int,
        default=7,
        help="Days between auto-retraining in live mode (default: 7)",
    )
    parser.add_argument(
        "--calibration_window",
        type=int,
        default=100,
        help="Days to use for calibration in live mode (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save live predictions (default: predictions)",
    )

    args = parser.parse_args()

    # Handle list models option
    if args.list_models:
        list_available_models(args.model_dir)
        return

    # Validate required arguments
    if not args.csv:
        parser.error("--csv is required (unless using --list_models)")

    try:
        # Create predictor
        predictor = StandalonePredictor(
            csv_path=args.csv,
            alpha=args.alpha,
            feature_parity=args.feature_parity,
            train_frac=args.train_frac,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )

        # Run in appropriate mode
        if args.live_mode:
            predictor.run_live_mode(
                mode=args.mode,
                retrain_days=args.retrain_days,
                calibration_window=args.calibration_window,
            )
        else:
            predictor.run_standard_mode(
                calibrate=args.calibrate,
                predict_next=args.predict_next,
                save_results=args.save_results,
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
