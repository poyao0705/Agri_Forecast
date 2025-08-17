#!/usr/bin/env python3
"""
Hybrid Live Prediction Script
Choose between full retraining or calibration for daily predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import time
import json
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.models.transformer_var_es_paper_exact import (
    build_inputs_from_prices,
    train_with_stride,
    rolling_online_factors,
    BasicVaRTransformer,
    CONTEXT_LEN,
    TRAIN_STRIDE,
)


class HybridPredictor:
    def __init__(self, csv_path, alpha=0.01):
        self.csv_path = csv_path
        self.alpha = alpha
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.last_training_date = None
        self.calibration_factors = {"c_v": 1.0, "c_e": 1.0}

        # Load last training date from file if it exists
        self.load_last_training_date()

    def load_last_training_date(self):
        """Load last training date from persistent storage."""
        try:
            training_info_file = "hybrid_predictions/last_training_info.json"
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

    def save_last_training_date(self):
        """Save last training date to persistent storage."""
        try:
            os.makedirs("hybrid_predictions", exist_ok=True)
            training_info_file = "hybrid_predictions/last_training_info.json"

            info = {
                "last_training_date": self.last_training_date.isoformat(),
                "alpha": self.alpha,
                "csv_path": self.csv_path,
            }

            with open(training_info_file, "w") as f:
                json.dump(info, f, indent=2)

            print(f"✓ Saved training info to: {training_info_file}")
        except Exception as e:
            print(f"Warning: Could not save training info: {e}")

    def prepare_data(self, use_all_data=True):
        """Prepare data for training or prediction."""
        print("Preparing data...")

        df = pd.read_csv(self.csv_path)
        df = build_inputs_from_prices(df)

        if use_all_data:
            # Use all data for training
            mu_train = df["log_ret"].mean()
        else:
            # Use data up to yesterday for calibration
            mu_train = df["log_ret"][:-1].mean()

        df = df.copy()
        df["x_cov"] = (df["log_ret"] - mu_train) ** 2

        feature_cols = ["x_cov"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["target_return"].values.astype(np.float32)

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X).astype(np.float32)

        print(f"✓ Prepared {len(X_scaled)} days of data")
        return X_scaled, y, scaler, feature_cols

    def train_model(self, X, y):
        """Train the model from scratch."""
        print(f"Training model on all data (alpha={self.alpha})...")
        start_time = time.time()

        input_dim = X.shape[1]
        model = train_with_stride(
            X_train=X,
            y_train=y,
            input_dim=input_dim,
            alpha=self.alpha,
            seq_len=CONTEXT_LEN,
            train_stride=TRAIN_STRIDE,
        )

        training_time = time.time() - start_time
        print(f"✓ Model trained in {training_time:.2f} seconds")

        return model, training_time

    def update_calibration(self, X, y):
        """Update calibration factors using recent data."""
        print("Updating calibration factors...")
        start_time = time.time()

        # Use last 100 days for calibration
        recent_X = X[-100:]
        recent_y = y[-100:]

        # Generate raw predictions for recent data
        model = self.model
        model.eval()

        v_raw, e_raw = [], []
        with torch.no_grad():
            for i in range(len(recent_X) - CONTEXT_LEN):
                x_input = torch.tensor(
                    recent_X[i : i + CONTEXT_LEN], dtype=torch.float32
                ).unsqueeze(0)
                prediction = model(x_input)
                v_raw.append(float(prediction[0, 0].cpu()))
                e_raw.append(float(prediction[0, 1].cpu()))

        v_raw = np.array(v_raw)
        e_raw = np.array(e_raw)
        y_recent = recent_y[CONTEXT_LEN:]

        # Calculate calibration factors
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

    def predict_tomorrow(self, mode="auto"):
        """
        Make prediction for tomorrow.

        Args:
            mode: 'retrain' (full retraining), 'calibrate' (use calibration), 'auto' (choose based on time since last training)
        """
        print("=" * 60)
        print("HYBRID LIVE PREDICTION")
        print("=" * 60)

        today = datetime.now().strftime("%Y-%m-%d")

        # Check if we should retrain
        should_retrain = False
        if mode == "retrain":
            should_retrain = True
        elif mode == "calibrate":
            should_retrain = False
        elif mode == "auto":
            # Retrain if it's been more than 7 days or if no model exists
            if self.last_training_date is None:
                should_retrain = True
            else:
                days_since_training = (datetime.now() - self.last_training_date).days
                should_retrain = days_since_training >= 7

        if should_retrain:
            print(f"Mode: FULL RETRAINING (using all data up to {today})")
            X, y, self.scaler, self.feature_cols = self.prepare_data(use_all_data=True)
            self.model, training_time = self.train_model(X, y)
            self.last_training_date = datetime.now()
            self.save_last_training_date()  # Save after successful training

            # Update calibration factors
            cal_time = self.update_calibration(X, y)
            total_time = training_time + cal_time

        else:
            last_trained_str = (
                self.last_training_date.strftime("%Y-%m-%d")
                if self.last_training_date
                else "Never"
            )
            print(f"Mode: CALIBRATION (last trained: {last_trained_str})")

            if self.model is None:
                print(
                    "❌ No trained model found. Please run with mode='retrain' first."
                )
                return None

            # Prepare data for calibration
            X, y, _, _ = self.prepare_data(use_all_data=True)

            # Update calibration factors
            cal_time = self.update_calibration(X, y)
            total_time = cal_time

        # Make prediction for tomorrow
        print(f"\nMaking prediction for tomorrow...")
        recent_features = X[-CONTEXT_LEN:]
        x_input = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x_input)
            var_raw = float(prediction[0, 0].cpu())
            es_raw = float(prediction[0, 1].cpu())

        # Apply calibration
        var_calibrated = var_raw * self.calibration_factors["c_v"]
        es_calibrated = es_raw * self.calibration_factors["c_e"]

        # Format results
        results = {
            "prediction_date": today,
            "target_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "mode": "retrain" if should_retrain else "calibrate",
            "var_raw": var_raw,
            "es_raw": es_raw,
            "var_calibrated": var_calibrated,
            "es_calibrated": es_calibrated,
            "var_pct": var_calibrated * 100,
            "es_pct": es_calibrated * 100,
            "calibration_factors": self.calibration_factors,
            "processing_time": total_time,
            "last_training_date": (
                self.last_training_date.strftime("%Y-%m-%d")
                if self.last_training_date
                else None
            ),
        }

        # Display results
        self.display_results(results)

        # Save results
        self.save_results(results)

        return results

    def display_results(self, results):
        """Display prediction results."""
        print("\n" + "=" * 60)
        print("TOMORROW'S RISK PREDICTION")
        print("=" * 60)

        print(f"Mode: {results['mode'].upper()}")
        print(f"Confidence Level: {(1-self.alpha)*100:.1f}%")
        print(f"VaR (Value at Risk): {results['var_pct']:.2f}%")
        print(f"ES (Expected Shortfall): {results['es_pct']:.2f}%")

        print(f"\nRaw vs Calibrated:")
        print(f"  Raw VaR: {results['var_raw']*100:.2f}%")
        print(f"  Calibrated VaR: {results['var_pct']:.2f}%")
        print(f"  Calibration factor: {results['calibration_factors']['c_v']:.3f}")

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
        print(f"Processing Time: {results['processing_time']:.2f} seconds")

    def save_results(self, results, output_dir="hybrid_predictions"):
        """Save prediction results."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(output_dir, f"prediction_{timestamp}.json")

        with open(prediction_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Prediction saved to: {prediction_file}")
        return prediction_file


def main():
    """Main function with command line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Live Prediction")
    parser.add_argument(
        "--mode",
        choices=["retrain", "calibrate", "auto"],
        default="auto",
        help="Prediction mode",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="VaR/ES confidence level"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/merged_data_with_realised_volatility.csv",
        help="Path to CSV file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Data file not found: {args.csv}")
        return

    # Create predictor and run
    predictor = HybridPredictor(args.csv, alpha=args.alpha)
    results = predictor.predict_tomorrow(mode=args.mode)

    if results:
        print("\n" + "=" * 60)
        print("SUCCESS! Tomorrow's risk prediction is ready.")
        print("=" * 60)


if __name__ == "__main__":
    main()
