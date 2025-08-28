#!/usr/bin/env python3
"""
Standalone Transformer Model for VaR/ES Prediction

A comprehensive class-based interface for training transformer models and making
VaR/ES predictions with automatic calibration and persistence.

Features:
- Full dataset training (no train/test split)
- Online calibration for improved accuracy
- Model persistence and reuse
- Live prediction mode
- Comprehensive backtesting

Usage:
    python standalone_transformer_class.py --csv your_data.csv --alpha 0.01 --live_mode
"""

import sys
import os
import argparse
import time
import json
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import transformer model components
from models.transformer_var_es_paper_exact import (
    build_inputs_from_prices,
    split_and_make_features,
    train_with_stride,
    evaluate_with_sliding_batch,
    BasicVaRTransformer,
    CONTEXT_LEN,
    TRAIN_STRIDE,
    ALPHA,
    rolling_online_factors,
)

from utils.eval_tools import (
    fz0_per_step,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    plot_var_es_diagnostics,
    _choose_window_for_alpha,
)


class StandalonePredictor:
    """
    Standalone Transformer Predictor for VaR/ES Prediction.

    A comprehensive class for training transformer models and making predictions
    with automatic calibration, persistence, and live prediction capabilities.

    Attributes:
        csv_path (str): Path to the CSV file containing price data
        alpha (float): VaR/ES confidence level (e.g., 0.01 for 1%)
        feature_parity (bool): Whether to use feature parity mode (x_cov only)
        train_frac (float): Training fraction (1.0 for full dataset training)
        model_dir (str): Directory to save/load models
        output_dir (str): Directory for output files and logs
        model: The trained transformer model
        df: Processed DataFrame with features
        data_hash (str): Hash of data for model naming
        feature_cols (list): List of feature column names
        last_training_date: Timestamp of last training
        calibration_factors (dict): Current calibration factors
        meta (dict): Training metadata (scaling, features, etc.)
    """

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

        # Initialize model and data state
        self.model = None
        self.df = None
        self.data_hash = None
        self.feature_cols = None
        self.last_training_date = None
        self.calibration_factors = {"c_v": 1.0, "c_e": 1.0}
        self.meta = None

        # Load previous training information if available
        self.load_training_info()

    # ============================
    # Persistence and Metadata
    # ============================

    def load_training_info(self):
        """
        Load previous training information from saved metadata.

        Loads the last training date and model path from the output directory.
        This enables model reuse and calibration mode functionality.
        """
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
                self.last_model_path = info.get("model_path")
                print(
                    f"✓ Loaded last training date: {self.last_training_date.strftime('%Y-%m-%d')}"
                )
                if self.last_model_path:
                    print(f"✓ Last model path: {self.last_model_path}")
        except Exception as e:
            print(f"Note: Could not load last training info: {e}")
            self.last_training_date = None
            self.last_model_path = None

    def save_training_info(self, model_path=None):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            training_info_file = os.path.join(
                self.output_dir, "last_training_info.json"
            )
            info = {
                "last_training_date": datetime.now().isoformat(),
                "alpha": self.alpha,
                "csv_path": self.csv_path,
                "model_path": model_path or "",  # <-- NEW
                "train_frac": self.train_frac,  # optional breadcrumb
            }
            with open(training_info_file, "w") as f:
                json.dump(info, f, indent=2)
            print(f"✓ Saved training info to: {training_info_file}")
        except Exception as e:
            print(f"Warning: Could not save training info: {e}")

    def load_model_from_path(self, model_path: str):
        """Load weights + meta from an explicit path (used by calibrate mode)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        meta_path = model_path.replace(".pth", ".meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        model = BasicVaRTransformer(input_dim=len(self.feature_cols))
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Model loaded from: {model_path}")
        return model

    # ============================
    # Data Loading and Processing
    # ============================

    def load_and_prepare_data(self):
        """
        Load and prepare data from CSV file.

        Loads price data, validates required columns, and processes features.
        Automatically handles case-insensitive column names and data validation.

        Returns:
            pd.DataFrame: Processed DataFrame with engineered features

        Raises:
            ValueError: If 'close' column is missing
        """
        print(f"Loading data from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        # Ensure 'close' exists (case-insensitive); keep only numeric + close
        close_col = None
        for col in df.columns:
            if col.lower() == "close":
                close_col = col
                break
        if close_col is None:
            raise ValueError("CSV must contain a 'close' column with price data")

        df = df.rename(columns={close_col: "close"})
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "close" not in numeric_cols:
            numeric_cols.append("close")
        df = df[numeric_cols]

        # Build inputs from prices (keeps 'close' and adds engineered cols)
        df_processed = build_inputs_from_prices(df)

        print(f"Data loaded: {len(df_processed)} rows")
        print(f"Index range: {df_processed.index[0]} → {df_processed.index[-1]}")

        self.df = df_processed
        self.data_hash = self.compute_data_hash(df_processed)
        print(f"Data hash: {self.data_hash}")
        return df_processed

    @staticmethod
    def compute_data_hash(df):
        """
        Compute a hash of the data for model naming.

        Creates a unique identifier based on data length and price characteristics
        to ensure model files are properly named and can be reused.

        Args:
            df (pd.DataFrame): DataFrame with 'close' column

        Returns:
            str: 8-character hexadecimal hash
        """
        hash_input = f"{len(df)}_{df['close'].iloc[-10:].sum():.2f}_{df['close'].iloc[:10].sum():.2f}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def split_data(self):
        # Uses pipeline’s canonical feature-making (parity vs full) and scaling
        X_train, y_train, X_test, y_test, meta = split_and_make_features(
            self.df, feature_parity=self.feature_parity, train_frac=self.train_frac
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set:     {len(y_test)} samples")
        print(f"Features:     {meta['feature_cols']}")
        self.feature_cols = meta["feature_cols"]
        self.meta = meta
        return X_train, y_train, X_test, y_test

    # Exact training-time transform applied to any df already passed through build_inputs_from_prices
    def _transform_with_meta(self, df_processed: pd.DataFrame) -> np.ndarray:
        if not self.meta:
            raise RuntimeError(
                "Meta not loaded. Train or load a model (with .meta.json) first."
            )
        mu_train = float(self.meta["mu_train"])
        feats = df_processed.copy()
        feats["x_cov"] = (feats["log_ret"] - mu_train) ** 2
        feats["neg_xcov"] = feats["neg_ret"] * feats["x_cov"]
        X = feats[self.meta["feature_cols"]].values.astype(np.float32)
        mean = np.asarray(self.meta["scaler_mean"], dtype=np.float32)
        scale = np.asarray(self.meta["scaler_scale"], dtype=np.float32)
        scale_safe = np.where(scale == 0.0, 1.0, scale)
        X_std = (X - mean) / scale_safe
        return X_std

    # ============================
    # Model Training and Loading
    # ============================
    def get_model_filename(self):
        """
        Generate filename for model persistence.

        Creates a unique filename based on model parameters and data hash
        to ensure proper model versioning and reuse.

        Returns:
            str: Model filename with parameter encoding
        """
        feat_suffix = "parity" if self.feature_parity else "full"
        train_suffix = f"train{int(self.train_frac*100)}"
        alpha_suffix = f"a{int(self.alpha*1000):03d}"
        return f"transformer_{alpha_suffix}_{feat_suffix}_{train_suffix}_{self.data_hash}.pth"

    def train_model(self, force_retrain=False):
        """
        Train or load a transformer model.

        Either loads an existing model from disk or trains a new one.
        Automatically handles model persistence and metadata management.

        Args:
            force_retrain (bool): If True, ignore existing models and retrain

        Returns:
            BasicVaRTransformer: The trained/loaded model
        """
        print("\n" + "=" * 50)
        print("MODEL TRAINING/LOADING")
        print("=" * 50)

        model_filename = self.get_model_filename()
        model_path = os.path.join(self.model_dir, model_filename)
        meta_path = model_path.replace(".pth", ".meta.json")

        # Try to load existing
        if os.path.exists(model_path) and not force_retrain:
            print(f"Loading existing model from: {model_path}")
            # Load meta (preferred)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.meta = json.load(f)
                self.feature_cols = self.meta["feature_cols"]
            else:
                # Fallback: split to reconstruct meta
                self.split_data()

            model = BasicVaRTransformer(input_dim=len(self.feature_cols))
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print("Model loaded successfully!")
            self.model = model  # Assign to self.model
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

        # Save model + meta
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)
        print(f"Model saved to: {model_path}")
        print(f"Meta  saved to: {meta_path}")
        print("Training completed!")
        self.model = model  # Assign to self.model
        return model

    # ============================
    # Model Evaluation
    # ============================
    def evaluate_model(self, calibrate=False):
        """
        Evaluate model performance on test data.

        Performs comprehensive backtesting with statistical tests including
        Kupiec test, Christoffersen independence test, and FZ0 loss calculation.

        Args:
            calibrate (bool): Whether to apply online calibration

        Returns:
            dict: Dictionary containing evaluation metrics and predictions
        """
        print("\n" + "=" * 50)
        print("EVALUATING MODEL")
        print("=" * 50)

        X_train, y_train, X_test, y_test = self.split_data()
        X_all = np.concatenate([X_train, X_test]).astype(np.float32)
        y_all = np.concatenate([y_train, y_test]).astype(np.float32)
        split_idx = len(X_train)

        v_pred, e_pred = evaluate_with_sliding_batch(
            self.model, X_all, y_all, split_idx, CONTEXT_LEN, 32
        )

        # Align predictions with actual returns (1-step labels)
        y_aligned = y_all[split_idx : split_idx + len(v_pred)]

        # Safety: lengths must match
        assert (
            len(v_pred) == len(e_pred) == len(y_aligned)
        ), "Eval: pred/label length mismatch"

        # Optional calibration (causal: rolling history up to t-1)
        if calibrate:
            print("Applying rolling online calibration...")
            c_v, c_e = rolling_online_factors(y_aligned, v_pred, e_pred, self.alpha)
            v_eval = v_pred * c_v
            e_eval = np.minimum(e_pred * c_e, v_eval - 1e-8)  # ES < VaR
            print(
                f"Calibration factors - c_v: {np.mean(c_v):.4f}, c_e: {np.mean(c_e):.4f}"
            )
        else:
            v_eval, e_eval = v_pred, e_pred

        # Metrics
        hits = (y_aligned <= v_eval).astype(int)
        hit_rate = hits.mean()
        LR_pof, p_pof, _, _ = kupiec_pof(hits, self.alpha)
        LR_ind, p_ind = christoffersen_independence(hits)
        LR_cc, p_cc = christoffersen_cc(hits, self.alpha)
        fz0 = fz0_per_step(y_aligned, v_eval, e_eval, self.alpha)

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

    # ============================
    # Live Prediction and Calibration
    # ============================

    def predict_next_day(self):
        """
        Make prediction for the next trading day.

        Uses the most recent data window to predict VaR and ES for tomorrow.
        Applies current calibration factors and provides risk level assessment.

        Returns:
            dict: Dictionary with VaR, ES predictions and risk level
        """
        print("\n" + "=" * 50)
        print("NEXT DAY PREDICTION")
        print("=" * 50)

        dfp = self.df
        X_std = self._transform_with_meta(dfp)

        # most recent context window (standardized, correct features)
        recent_features = X_std[-CONTEXT_LEN:]
        x_input = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_input)
            var_pred = float(pred[0, 0].cpu())
            es_pred = float(pred[0, 1].cpu())

        var_calibrated = var_pred * self.calibration_factors["c_v"]
        es_calibrated = es_pred * self.calibration_factors["c_e"]
        es_calibrated = min(es_calibrated, var_calibrated - 1e-8)  # coherence

        print(f"VaR prediction: {var_calibrated:.4f} ({var_calibrated*100:.2f}%)")
        print(f"ES prediction:  {es_calibrated:.4f} ({es_calibrated*100:.2f}%)")

        # Simple risk band
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

    def update_calibration_factors(self, calibration_window=3000):
        """
        Update calibration factors using recent history.

        Computes rolling online calibration factors using the most recent data
        to improve prediction accuracy. Uses causal, 1-step aligned windows.

        Args:
            calibration_window (int): Number of recent observations to use

        Returns:
            float: Time taken for calibration update
        """
        print("Updating calibration factors...")
        start_time = time.time()

        # Use last `calibration_window` rows from already-processed frame
        df_recent = self.df.tail(calibration_window)
        # dfp = build_inputs_from_prices(df_recent)
        dfp = df_recent
        X_std = self._transform_with_meta(dfp)

        # Sliding one-step windows (predict t+1 from context ending at t)
        nwin = len(X_std) - CONTEXT_LEN  # <-- correct count for next-day forecasts
        if nwin <= 0:
            print("Not enough recent data for calibration; keeping factors at 1.0")
            return 0.0

        self.model.eval()
        v_raw, e_raw = [], []
        with torch.no_grad():
            for i in range(nwin):
                xwin = X_std[i : i + CONTEXT_LEN]
                x_input = torch.tensor(xwin, dtype=torch.float32).unsqueeze(0)
                pred = self.model(x_input)
                v_raw.append(float(pred[0, 0].cpu()))
                e_raw.append(float(pred[0, 1].cpu()))
        v_raw = np.asarray(v_raw, dtype=float)
        e_raw = np.asarray(e_raw, dtype=float)

        # 1-step label alignment: window i (ending at t) -> label y[t+1]
        y_arr = dfp["target_return"].values
        y_recent = y_arr[CONTEXT_LEN : CONTEXT_LEN + nwin]

        assert (
            len(v_raw) == len(e_raw) == len(y_recent)
        ), f"Calib: pred/label length mismatch ({len(v_raw)}, {len(e_raw)}, {len(y_recent)})"

        # Rolling online factors (history up to t-1)
        c_v, c_e = rolling_online_factors(y_recent, v_raw, e_raw, self.alpha)

        # Store most recent values
        self.calibration_factors = {
            "c_v": float(c_v[-1]) if len(c_v) else 1.0,
            "c_e": float(c_e[-1]) if len(c_e) else 1.0,
        }

        calibration_time = time.time() - start_time
        print(f"✓ Calibration updated in {calibration_time:.2f} seconds")
        print(
            f"  Factors: c_v={self.calibration_factors['c_v']:.3f}, c_e={self.calibration_factors['c_e']:.3f}"
        )
        return calibration_time

    def make_live_prediction(self):
        """
        Make prediction for tomorrow using current model and calibration factors.

        Uses the most recent data window to predict VaR and ES for the next trading day.
        Applies current calibration factors for improved accuracy.

        Returns:
            dict: Dictionary with raw and calibrated predictions
        """
        print("Making prediction for tomorrow...")
        # dfp = build_inputs_from_prices(self.df)
        dfp = self.df
        X_std = self._transform_with_meta(dfp)
        recent_features = X_std[-CONTEXT_LEN:]
        x_input = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_input)
            var_raw = float(pred[0, 0].cpu())
            es_raw = float(pred[0, 1].cpu())

        var_calibrated = var_raw * self.calibration_factors["c_v"]
        es_calibrated = es_raw * self.calibration_factors["c_e"]
        es_calibrated = min(es_calibrated, var_calibrated - 1e-8)

        return {
            "var_raw": var_raw,
            "es_raw": es_raw,
            "var_calibrated": var_calibrated,
            "es_calibrated": es_calibrated,
            "var_pct": var_calibrated * 100,
            "es_pct": es_calibrated * 100,
        }

    # ============================
    # Live Mode and Results Display
    # ============================
    def display_live_results(self, results, mode, processing_time):
        """
        Display live prediction results in a formatted output.

        Shows VaR/ES predictions, risk level assessment, and interpretation
        of the results for easy understanding.

        Args:
            results (dict): Prediction results from make_live_prediction
            mode (str): Operating mode (retrain/calibrate/auto)
            processing_time (float): Time taken for processing
        """
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
            f"• VaR: We expect the (α-quantile) loss tomorrow to be about {abs(results['var_pct']):.2f}%"
        )
        print(
            f"• ES: If tomorrow lands in the worst α% tail, the average loss is ~{abs(results['es_pct']):.2f}%"
        )
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
        """
        Save live prediction results to file.

        Creates a timestamped JSON file with all prediction details
        for record keeping and analysis.

        Args:
            results (dict): Prediction results from make_live_prediction
            mode (str): Operating mode
            processing_time (float): Time taken for processing

        Returns:
            str: Path to the saved prediction file
        """
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

    def should_retrain(self, retrain_days, mode):
        """
        Determine if model should be retrained based on mode and timing.

        Args:
            retrain_days (int): Days between auto-retraining
            mode (str): Operating mode (retrain/calibrate/auto)

        Returns:
            bool: True if model should be retrained
        """
        if mode == "retrain":
            return True
        if mode == "calibrate":
            return False
        if mode == "auto":
            if self.last_training_date is None:
                return True
            days_since_training = (datetime.now() - self.last_training_date).days
            return days_since_training >= retrain_days
        return False

    def run_live_mode(
        self, mode="auto", retrain_days=7, calibration_window=100, live_train_frac=1.0
    ):
        """
        Run live prediction mode with automatic retraining and calibration.

        This is the main method for production use. It handles:
        - Automatic model retraining based on schedule
        - Online calibration for improved accuracy
        - Live prediction for the next trading day
        - Results display and persistence

        Args:
            mode (str): Operating mode ('retrain', 'calibrate', or 'auto')
            retrain_days (int): Days between auto-retraining
            calibration_window (int): Observations for calibration
            live_train_frac (float): Training fraction (1.0 for full dataset)

        Returns:
            dict: Prediction results for tomorrow
        """
        print("=" * 60)
        print("HYBRID LIVE PREDICTION MODE")
        print("=" * 60)
        should_retrain_model = self.should_retrain(retrain_days, mode)
        if should_retrain_model:
            print("Mode: FULL RETRAINING (using all data)")
            start_time = time.time()
            self.load_and_prepare_data()
            self.train_frac = live_train_frac
            print(f"[LIVE] Overriding train_frac to {self.train_frac}.")

            self.train_model(force_retrain=True)

            # persist the exact model path we just wrote
            model_filename = self.get_model_filename()
            model_path = os.path.join(self.model_dir, model_filename)
            self.last_model_path = model_path
            self.save_training_info(model_path=model_path)

            training_time = time.time() - start_time
            cal_time = self.update_calibration_factors(calibration_window)
            total_time = training_time + cal_time
        else:
            last_trained_str = (
                self.last_training_date.strftime("%Y-%m-%d")
                if self.last_training_date
                else "Never"
            )
            print(f"Mode: CALIBRATION (last trained: {last_trained_str})")
            self.load_and_prepare_data()

            # In calibration mode, try to use the saved model path first
            if (
                hasattr(self, "last_model_path")
                and self.last_model_path
                and os.path.exists(self.last_model_path)
            ):
                print(f"Loading previous model from saved path: {self.last_model_path}")
                self.model = self.load_model_from_path(self.last_model_path)
            else:
                # Fallback: try current filename (data hash may have changed)
                model_filename = self.get_model_filename()
                model_path = os.path.join(self.model_dir, model_filename)

                if os.path.exists(model_path):
                    print(f"Found current model: {model_path}")
                    self.train_model(force_retrain=False)  # loads by current name
                else:
                    print(
                        "❌ No trained model found. Please run with mode='retrain' first."
                    )
                    return None

            cal_time = self.update_calibration_factors(calibration_window)
            total_time = cal_time

        results = self.make_live_prediction()
        self.display_live_results(results, mode, total_time)
        self.save_live_results(results, mode, total_time)
        print("\n" + "=" * 60)
        print("SUCCESS! Tomorrow's risk prediction is ready.")
        print("=" * 60)
        return results

    def run_standard_mode(
        self, calibrate=False, predict_next=False, save_results=False
    ):
        """
        Run standard evaluation mode for backtesting and analysis.

        Performs train/test split evaluation with optional calibration
        and next-day prediction. Useful for model development and testing.

        Args:
            calibrate (bool): Whether to apply online calibration
            predict_next (bool): Whether to make next-day prediction
            save_results (bool): Whether to save results (placeholder)

        Returns:
            dict: Evaluation results and metrics
        """
        self.load_and_prepare_data()
        self.train_model(force_retrain=False)
        results = self.evaluate_model(calibrate=calibrate)
        if predict_next:
            _ = self.predict_next_day()
        if save_results:
            pass  # extend if you want file-saving here
        print("\n" + "=" * 50)
        print("COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        return results


def list_available_models(model_dir):
    """
    List all available trained models in the specified directory.

    Args:
        model_dir (str): Directory containing model files
    """
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
    """
    Main entry point for the standalone transformer predictor.

    Handles command-line argument parsing and orchestrates the prediction workflow.
    Supports both live prediction mode and standard evaluation mode.
    """
    parser = argparse.ArgumentParser(
        description="Standalone Transformer Model for VaR/ES prediction"
    )
    parser.add_argument(
        "--csv", required=False, help="Path to CSV file with 'close' column"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="VaR/ES level (default: 0.01)"
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
        "--model_dir", default="models", help="Directory to save/load models"
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
        "--calibrate", action="store_true", help="Apply calibration during evaluation"
    )
    # Live prediction mode arguments
    parser.add_argument(
        "--live_mode", action="store_true", help="Enable live prediction mode"
    )
    parser.add_argument(
        "--mode",
        choices=["retrain", "calibrate", "auto"],
        default="auto",
        help="Live mode behavior",
    )
    parser.add_argument(
        "--retrain_days",
        type=int,
        default=7,
        help="Days between auto-retraining in live mode",
    )
    parser.add_argument(
        "--calibration_window",
        type=int,
        default=100,
        help="Days to use for live calibration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save live predictions",
    )
    parser.add_argument(
        "--live_train_frac",
        type=float,
        default=1.0,
        help="Training fraction used when retraining in --live_mode (default: 1.0)",
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models(args.model_dir)
        return

    if not args.csv:
        parser.error("--csv is required (unless using --list_models)")

    try:
        predictor = StandalonePredictor(
            csv_path=args.csv,
            alpha=args.alpha,
            feature_parity=args.feature_parity,
            train_frac=args.train_frac,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )
        if args.live_mode:
            predictor.run_live_mode(
                mode=args.mode,
                retrain_days=args.retrain_days,
                calibration_window=args.calibration_window,
                live_train_frac=args.live_train_frac,
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
