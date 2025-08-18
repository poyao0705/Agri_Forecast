# src/models/garch.py
# GARCH(1,1)-t model implementation.

from typing import Tuple, Any, Dict
import numpy as np
import pandas as pd

# Use the baseline module you already have
from src.baselines.baseline_classic_var_es import pipeline as baseline_pipeline


def pipeline(
    csv_path: str,
    alpha: float,
    calibrate: bool,
    feature_parity: bool,
    run_tag: str,
    out_dir: str,
    fig_dir: str,
) -> Tuple[Any, Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    GARCH(1,1)-t model implementation.
    Returns: (model_stub, metrics, (v_eval, e_eval, y_aligned, fz0_mean))
    """
    # Run your existing baseline in "garch_t" mode
    # Use expanding window starting from 50% split to match transformer
    df = pd.read_csv(csv_path)
    n_total = len(df)
    init_window = int(0.5 * n_total)  # Match transformer's 50/50 split

    model_name, metrics, (v_eval, e_eval, y_index, fz0) = baseline_pipeline(
        csv_path=csv_path,
        method="garch_t",
        alpha=alpha,
        init_window=init_window,  # Use 50% split instead of 2000
        calibrate=calibrate,
        run_tag=run_tag,
        out_dir=out_dir,
        show_progress=False,  # Disable verbose progress printing
        include_train_history=False,  # keep test-only alignment
    )

    # The baseline returns y_index (timestamps), but we need the actual return values
    # Calculate returns from close prices (same as baseline)
    px = pd.Series(df["close"].astype(float))
    returns = np.log(px / px.shift(1)).dropna().values

    # Get the actual return values corresponding to the test period
    # The GARCH baseline uses expanding window from init_window to n-1
    # So test period is from init_window+1 to n-1 (since we predict t+1 at time t)
    y_aligned = returns[init_window + 1 : init_window + 1 + len(v_eval)]

    # Calculate mean FZ0 loss
    fz0_mean = float(np.mean(fz0))

    # Calculate additional metrics for printing
    hits = (y_aligned <= v_eval).astype(int)
    hit_rate = hits.mean()

    # Import evaluation tools for statistical tests
    from src.utils.eval_tools import (
        kupiec_pof,
        christoffersen_independence,
        christoffersen_cc,
        plot_var_es_diagnostics,
    )

    # Perform statistical tests
    LR_pof, p_pof, _, _ = kupiec_pof(hits, alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    LR_cc, p_cc = christoffersen_cc(hits, alpha)

    # Print diagnostic information similar to transformer
    title = f"GARCH ({'parity' if feature_parity else 'full'}, {'calibrated' if calibrate else 'raw'})"
    print("=" * 60)
    print(title + (f"  [{run_tag}]" if run_tag else ""))
    print("=" * 60)
    print(f"Hit rate: {hit_rate:.4f} (Target {alpha:.4f})")
    print(f"Kupiec: LR={LR_pof:.4f}, p={p_pof:.4f}")
    print(f"Christoffersen IND: LR={LR_ind:.4f}, p={p_ind:.4f}")
    print(f"Christoffersen CC : LR={LR_cc:.4f}, p={p_cc:.4f}")
    print(f"Avg FZ0: {fz0_mean:.6f}")

    # Define base filename for saving (same as transformer)
    base = f"garch_{(run_tag + '_') if run_tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
        " ", ""
    )

    # Generate diagnostic plots (same as transformer)
    plot_var_es_diagnostics(
        y_true=y_aligned,
        var_pred=v_eval,
        es_pred=e_eval,
        alpha=alpha,
        title=title,
        out_dir=fig_dir,
        fname_prefix=base,
    )

    # Update the returned metrics with our custom fields
    metrics.update(
        {
            "title": title,
            "feature_parity": feature_parity,
            "features": (
                ["x_cov"]
                if feature_parity
                else ["ewma94", "ewma97", "x_cov", "neg_xcov", "neg_ret"]
            ),
            "model_desc": "GARCH(1,1) with Student-t innovations",
            "model_name": "GARCH",
            "hit_rate": hit_rate,
            "kupiec_LR": float(LR_pof),
            "kupiec_p": float(p_pof),
            "ind_LR": float(LR_ind),
            "ind_p": float(p_ind),
            "cc_LR": float(LR_cc),
            "cc_p": float(p_cc),
            "fz0_mean": fz0_mean,
            # Remove loss_series to avoid verbose output - the data is still saved in .npz file
        }
    )

    # Save predictions to .npz file for consistency with transformer
    import os

    base = f"garch_{(run_tag + '_') if run_tag else ''}{'calibrated' if calibrate else 'raw'}".replace(
        " ", ""
    )
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        y=y_aligned,
        var=v_eval,
        es=e_eval,
        fz0=fz0,
        hits=(y_aligned <= v_eval).astype(int),
        features=metrics["features"],
        feature_parity=bool(feature_parity),
        c_v=1.0 if not calibrate else metrics.get("c_v", 1.0),
        c_e=1.0 if not calibrate else metrics.get("c_e", 1.0),
    )

    # We don't need to return a model object; keep None to match your runner style
    return None, metrics, (v_eval, e_eval, y_aligned, fz0)
