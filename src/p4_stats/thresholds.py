"""Precision-constrained upper threshold calculation with clinical bounds and clamps (Section 9.2)."""

from typing import Tuple
import numpy as np


def compute_precision_upper_threshold(
    tb_probs_calibration: np.ndarray | list,
    labels_calibration: np.ndarray | list,
    t_lower: float,
    target_precision: float = 0.80,
    grid_steps: int = 1000
) -> Tuple[float, float]:
    """
    Finds the lowest qualifying upper threshold t_upper >= t_lower achieving
    PPV / Precision >= target_precision on the calibration split.
    
    Clamps & Edge Cases:
        - If computed upper < lower, it is clamped to t_lower.
        - If target_precision is unattainable by any candidate threshold,
          t_upper is set to 1.01, leaving the TB category empty and routing
          all screen-positive cases to REFER.
          
    Args:
        tb_probs_calibration: calibrated probabilities for all calibration cases
        labels_calibration: binary ground truth labels (1 = TB, 0 = Not TB)
        t_lower: conformal lower threshold
        target_precision: desired precision for the 'TB' triage category (default 0.80)
        grid_steps: search grid granularity
        
    Returns:
        (t_upper, achieved_precision)
    """
    probs = np.asarray(tb_probs_calibration, dtype=float)
    y = np.asarray(labels_calibration, dtype=int)

    # Candidate thresholds starting from t_lower up to 1.0
    candidates = np.linspace(t_lower, 1.0, grid_steps)
    qualifying_thresholds = []

    for t in candidates:
        mask = probs >= t
        n_pred = np.sum(mask)
        if n_pred > 0:
            prec = float(np.sum(y[mask]) / n_pred)
            if prec >= target_precision:
                qualifying_thresholds.append((t, prec))

    if qualifying_thresholds:
        # Choose the lowest qualifying threshold to maximize TB triage yield
        t_upper, achieved_prec = qualifying_thresholds[0]
        # Clamp: ensure t_upper >= t_lower
        t_upper = max(float(t_upper), float(t_lower))
        return float(t_upper), float(achieved_prec)
    else:
        # Unattainable: TB category left empty, all screen-positive cases routed to REFER
        return 1.01, 0.0
