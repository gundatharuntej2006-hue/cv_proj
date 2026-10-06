"""Conformal risk control for safety-critical false-negative triage thresholding (Section 9.1)."""

import numpy as np


def compute_conformal_lower_threshold(
    tb_calibration_scores: np.ndarray | list,
    alpha: float = 0.10
) -> float:
    """
    Computes the conformal lower threshold t_lower on the calibration split
    to guarantee finite-sample False Negative Risk E[FN] <= alpha (e.g. 10% miss rate).
    
    Mathematical Formulation:
        Let s_(1) <= s_(2) <= ... <= s_(n) be sorted calibration TB scores.
        Let k = floor((n + 1) * alpha).
        t_lower is set to s_(k) (1-indexed, index k-1 in 0-indexed array).
        For n=100, alpha=0.10: k = floor(101 * 0.10) = 10.
        t_lower is the largest threshold with <= 9 calibration TB scores below it.
        
    Args:
        tb_calibration_scores: 1D array of calibrated TB probabilities for true-TB cases (Y=1)
        alpha: target maximum false negative rate (default 0.10 -> >=90% sensitivity)
        
    Returns:
        t_lower in [0.0, 1.0]
    """
    scores = np.sort(np.asarray(tb_calibration_scores, dtype=float))
    n = len(scores)

    if n == 0:
        return 0.0

    k = int(np.floor((n + 1) * alpha))

    if k <= 0:
        return float(scores[0])
    elif k > n:
        return float(scores[-1])
    else:
        # 1-indexed k corresponds to index k-1 in 0-indexed sorted array
        t_lower = float(scores[k - 1])
        return t_lower


def evaluate_conformal_empirical_coverage(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float = 0.10
) -> float:
    """
    Evaluates empirical false negative rate on test set using conformal lower threshold fit on cal_scores.
    """
    t_lower = compute_conformal_lower_threshold(cal_scores, alpha=alpha)
    fn_rate = np.mean(test_scores < t_lower)
    return float(fn_rate)
