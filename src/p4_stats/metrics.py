"""Evaluation metrics and calibration diagnostic functions."""

from typing import Dict, List, Tuple
import numpy as np


def compute_roc_auc(y_true: np.ndarray | list, y_scores: np.ndarray | list) -> float:
    """Computes Area Under ROC Curve via trapezoidal rule without external dependencies."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank-sum approach (Mann-Whitney U statistic)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores)) + 1

    # Handle ties by averaging ranks
    unique_scores, inverse_indices, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tied_values = unique_scores[counts > 1]
    for val in tied_values:
        mask = (scores == val)
        ranks[mask] = np.mean(ranks[mask])

    sum_pos_ranks = np.sum(ranks[y == 1])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(auc)


def compute_sensitivity_at_specificity(
    y_true: np.ndarray | list,
    y_scores: np.ndarray | list,
    target_spec: float = 0.70
) -> Tuple[float, float]:
    """
    Computes sensitivity achieved at target specificity (e.g. 70% per WHO profile).
    
    Returns:
        (sensitivity, threshold_used)
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    neg_scores = np.sort(scores[y == 0])
    n_neg = len(neg_scores)
    if n_neg == 0:
        return 0.0, 0.5

    # Threshold where at least target_spec fraction of negatives are below threshold (score < thr)
    idx = int(np.floor(target_spec * n_neg))
    idx = min(idx, n_neg - 1)
    thr = neg_scores[idx]

    pos_scores = scores[y == 1]
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0, float(thr)

    sens = np.sum(pos_scores >= thr) / n_pos
    return float(sens), float(thr)


def compute_specificity_at_sensitivity(
    y_true: np.ndarray | list,
    y_scores: np.ndarray | list,
    target_sens: float = 0.90
) -> Tuple[float, float]:
    """
    Computes specificity achieved at target sensitivity (e.g. 90% per WHO profile).
    
    Returns:
        (specificity, threshold_used)
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    pos_scores = np.sort(scores[y == 1])
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0, 0.5

    # Threshold where at least target_sens fraction of positives are >= thr
    # (1 - target_sens) fraction are below threshold
    idx = int(np.floor((1.0 - target_sens) * n_pos))
    idx = min(idx, n_pos - 1)
    thr = pos_scores[idx]

    neg_scores = scores[y == 0]
    n_neg = len(neg_scores)
    if n_neg == 0:
        return 0.0, float(thr)

    spec = np.sum(neg_scores < thr) / n_neg
    return float(spec), float(thr)


def compute_ppv_npv(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list
) -> Tuple[float, float]:
    """Computes Positive Predictive Value (Precision) and Negative Predictive Value."""
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)

    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    tn = np.sum((pred == 0) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))

    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return ppv, npv


def compute_confusion_matrix(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    labels: Optional[List[int]] = None
) -> np.ndarray:
    """Computes confusion matrix for 2 or 3 class predictions."""
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(list(set(y) | set(pred)))

    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for i, true_lbl in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            mat[i, j] = np.sum((y == true_lbl) & (pred == pred_lbl))
    return mat


def compute_ece(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width probability bins.
    
    ECE = sum_{b=1}^B (n_b / N) * |acc(b) - conf(b)|
    """
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    N = len(y)
    if N == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (prob >= low) & (prob <= high) if i == n_bins - 1 else (prob >= low) & (prob < high)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_acc = np.mean(y[mask])
            bin_conf = np.mean(prob[mask])
            ece += (n_b / N) * np.abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    n_bins: int = 10
) -> List[Dict[str, float]]:
    """Returns binned calibration data for reliability diagrams."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (prob >= low) & (prob <= high) if i == n_bins - 1 else (prob >= low) & (prob < high)
        n_b = int(np.sum(mask))
        bin_acc = float(np.mean(y[mask])) if n_b > 0 else 0.0
        bin_conf = float(np.mean(prob[mask])) if n_b > 0 else (low + high) / 2.0

        bins_data.append({
            "bin": i + 1,
            "bin_lower": float(low),
            "bin_upper": float(high),
            "count": n_b,
            "mean_confidence": bin_conf,
            "empirical_accuracy": bin_acc,
        })

    return bins_data
