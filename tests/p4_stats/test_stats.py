"""Tests for p4_stats: Wilson CI, bootstrap, metrics, temperature, conformal, thresholds, triage."""

import numpy as np
import pytest

from p4_stats.ci import bootstrap_ci, wilson_score_interval
from p4_stats.conformal import compute_conformal_lower_threshold, evaluate_conformal_empirical_coverage
from p4_stats.metrics import (
    compute_confusion_matrix,
    compute_ece,
    compute_ppv_npv,
    compute_roc_auc,
    compute_sensitivity_at_specificity,
    compute_specificity_at_sensitivity,
)
from p4_stats.temperature import TemperatureScaler
from p4_stats.thresholds import compute_precision_upper_threshold
from p4_stats.triage import generate_operating_curve, triage_case


def test_wilson_score_interval():
    # Exact properties
    low, high = wilson_score_interval(90, 100, confidence=0.95)
    assert 0.0 <= low < 0.90 < high <= 1.0
    assert np.isclose(low, 0.8238, atol=1e-2)
    assert np.isclose(high, 0.9482, atol=1e-2)

    # Edge cases
    assert wilson_score_interval(0, 100)[0] == 0.0
    assert wilson_score_interval(100, 100)[1] == 1.0
    assert wilson_score_interval(0, 0) == (0.0, 1.0)


def test_bootstrap_ci():
    rng = np.random.RandomState(42)
    data = rng.normal(10.0, 2.0, size=500)
    pt, low, high = bootstrap_ci(data, np.mean, B=500, confidence=0.95, seed=42)
    assert low < pt < high
    assert 9.5 < pt < 10.5


def test_metrics():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    # Perfect ranking -> AUC = 1.0
    assert compute_roc_auc(y_true, scores) == 1.0

    # Sensitivity at 70% specificity
    sens, thr = compute_sensitivity_at_specificity(y_true, scores, target_spec=0.70)
    assert sens == 1.0

    # PPV / NPV
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    ppv, npv = compute_ppv_npv(y_true, y_pred)
    assert ppv == 1.0 and npv == 1.0

    # ECE
    ece = compute_ece(y_true, scores, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_temperature_scaling():
    rng = np.random.RandomState(123)
    y = np.array([1] * 50 + [0] * 50)
    # Overconfident uncalibrated logits
    logits = np.concatenate([rng.normal(5.0, 1.0, 50), rng.normal(-5.0, 1.0, 50)])

    scaler = TemperatureScaler()
    t_opt = scaler.fit_binary(logits, y, is_prob=False)
    assert t_opt > 0.0

    cal_probs = scaler.calibrate(logits, is_prob=False)
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)


def test_conformal_lower_threshold():
    # Acceptance test P4-06:
    # With n = 100, alpha = 0.10: t is the largest value with <= 9 calibration TB scores below it
    scores = np.linspace(0.01, 1.0, 100)
    t_lower = compute_conformal_lower_threshold(scores, alpha=0.10)

    # Scores strictly below t_lower
    n_below = np.sum(scores < t_lower)
    assert n_below <= 9
    assert np.isclose(t_lower, scores[9], atol=1e-5)


def test_precision_upper_threshold_clamps():
    cal_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.85, 0.95])
    cal_labels = np.array([0, 0, 0, 0, 1, 1, 1])

    t_lower = 0.3
    t_upper, prec = compute_precision_upper_threshold(cal_probs, cal_labels, t_lower, target_precision=0.80)
    assert t_upper >= t_lower
    assert prec >= 0.80

    # Unattainable precision target -> clamps to 1.01
    all_zero_labels = np.zeros(len(cal_probs))
    t_upper_unatt, _ = compute_precision_upper_threshold(cal_probs, all_zero_labels, t_lower, target_precision=0.99)
    assert t_upper_unatt == 1.01


def test_triage_link_rule():
    t_lower = 0.20
    t_upper = 0.80

    # TB case (p >= t_upper)
    tr_tb = triage_case(0.85, n_lesions=0, t_lower=t_lower, t_upper=t_upper)
    assert tr_tb["category_final"] == "TB"
    assert tr_tb["link_fired"] is False
    assert tr_tb["screen_positive"] is True

    # REFER case (t_lower <= p < t_upper)
    tr_ref = triage_case(0.50, n_lesions=0, t_lower=t_lower, t_upper=t_upper)
    assert tr_ref["category_final"] == "REFER"
    assert tr_ref["link_fired"] is False
    assert tr_ref["screen_positive"] is True

    # NOT_TB without lesion
    tr_neg = triage_case(0.10, n_lesions=0, t_lower=t_lower, t_upper=t_upper)
    assert tr_neg["category_final"] == "NOT_TB"
    assert tr_neg["link_fired"] is False
    assert tr_neg["screen_positive"] is False

    # NOT_TB WITH lesion -> link rule moves to REFER
    tr_link = triage_case(0.10, n_lesions=1, t_lower=t_lower, t_upper=t_upper)
    assert tr_link["category_pre_link"] == "NOT_TB"
    assert tr_link["link_fired"] is True
    assert tr_link["category_final"] == "REFER"
    assert tr_link["screen_positive"] is True
