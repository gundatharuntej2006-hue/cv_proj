"""p4_stats - Statistical confidence intervals, calibration, conformal risk control, and triage."""

from p4_stats.ci import bootstrap_ci, wilson_score_interval
from p4_stats.conformal import compute_conformal_lower_threshold, evaluate_conformal_empirical_coverage
from p4_stats.metrics import (
    compute_confusion_matrix,
    compute_ece,
    compute_ppv_npv,
    compute_roc_auc,
    compute_sensitivity_at_specificity,
    compute_specificity_at_sensitivity,
    reliability_diagram_data,
)
from p4_stats.temperature import TemperatureScaler
from p4_stats.thresholds import compute_precision_upper_threshold
from p4_stats.triage import generate_operating_curve, triage_case
from p4_stats.fit_triage import build_triage_table, fit_temperature_on_dev, fit_thresholds_on_calibration

__all__ = [
    "wilson_score_interval",
    "bootstrap_ci",
    "compute_conformal_lower_threshold",
    "evaluate_conformal_empirical_coverage",
    "compute_roc_auc",
    "compute_sensitivity_at_specificity",
    "compute_specificity_at_sensitivity",
    "compute_ppv_npv",
    "compute_confusion_matrix",
    "compute_ece",
    "reliability_diagram_data",
    "TemperatureScaler",
    "compute_precision_upper_threshold",
    "triage_case",
    "generate_operating_curve",
    "fit_temperature_on_dev",
    "fit_thresholds_on_calibration",
    "build_triage_table",
]
