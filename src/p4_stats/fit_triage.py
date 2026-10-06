"""Fit temperature scaling on development, compute conformal thresholds on calibration, and export C12/C13."""

import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from tbcore.paths import get_artifacts_dir, get_decisions_dir, resolve_path
from tbcore.version import get_git_sha
from p4_stats.ci import wilson_score_interval
from p4_stats.conformal import compute_conformal_lower_threshold
from p4_stats.temperature import TemperatureScaler
from p4_stats.thresholds import compute_precision_upper_threshold
from p4_stats.triage import generate_operating_curve, triage_case


def fit_temperature_on_dev(dev_scores_df: pd.DataFrame) -> Tuple[float, dict]:
    """
    Fits optimal temperature T on development split to minimize NLL.
    """
    scaler = TemperatureScaler()
    y_dev = (dev_scores_df["split"] == "development")  # in practice filtered by caller
    probs_raw = dev_scores_df["p_tb_raw"].to_numpy()
    tb_labels = dev_scores_df["tb_label"].to_numpy() if "tb_label" in dev_scores_df.columns else dev_scores_df["y"].to_numpy()

    t_fitted = scaler.fit_binary(probs_raw, tb_labels, is_prob=True)

    d22_record = {
        "decision_id": "D22",
        "owner": "p4",
        "title": "Temperature Scaling Parameter",
        "criterion_preregistered": "Minimise NLL on development split after D08-D13 choices",
        "preregistered_at": "2026-10-05",
        "candidates": ["uncalibrated (T=1.0)", "grid search [0.1, 5.0]"],
        "value": {"temperature": round(float(t_fitted), 4)},
        "evidence": ["artifacts/p4/calibration/dev_temperature_fit.json"],
        "data_used": ["development"],
        "reviewed_by": "p2",
        "guide_ack": False,
        "locked": True,
        "locked_at": str(date.today()),
        "git_sha": get_git_sha(),
    }

    return float(t_fitted), d22_record


def fit_thresholds_on_calibration(
    cal_scores_df: pd.DataFrame,
    temperature: float,
    alpha: float = 0.10,
    target_precision: float = 0.80
) -> Tuple[float, float, dict]:
    """
    Fits conformal lower threshold and precision upper threshold on calibration split.
    """
    scaler = TemperatureScaler(temperature=temperature)
    raw_probs = cal_scores_df["p_tb_raw"].to_numpy()
    cal_probs = scaler.calibrate(raw_probs, is_prob=True)
    tb_labels = cal_scores_df["tb_label"].to_numpy() if "tb_label" in cal_scores_df.columns else cal_scores_df["y"].to_numpy()

    tb_scores_cal = cal_probs[tb_labels == 1]
    t_lower = compute_conformal_lower_threshold(tb_scores_cal, alpha=alpha)
    t_upper, prec = compute_precision_upper_threshold(cal_probs, tb_labels, t_lower, target_precision=target_precision)

    d23_record = {
        "decision_id": "D23",
        "owner": "p4",
        "title": "Triage Decision Thresholds",
        "criterion_preregistered": "Conformal lower threshold (alpha=0.10) and precision upper threshold (target=80%) on calibration",
        "preregistered_at": "2026-10-05",
        "candidates": ["conformal lower + precision upper"],
        "value": {
            "t_lower": round(float(t_lower), 4),
            "t_upper": round(float(t_upper), 4),
            "alpha": alpha,
            "target_precision": target_precision,
        },
        "evidence": ["artifacts/p4/triage/operating_curve_calibration_v1.parquet"],
        "data_used": ["calibration"],
        "reviewed_by": "p1",
        "guide_ack": False,
        "locked": True,
        "locked_at": str(date.today()),
        "git_sha": get_git_sha(),
    }

    return float(t_lower), float(t_upper), d23_record


def build_triage_table(
    df_scores: pd.DataFrame,
    split_name: str,
    temperature: float,
    t_lower: float,
    t_upper: float,
    df_det: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Constructs C12 triage table for a given split.
    """
    scaler = TemperatureScaler(temperature=temperature)
    df = df_scores.copy()
    raw_probs = df["p_tb_raw"].to_numpy()
    cal_probs = scaler.calibrate(raw_probs, is_prob=True)

    rows = []
    for idx, row in df.iterrows():
        img_id = str(row["image_id"])
        p_cal = float(cal_probs[idx])
        n_les = int(row.get("n_lesions", 0))
        if df_det is not None and img_id in df_det.index:
            n_les = int(df_det.loc[img_id, "n_lesions_above_thr"])

        tr = triage_case(p_cal, n_les, t_lower, t_upper)

        rows.append({
            "image_id": img_id,
            "split": split_name,
            "model_id": str(row.get("model_id", "radino_crop_m0.10_clahe0_bin")),
            "p_tb_cal": p_cal,
            "temperature": temperature,
            "t_lower": t_lower,
            "t_upper": t_upper,
            "category_pre_link": tr["category_pre_link"],
            "link_fired": tr["link_fired"],
            "category_final": tr["category_final"],
            "screen_positive": tr["screen_positive"],
            "gate_pass": bool(row.get("gate_pass", True)),
            "sanity_pass": bool(row.get("sanity_pass", True)),
            "n_lesions": n_les,
        })

    return pd.DataFrame(rows)
