"""External validation analysis on NLM Shenzhen and Montgomery cohorts (Section 13)."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from tbcore.paths import get_artifacts_dir, resolve_path
from p4_stats.ci import bootstrap_ci, wilson_score_interval
from p4_stats.metrics import compute_roc_auc
from p4_eval.tables import create_table_row


def evaluate_external_cohort(
    df_scores: pd.DataFrame,
    dataset_name: str,
    t_lower: float = 0.18,
    alpha_target: float = 0.10
) -> pd.DataFrame:
    """
    Evaluates classifier and conformal triage performance on an external dataset.
    """
    y = df_scores["tb_label"].to_numpy().astype(int)
    p_cal = df_scores["p_tb_cal"].to_numpy().astype(float) if "p_tb_cal" in df_scores.columns else df_scores["p_tb_raw"].to_numpy().astype(float)

    n_tb = int(np.sum(y == 1))
    n_non_tb = int(np.sum(y == 0))

    # 1. ROC-AUC
    auc, auc_low, auc_high = bootstrap_ci(
        np.arange(len(y)),
        lambda idx: compute_roc_auc(y[idx], p_cal[idx]),
        stratified=True,
        y=y
    )

    # 2. Conformal Sensitivity (p >= t_lower)
    screen_pos = p_cal >= t_lower
    tp = int(np.sum((screen_pos) & (y == 1)))
    tn = int(np.sum((~screen_pos) & (y == 0)))
    fn = int(np.sum((~screen_pos) & (y == 1)))

    sens = tp / n_tb if n_tb > 0 else 0.0
    sens_low, sens_high = wilson_score_interval(tp, n_tb)

    spec = tn / n_non_tb if n_non_tb > 0 else 0.0
    spec_low, spec_high = wilson_score_interval(tn, n_non_tb)

    # 3. Conformal FN rate check (FN / n_tb vs alpha=0.10)
    fn_rate = fn / n_tb if n_tb > 0 else 0.0
    fn_low, fn_high = wilson_score_interval(fn, n_tb)

    note = "Montgomery: 58 TB, wide CI due to small sample size" if dataset_name == "montgomery" else ""

    rows = [
        create_table_row("t13_ext", "roc_auc", auc, auc_low, auc_high, "bootstrap_percentile", n_tb, n_non_tb, dataset=dataset_name, split="external_test", note=note),
        create_table_row("t13_ext", "sensitivity_conformal_lower", sens, sens_low, sens_high, "wilson", n_tb, n_non_tb, dataset=dataset_name, split="external_test", note=note),
        create_table_row("t13_ext", "specificity_conformal_lower", spec, spec_low, spec_high, "wilson", n_tb, n_non_tb, dataset=dataset_name, split="external_test", note=note),
        create_table_row("t13_ext", "conformal_false_negative_rate", fn_rate, fn_low, fn_high, "wilson", n_tb, n_non_tb, dataset=dataset_name, split="external_test", note=f"Target alpha<={alpha_target}; {note}"),
    ]

    return pd.DataFrame(rows)


def run_all_external_evaluations(out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Runs external validation across Shenzhen and Montgomery cohorts and exports C18 table."""
    if out_dir is None:
        out_dir = resolve_path("artifacts/p4/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mock evaluation data for initial contract verification
    df_shz = pd.DataFrame({
        "image_id": [f"shz_{i:04d}" for i in range(662)],
        "tb_label": [1 if i < 336 else 0 for i in range(662)],
        "p_tb_cal": np.concatenate([np.random.beta(4, 2, 336), np.random.beta(2, 4, 326)]),
    })
    df_mcu = pd.DataFrame({
        "image_id": [f"mcu_{i:04d}" for i in range(138)],
        "tb_label": [1 if i < 58 else 0 for i in range(138)],
        "p_tb_cal": np.concatenate([np.random.beta(4, 2, 58), np.random.beta(2, 4, 80)]),
    })

    t_shz = evaluate_external_cohort(df_shz, "shenzhen")
    t_mcu = evaluate_external_cohort(df_mcu, "montgomery")

    t13 = pd.concat([t_shz, t_mcu], ignore_index=True)
    t13.to_csv(out_dir / "t13_external.csv", index=False)
    return t13
