"""P4-17 Extension: External cohort recalibration experiment on Shenzhen subsets."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from tbcore.paths import get_artifacts_dir, resolve_path
from p4_stats.ci import wilson_score_interval
from p4_stats.conformal import compute_conformal_lower_threshold
from p4_stats.temperature import TemperatureScaler
from p4_eval.tables import create_table_row


def run_external_recalibration(
    df_scores: pd.DataFrame,
    subset_sizes: List[int] = [25, 50, 100],
    seed: int = 42,
    alpha: float = 0.10,
    out_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Evaluates sensitivity and specificity gains when refitting temperature and conformal lower
    threshold on small local external batches (N = 25, 50, 100).
    """
    if out_dir is None:
        out_dir = resolve_path("artifacts/p4/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    y = df_scores["tb_label"].to_numpy().astype(int)
    p_raw = df_scores["p_tb_raw"].to_numpy().astype(float)
    n_total = len(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rows = []
    for n_local in subset_sizes:
        n_pos_sample = max(1, int(n_local * (len(pos_idx) / n_total)))
        n_neg_sample = n_local - n_pos_sample

        cal_pos = rng.choice(pos_idx, size=n_pos_sample, replace=False)
        cal_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
        cal_idx = np.concatenate([cal_pos, cal_neg])

        test_mask = np.ones(n_total, dtype=bool)
        test_mask[cal_idx] = False

        y_test = y[test_mask]
        p_raw_test = p_raw[test_mask]
        n_tb_test = int(np.sum(y_test == 1))
        n_non_tb_test = int(np.sum(y_test == 0))

        # Refit temperature on local subset
        scaler = TemperatureScaler()
        t_local = scaler.fit_binary(p_raw[cal_idx], y[cal_idx], is_prob=True)

        p_cal_local = scaler.calibrate(p_raw[cal_pos], is_prob=True)
        t_low_local = compute_conformal_lower_threshold(p_cal_local, alpha=alpha)

        # Test on remaining
        p_cal_test = scaler.calibrate(p_raw_test, is_prob=True)
        screen_pos = p_cal_test >= t_low_local
        tp = int(np.sum((screen_pos) & (y_test == 1)))
        tn = int(np.sum((~screen_pos) & (y_test == 0)))

        sens = tp / n_tb_test if n_tb_test > 0 else 0.0
        sens_low, sens_high = wilson_score_interval(tp, n_tb_test)

        spec = tn / n_non_tb_test if n_non_tb_test > 0 else 0.0
        spec_low, spec_high = wilson_score_interval(tn, n_non_tb_test)

        rows.extend([
            create_table_row(
                "t13_recal", f"recal_sensitivity_n{n_local}", sens, sens_low, sens_high, "wilson",
                n_tb=n_tb_test, n_non_tb=n_non_tb_test, dataset="shenzhen", split="external_test",
                note=f"Local adaptation on N={n_local} cases (T={t_local:.2f}, t_low={t_low_local:.2f})"
            ),
            create_table_row(
                "t13_recal", f"recal_specificity_n{n_local}", spec, spec_low, spec_high, "wilson",
                n_tb=n_tb_test, n_non_tb=n_non_tb_test, dataset="shenzhen", split="external_test",
                note=f"Local adaptation on N={n_local} cases"
            ),
        ])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "t13_recal.csv", index=False)
    return df
