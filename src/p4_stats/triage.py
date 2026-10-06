"""Conformal 3-way triage logic, detector link rule, and calibration operating curve generation."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def triage_case(
    p_tb_cal: float,
    n_lesions: int,
    t_lower: float,
    t_upper: float
) -> Dict[str, bool | str]:
    """
    Assigns a radiograph to TB / REFER / NOT_TB using conformal thresholds and detector link rule.
    
    Link Rule (Section 9.3):
        If pre-link category is NOT_TB (p < t_lower), but the detector identifies >= 1
        high-confidence lesion (n_lesions >= 1), the decision is safely upgraded to REFER.
        The link rule NEVER modifies TB or REFER cases.
        
    Returns:
        dict with:
            - category_pre_link ("TB", "REFER", "NOT_TB")
            - link_fired (bool)
            - category_final ("TB", "REFER", "NOT_TB")
            - screen_positive (bool: category_final in ("TB", "REFER"))
    """
    p = float(p_tb_cal)
    t_low = float(t_lower)
    t_high = float(t_upper)

    if p < t_low:
        cat_pre = "NOT_TB"
        if n_lesions >= 1:
            link_fired = True
            cat_final = "REFER"
        else:
            link_fired = False
            cat_final = "NOT_TB"
    elif p < t_high:
        cat_pre = "REFER"
        link_fired = False
        cat_final = "REFER"
    else:
        cat_pre = "TB"
        link_fired = False
        cat_final = "TB"

    screen_pos = cat_final in ("TB", "REFER")

    return {
        "category_pre_link": cat_pre,
        "link_fired": link_fired,
        "category_final": cat_final,
        "screen_positive": screen_pos,
    }


def generate_operating_curve(
    tb_probs_cal: np.ndarray | list,
    labels_cal: np.ndarray | list,
    t_upper: float,
    n_points: int = 100,
    locked_t_lower: float = 0.18
) -> pd.DataFrame:
    """
    Generates the calibration operating curve (Contract C13) across candidate lower thresholds.
    """
    probs = np.asarray(tb_probs_cal, dtype=float)
    y = np.asarray(labels_cal, dtype=int)

    n_total = len(y)
    n_tb = int(np.sum(y == 1))
    n_non_tb = int(np.sum(y == 0))

    threshold_grid = np.linspace(0.01, 0.99, n_points)
    # Ensure locked_t_lower is in grid
    if locked_t_lower not in threshold_grid:
        threshold_grid = np.sort(np.append(threshold_grid, locked_t_lower))

    rows = []
    for t_low in threshold_grid:
        # Screen positive without link rule for calibration curve (standard classifier operating point)
        screen_pos = probs >= t_low
        tp = np.sum((screen_pos) & (y == 1))
        fp = np.sum((screen_pos) & (y == 0))
        tn = np.sum((~screen_pos) & (y == 0))
        fn = np.sum((~screen_pos) & (y == 1))

        sens = float(tp / n_tb) if n_tb > 0 else 0.0
        spec = float(tn / n_non_tb) if n_non_tb > 0 else 0.0

        # Category shares
        tb_count = np.sum(probs >= t_upper)
        refer_count = np.sum((probs >= t_low) & (probs < t_upper))

        tb_rate = float(tb_count / n_total) if n_total > 0 else 0.0
        refer_rate = float(refer_count / n_total) if n_total > 0 else 0.0

        is_locked = bool(np.isclose(t_low, locked_t_lower, atol=1e-4))

        rows.append({
            "t_lower": float(t_low),
            "sensitivity": sens,
            "specificity": spec,
            "refer_rate": refer_rate,
            "tb_rate": tb_rate,
            "n_tb": n_tb,
            "n_non_tb": n_non_tb,
            "is_locked_default": is_locked,
        })

    df = pd.DataFrame(rows)
    return df
