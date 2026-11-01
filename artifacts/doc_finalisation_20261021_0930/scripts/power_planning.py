# -*- coding: utf-8 -*-
"""Sample-size planning for the next screening study, from saved predictions only.

No model is trained and no cohort is rescored. This resamples the committed Vietnam
per-case score file to estimate the sampling variability of a paired AUROC difference and
of sensitivity at fixed specificity, then inverts that into a required n.

Method and assumptions are printed with the result and repeated in the protocol document.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"

N_BOOT = 4000
SEED = 0
ALPHA = 0.05
POWER = 0.80
Z_A = 1.959964          # two-sided alpha 0.05
Z_B = 0.841621          # power 0.80


def main():
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8-sig")))
    rep = [r for r in rows if r["split_half"] == "REPORT"]
    y = np.array([int(r["xpert_result"]) for r in rep])
    base = np.array([float(r["score_baseline"]) for r in rep])
    scan = np.array([int(r["rad_system"]) for r in rep])

    # the offset-corrected arm, using the frozen TUNE-derived offsets
    fp = json.load(io.open(ROOT / "artifacts/overnight_scanner_run_20261021_0208"
                                  "/tables/frozen_parameters.json", encoding="utf-8"))
    off = fp["scanner_offsets_tune_non_tb_median"]
    corr = base.copy()
    for g in (1, 2, 3):
        corr[scan == g] -= off[str(g)]

    n0 = len(y)
    prev = float(y.mean())
    d_obs = roc_auc_score(y, corr) - roc_auc_score(y, base)

    # --- empirical SE of the paired AUROC difference at the observed n
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(N_BOOT):
        k = rng.integers(0, n0, n0)
        if len(np.unique(y[k])) < 2:
            continue
        d.append(roc_auc_score(y[k], corr[k]) - roc_auc_score(y[k], base[k]))
    se0 = float(np.std(d, ddof=1))

    # SE scales as 1/sqrt(n) for a fixed case mix, so se(n) = se0 * sqrt(n0/n).
    # Required n for a two-sided test at ALPHA with POWER:
    #   n = n0 * ((Z_A + Z_B) * se0 / delta)^2
    def n_required(delta):
        return float(n0 * ((Z_A + Z_B) * se0 / abs(delta)) ** 2)

    deltas = [0.0206, 0.03, 0.04, 0.05, 0.06, 0.08]
    auroc_plan = [{"target_delta_auroc": t, "required_n": int(np.ceil(n_required(t))),
                   "required_tb_cases": int(np.ceil(n_required(t) * prev))}
                  for t in deltas]

    # --- sensitivity at fixed specificity: SE from the same resampling
    def sens_at_spec(yy, ss, spec=0.70):
        thr = np.quantile(ss[yy == 0], spec)
        return float((ss[yy == 1] >= thr).mean())

    s_b, s_c = [], []
    for _ in range(N_BOOT):
        k = rng.integers(0, n0, n0)
        if len(np.unique(y[k])) < 2:
            continue
        s_b.append(sens_at_spec(y[k], base[k]))
        s_c.append(sens_at_spec(y[k], corr[k]))
    ds = np.array(s_c) - np.array(s_b)
    se_sens = float(ds.std(ddof=1))
    n_pos0 = int(y.sum())

    def n_pos_required(delta):
        return float(n_pos0 * ((Z_A + Z_B) * se_sens / abs(delta)) ** 2)

    sens_plan = [{"target_delta_sensitivity": t,
                  "required_tb_cases": int(np.ceil(n_pos_required(t))),
                  "implied_n_at_this_prevalence": int(np.ceil(n_pos_required(t) / prev))}
                 for t in (0.05, 0.075, 0.10, 0.15)]

    out = {
        "source_file": str(SRC.relative_to(ROOT)),
        "method": ("empirical SE from 4,000 paired bootstrap resamples of the committed "
                   "Vietnam REPORT per-case scores, extrapolated as SE proportional to "
                   "1/sqrt(n); required n = n0 * ((z_alpha + z_beta) * SE0 / delta)^2"),
        "assumptions": [
            "the new cohort has a case mix, negative-class spectrum and score "
            "distribution similar to the Vietnam screening cohort",
            "one image per patient, so cases are independent",
            "paired design: both arms scored on identical patients",
            "two-sided alpha 0.05, power 0.80",
            "prevalence %.3f, as observed in Vietnam REPORT" % prev,
            "AUROC differences of this size behave approximately normally",
            "no adjustment for multiple comparisons; add one if more than a single "
            "primary endpoint is tested",
        ],
        "observed": {"n": n0, "n_tb": n_pos0, "prevalence": round(prev, 4),
                     "paired_delta_auroc_offset_correction": round(d_obs, 6),
                     "se_paired_delta_auroc": round(se0, 6),
                     "se_paired_delta_sensitivity_at_70_spec": round(se_sens, 6)},
        "auroc_sample_size": auroc_plan,
        "sensitivity_sample_size": sens_plan,
        "caveat": ("these numbers describe the power to detect a difference between two "
                   "arms on the SAME cohort. They say nothing about the cohort size needed "
                   "to estimate absolute performance to a given precision, which is a "
                   "different and usually smaller requirement."),
    }
    json.dump(out, io.open(RUN / "tables/power_planning.json", "w", encoding="utf-8"),
              indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
