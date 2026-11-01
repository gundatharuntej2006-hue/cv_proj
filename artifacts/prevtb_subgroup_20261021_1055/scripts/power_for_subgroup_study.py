# -*- coding: utf-8 -*-
"""Phase 4: power for the chosen primary endpoint, recalculated from scratch.

The primary endpoint is the BETWEEN-SUBGROUP AUROC difference (no previous TB minus
previous TB) for one model. That is a different statistic from anything sized in earlier
runs, so no earlier sample-size number is reused.

Method: empirical SE of the statistic from the stratified bootstrap already run on the
Vietnam cohort, extrapolated as SE proportional to 1/sqrt(n) holding the design constant
(same positive fraction, same previous-TB fraction among negatives), then inverted.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

SRC = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
N_BOOT = 10000
SEED = 20261021
Z_A, Z_B = 1.959964, 0.841621          # two-sided alpha 0.05, power 0.80


def main():
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8-sig")))
    meta = {r["pid"]: r for r in csv.DictReader(io.open(META, encoding="utf-8-sig"))}
    pid = [r["pid"] for r in rows]
    y = np.array([int(meta[p]["xpert_result"]) for p in pid])
    past = np.array([int(meta[p]["past_tb"]) for p in pid])
    ours = np.array([float(r["score_baseline"]) for r in rows])
    lunit = np.array([float(meta[p]["lunit_3901"]) for p in pid])

    iP, iA, iB = np.where(y == 1)[0], np.where((y == 0) & (past == 0))[0], \
        np.where((y == 0) & (past == 1))[0]
    n0 = len(y)

    def auc(p_, n_, s):
        k = np.concatenate([p_, n_])
        return roc_auc_score(y[k], s[k])

    rng = np.random.default_rng(SEED)
    dO, dD = [], []
    for _ in range(N_BOOT):
        bP = rng.choice(iP, len(iP), replace=True)
        bA = rng.choice(iA, len(iA), replace=True)
        bB = rng.choice(iB, len(iB), replace=True)
        o = auc(bP, bA, ours) - auc(bP, bB, ours)
        l = auc(bP, bA, lunit) - auc(bP, bB, lunit)
        dO.append(o)
        dD.append(o - l)
    se_gap = float(np.std(dO, ddof=1))
    se_did = float(np.std(dD, ddof=1))
    obs_gap = auc(iP, iA, ours) - auc(iP, iB, ours)
    obs_did = obs_gap - (auc(iP, iA, lunit) - auc(iP, iB, lunit))

    def need(delta, se):
        return float(n0 * ((Z_A + Z_B) * se / abs(delta)) ** 2)

    # design constants observed in Vietnam, used to convert n into participants screened
    prev = float((y == 1).mean())
    past_frac_neg = float(len(iB) / (len(iA) + len(iB)))

    out = {
        "primary_endpoint": "between-subgroup AUROC difference for ONE model: "
                            "AUROC(active TB vs negatives without previous TB) minus "
                            "AUROC(active TB vs negatives with previous TB)",
        "secondary_endpoint": "difference in differences against a commercial comparator "
                              "scored on identical participants",
        "method": "empirical SE from a %d-replicate stratified bootstrap on the Vietnam "
                  "cohort (positives shared within replicate), extrapolated as "
                  "SE proportional to 1/sqrt(n) at fixed design, then inverted: "
                  "n = n0 * ((z_a + z_b) * SE0 / delta)^2" % N_BOOT,
        "design_constants_from_vietnam": {
            "n0": n0, "prevalence": round(prev, 4),
            "previous_TB_fraction_among_negatives": round(past_frac_neg, 4),
            "n_positives": int(len(iP)), "n_neg_no_past": int(len(iA)),
            "n_neg_past": int(len(iB))},
        "observed": {
            "gap_estimate": round(float(obs_gap), 6), "se_gap": round(se_gap, 6),
            "did_estimate": round(float(obs_did), 6), "se_did": round(se_did, 6)},
        "assumptions": [
            "a new cohort with a similar design: ~10% prevalence, ~35% of negatives with "
            "previous TB, one image per patient",
            "similar class-conditional score distributions, so a similar SE at a given n",
            "two-sided alpha 0.05, power 0.80",
            "SE scales as 1/sqrt(n); approximate normality of the difference",
            "NO multiplicity adjustment included in the headline numbers; see the "
            "sensitivity analysis for a Bonferroni-adjusted version",
            "the Vietnam SE is itself estimated and carries its own uncertainty",
        ],
        "primary_sample_size": [],
        "secondary_sample_size": [],
        "sensitivity_analyses": {},
    }

    for d in (0.05, 0.075, 0.10, 0.1629):
        n = need(d, se_gap)
        out["primary_sample_size"].append({
            "minimum_detectable_gap": d, "required_participants": int(np.ceil(n)),
            "required_TB_cases": int(np.ceil(n * prev)),
            "required_previous_TB_negatives": int(np.ceil(n * (1 - prev) * past_frac_neg))})
    for d in (0.05, 0.065, 0.075, 0.10):
        n = need(d, se_did)
        out["secondary_sample_size"].append({
            "minimum_detectable_did": d, "required_participants": int(np.ceil(n)),
            "required_TB_cases": int(np.ceil(n * prev))})

    # --- sensitivity analyses
    sens = out["sensitivity_analyses"]
    sens["previous_TB_fraction"] = [
        {"fraction_of_negatives_with_previous_TB": f,
         "required_participants_for_gap_0.05": int(np.ceil(
             need(0.05, se_gap) * (past_frac_neg / f)))}
        for f in (0.35, 0.24, 0.15, 0.10)]
    sens["note_previous_TB_fraction"] = (
        "Kagujje et al. observed 24%; Vietnam has 35%. The scaling shown is approximate: "
        "it assumes the SE is driven by the size of the smaller negative stratum.")
    sens["prevalence"] = [
        {"prevalence": p,
         "required_participants_for_gap_0.05": int(np.ceil(need(0.05, se_gap) * (prev / p)))}
        for p in (0.103, 0.05, 0.03)]
    sens["note_prevalence"] = (
        "the positives are shared by both subgroup AUROCs, so they drive the SE; halving "
        "prevalence roughly doubles the requirement.")
    sens["larger_SE_in_a_new_setting"] = [
        {"SE_multiplier": m,
         "required_participants_for_gap_0.05": int(np.ceil(need(0.05, se_gap * m)))}
        for m in (1.0, 1.25, 1.5, 2.0)]
    sens["multiplicity_bonferroni_5_subgroup_variables"] = {
        "alpha": 0.01, "z_alpha": 2.575829,
        "required_participants_for_gap_0.05": int(np.ceil(
            n0 * ((2.575829 + Z_B) * se_gap / 0.05) ** 2)),
        "rationale": "five clinical variables were screened before previous TB was "
                     "selected, so a confirmatory study should budget for that."}

    json.dump(out, io.open(RUN / "tables/power_subgroup_study.json", "w", encoding="utf-8"),
              indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
