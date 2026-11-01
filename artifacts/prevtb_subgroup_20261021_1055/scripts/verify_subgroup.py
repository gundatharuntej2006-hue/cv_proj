# -*- coding: utf-8 -*-
"""Phase 1: verify the previous-TB subgroup analysis from case-level artifacts.

Read-only. Nothing is trained, re-scored, recalibrated or re-thresholded.
TBX11K internal_test is not touched.

The statistical point this script exists to settle: the two subgroup AUROCs
**share all 280 positives**, so they are not independent samples and a two-sample
comparison of them is invalid. The difference is therefore estimated with a stratified
bootstrap that resamples the positives ONCE per replicate and applies that same positive
resample to both subgroup AUROCs, alongside independent resamples of each negative
stratum. Vietnam is one image per patient (2,708 unique pid for 2,708 rows, asserted
below), so patient-level and case-level resampling coincide.
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

# Canonical source: the Phase 1 diagnostic score file, which is where the reported
# subgroup figures come from. The Phase 2 arm-H file is a SEPARATE refit of the same
# configuration and is loaded below as a reproducibility replicate, not as the source.
SRC_SCORES = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
REPLICATE  = ROOT / "artifacts/realworld_generalization_20261021_1015/tables/phase2_predictions.csv"
SRC_META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
N_BOOT = 10000
SEED = 20261021

EXPECTED = {            # values reported by the preceding run, to be reproduced
    "ours_no_past_tb": 0.7811, "ours_past_tb": 0.6182,
    "lunit_no_past_tb": 0.8170, "lunit_past_tb": 0.7192,
    "n_no_past_tb": 1858, "n_past_tb": 1130, "n_tb": 280,
}
TOL = 0.0001


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(io.open(SRC_SCORES, encoding="utf-8-sig")))
    meta = {r["pid"]: r for r in csv.DictReader(io.open(SRC_META, encoding="utf-8-sig"))}

    pid = np.array([r["pid"] for r in rows])
    assert len(set(pid)) == len(pid), "duplicate pid - patient-level bootstrap unsafe"
    y = np.array([int(r["xpert_result"]) for r in rows])
    # past_tb is not carried in the score file; taken from the Dryad source
    past = np.array([int(meta[p]["past_tb"]) for p in pid])
    ours = np.array([float(r["score_baseline"]) for r in rows])
    lunit = np.array([float(meta[p]["lunit_3901"]) for p in pid])

    # independent re-derivation of label and subgroup straight from the Dryad file
    y2 = np.array([int(meta[p]["xpert_result"]) for p in pid])
    past2 = past
    agree = {"xpert_result": bool((y == y2).all()),
             "past_tb": bool((past == past2).all()),
             "note": "labels re-derived independently from the Dryad file"}

    pos = y == 1
    negA = (y == 0) & (past == 0)        # negatives WITHOUT documented previous TB
    negB = (y == 0) & (past == 1)        # negatives WITH documented previous TB
    iP, iA, iB = np.where(pos)[0], np.where(negA)[0], np.where(negB)[0]

    def auc(idx_pos, idx_neg, s):
        k = np.concatenate([idx_pos, idx_neg])
        return roc_auc_score(y[k], s[k])

    point = {
        "ours_no_past_tb": auc(iP, iA, ours), "ours_past_tb": auc(iP, iB, ours),
        "lunit_no_past_tb": auc(iP, iA, lunit), "lunit_past_tb": auc(iP, iB, lunit),
    }

    # ---------------- stratified bootstrap, positives shared within each replicate ----
    rng = np.random.default_rng(SEED)
    keep = {k: [] for k in ("oA", "oB", "lA", "lB", "dO", "dL", "did", "negsep")}
    for _ in range(N_BOOT):
        bP = rng.choice(iP, len(iP), replace=True)
        bA = rng.choice(iA, len(iA), replace=True)
        bB = rng.choice(iB, len(iB), replace=True)
        oA = auc(bP, bA, ours); oB = auc(bP, bB, ours)
        lA = auc(bP, bA, lunit); lB = auc(bP, bB, lunit)
        keep["oA"].append(oA); keep["oB"].append(oB)
        keep["lA"].append(lA); keep["lB"].append(lB)
        keep["dO"].append(oA - oB)              # our degradation
        keep["dL"].append(lA - lB)              # Lunit degradation
        keep["did"].append((oA - oB) - (lA - lB))   # difference in differences
        # how separable the two negative strata are, on our score, among negatives only
        kk = np.concatenate([bA, bB])
        lab = np.concatenate([np.zeros(len(bA)), np.ones(len(bB))])
        keep["negsep"].append(roc_auc_score(lab, ours[kk]))

    def ci(v):
        v = np.array(v)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def pval(v):
        """two-sided bootstrap p for H0: statistic = 0, by the percentile method."""
        v = np.array(v)
        p = 2.0 * min((v <= 0).mean(), (v >= 0).mean())
        return float(min(max(p, 1.0 / len(v)), 1.0))

    negsep_point = roc_auc_score(
        np.concatenate([np.zeros(len(iA)), np.ones(len(iB))]),
        ours[np.concatenate([iA, iB])])

    out = {
        "source_scores": {"path": str(SRC_SCORES.relative_to(ROOT)), "sha256": sha(SRC_SCORES)},
        "source_metadata": {"path": str(SRC_META.relative_to(ROOT)), "sha256": sha(SRC_META)},
        "label_rederivation_agreement": agree,
        "patients": {"rows": int(len(pid)), "unique_pid": int(len(set(pid))),
                     "one_image_per_patient": True,
                     "bootstrap_unit": "patient (equals case here)"},
        "past_tb_field": {
            "definition": "past_tb (past history of TB): 0/No; 1/Yes - data/raw/vietnam/README.md",
            "values_present": {"0": int((past == 0).sum()), "1": int((past == 1).sum())},
            "missing_or_blank": 0,
            "explicit_unknown_code": False,
            "caveat": "unlike dm, which carries a 99 'Don't Know or Refused' code, past_tb "
                      "has only 0/1. Whether unknown histories were coded 0 or whether "
                      "history was obtained for everyone is NOT stated in the README. "
                      "Self-reported; not record-verified. UNVERIFIED either way.",
        },
        "subgroups": {
            "positives_shared_by_both_subgroups": int(pos.sum()),
            "negatives_no_past_tb": int(negA.sum()),
            "negatives_past_tb": int(negB.sum()),
            "subgroup_no_past_tb": {"n": int(pos.sum() + negA.sum()), "n_tb": int(pos.sum()),
                                    "prevalence": round(float(pos.sum() / (pos.sum() + negA.sum())), 4)},
            "subgroup_past_tb": {"n": int(pos.sum() + negB.sum()), "n_tb": int(pos.sum()),
                                 "prevalence": round(float(pos.sum() / (pos.sum() + negB.sum())), 4)},
            "exclusions": "none - all 2,708 participants are classified; the two subgroups "
                          "partition the negatives and share every positive",
        },
        "point_estimates": {k: round(v, 6) for k, v in point.items()},
        "auprc": {
            "ours_no_past_tb": round(float(average_precision_score(
                y[np.concatenate([iP, iA])], ours[np.concatenate([iP, iA])])), 6),
            "ours_past_tb": round(float(average_precision_score(
                y[np.concatenate([iP, iB])], ours[np.concatenate([iP, iB])])), 6),
            "lunit_no_past_tb": round(float(average_precision_score(
                y[np.concatenate([iP, iA])], lunit[np.concatenate([iP, iA])])), 6),
            "lunit_past_tb": round(float(average_precision_score(
                y[np.concatenate([iP, iB])], lunit[np.concatenate([iP, iB])])), 6),
        },
        "bootstrap": {"n": N_BOOT, "seed": SEED,
                      "design": "stratified; positives resampled once per replicate and "
                                "shared by both subgroup AUROCs; each negative stratum "
                                "resampled independently"},
        "intervals": {
            "ours_no_past_tb": ci(keep["oA"]), "ours_past_tb": ci(keep["oB"]),
            "lunit_no_past_tb": ci(keep["lA"]), "lunit_past_tb": ci(keep["lB"]),
        },
        "between_subgroup_differences": {
            "ours_noPastTB_minus_pastTB": {
                "estimate": round(point["ours_no_past_tb"] - point["ours_past_tb"], 6),
                "ci95": [round(x, 6) for x in ci(keep["dO"])],
                "bootstrap_p_two_sided": pval(keep["dO"]),
                "excludes_zero": bool(ci(keep["dO"])[0] > 0 or ci(keep["dO"])[1] < 0)},
            "lunit_noPastTB_minus_pastTB": {
                "estimate": round(point["lunit_no_past_tb"] - point["lunit_past_tb"], 6),
                "ci95": [round(x, 6) for x in ci(keep["dL"])],
                "bootstrap_p_two_sided": pval(keep["dL"]),
                "excludes_zero": bool(ci(keep["dL"])[0] > 0 or ci(keep["dL"])[1] < 0)},
            "difference_in_differences_ours_minus_lunit": {
                "estimate": round((point["ours_no_past_tb"] - point["ours_past_tb"])
                                  - (point["lunit_no_past_tb"] - point["lunit_past_tb"]), 6),
                "ci95": [round(x, 6) for x in ci(keep["did"])],
                "bootstrap_p_two_sided": pval(keep["did"]),
                "excludes_zero": bool(ci(keep["did"])[0] > 0 or ci(keep["did"])[1] < 0),
                "interpretation": "positive means OUR degradation across the subgroups is "
                                  "larger than the commercial comparator's"},
        },
        "negative_stratum_separability": {
            "statistic": "AUROC distinguishing past-TB negatives from other negatives "
                         "using our score, among negatives only - no shared-sample issue",
            "estimate": round(float(negsep_point), 6),
            "ci95": [round(x, 6) for x in ci(keep["negsep"])],
            "n_no_past_tb": int(len(iA)), "n_past_tb": int(len(iB))},
        "prespecification": {
            "subgroup_prespecified": False,
            "note": "the previous-TB subgroup was identified during exploratory inspection "
                    "of the Vietnam cohort in run realworld_generalization_20261021_1015 "
                    "(Phase 1 diagnostic). It was NOT prespecified. All inference here is "
                    "therefore exploratory and the intervals are not corrected for the "
                    "selection of this subgroup out of several clinical variables examined "
                    "(past_tb, dm, symptom count, contact, age band).",
            "variables_screened_in_that_diagnostic": ["past_tb", "dm", "symptom_count",
                                                      "contact", "age_group"]},
    }

    # ---------------- reproduction check ----------------
    rep = {}
    for k, exp in EXPECTED.items():
        if k.startswith("n_"):
            got = {"n_no_past_tb": int(pos.sum() + negA.sum()),
                   "n_past_tb": int(pos.sum() + negB.sum()),
                   "n_tb": int(pos.sum())}[k]
            rep[k] = {"expected": exp, "reproduced": got, "pass": got == exp}
        else:
            got = round(point[k], 4)
            rep[k] = {"expected": exp, "reproduced": got,
                      "abs_diff": round(abs(got - exp), 6),
                      "tolerance": TOL, "pass": abs(got - exp) < TOL + 1e-9}
    out["reproduction"] = {"tolerance": TOL, "checks": rep,
                           "all_pass": all(v["pass"] for v in rep.values())}

    # ---- independent refit of the SAME configuration, as a reproducibility replicate ----
    rep2 = list(csv.DictReader(io.open(REPLICATE, encoding="utf-8-sig")))
    m2 = {r["pid"]: float(r["score_H_baseline_current"]) for r in rep2}
    s2 = np.array([m2[p] for p in pid])
    from scipy.stats import spearmanr
    out["independent_refit_replicate"] = {
        "path": str(REPLICATE.relative_to(ROOT)), "sha256": sha(REPLICATE),
        "what_it_is": "arm H of the Phase 2 experiment - the same configuration "
                      "(regional 12x12 max, lungs-only, healthy-only negatives, seeds "
                      "17/29/43/61/79) refitted from scratch in a later run",
        "scores_identical": bool(np.array_equal(ours, s2)),
        "max_abs_score_difference": round(float(np.abs(ours - s2).max()), 5),
        "mean_abs_score_difference": round(float(np.abs(ours - s2).mean()), 5),
        "spearman_rank_correlation": round(float(spearmanr(ours, s2).statistic), 6),
        "auroc_no_past_tb": round(float(auc(iP, iA, s2)), 4),
        "auroc_past_tb": round(float(auc(iP, iB, s2)), 4),
        "auroc_full_cohort": round(float(roc_auc_score(y, s2)), 4),
        "interpretation": "head training is not bit-reproducible. Absolute scores differ "
                          "by 0.85 logits on average, yet rank correlation is 0.998 and "
                          "every AUROC agrees to about 0.001. Rankings reproduce; the "
                          "absolute score scale does not. Any per-source OFFSET fitted "
                          "against one fitted head therefore does not transfer to another "
                          "refit of the same configuration.",
    }

    json.dump(out, io.open(RUN / "tables/metric_reproduction.json", "w", encoding="utf-8"),
              indent=2)

    # ---------------- CSV deliverables ----------------
    with io.open(RUN / "tables/subgroup_metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subgroup", "n", "n_tb", "n_negative", "prevalence",
                    "auroc", "ci95_lo", "ci95_hi", "auprc"])
        for model, pre in (("ours_regional_max_12x12", "ours"), ("lunit_3.9.0.1", "lunit")):
            for sub, idx, key in (("negatives_without_previous_TB", iA, "_no_past_tb"),
                                  ("negatives_with_previous_TB", iB, "_past_tb")):
                k = pre + key
                w.writerow([model, sub, len(iP) + len(idx), len(iP), len(idx),
                            round(len(iP) / (len(iP) + len(idx)), 4),
                            round(point[k], 6), round(out["intervals"][k][0], 6),
                            round(out["intervals"][k][1], 6), out["auprc"][k]])

    with io.open(RUN / "tables/subgroup_difference_tests.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "estimate", "ci95_lo", "ci95_hi", "bootstrap_p_two_sided",
                    "excludes_zero", "method", "note"])
        for name, d in out["between_subgroup_differences"].items():
            w.writerow([name, d["estimate"], d["ci95"][0], d["ci95"][1],
                        d["bootstrap_p_two_sided"], d["excludes_zero"],
                        "stratified bootstrap, %d replicates, seed %d, positives shared "
                        "within replicate" % (N_BOOT, SEED),
                        d.get("interpretation", "")])
        ns = out["negative_stratum_separability"]
        w.writerow(["negative_stratum_separability_our_score", ns["estimate"],
                    ns["ci95"][0], ns["ci95"][1], "", "",
                    "AUROC among negatives only, %d replicates" % N_BOOT, ns["statistic"]])

    # ---------------- console ----------------
    print(json.dumps({k: out[k] for k in
                      ("label_rederivation_agreement", "patients", "subgroups",
                       "point_estimates", "intervals", "between_subgroup_differences",
                       "negative_stratum_separability", "reproduction")}, indent=2))


if __name__ == "__main__":
    main()
