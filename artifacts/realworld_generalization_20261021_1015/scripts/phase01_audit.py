# -*- coding: utf-8 -*-
"""Phase 0 + 1: exposure registry, training label structure, negative-spectrum diagnostic.

Read-only. Nothing is fitted, nothing is rescored. Every label used here is documented in
a primary source:
  * Vietnam variables - data/raw/vietnam/README.md (Dryad DOI 10.5061/dryad.5tb2rbpcv)
  * TBX11K categories - artifacts/p2/labels_tbx11k_trainval_v1.csv, derived from the
    archive's own folder structure (health / sick / tb) by scripts/p1_make_tbx11k_labels.py

No metadata is inferred from filenames beyond the label derivations already committed and
verified in earlier runs.
"""
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

VN_META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
VN_SCORES = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
TBX_LABELS = ROOT / "artifacts/p2/labels_tbx11k_trainval_v1.csv"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"


def rd(p):
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


def boot_ci(y, s, n=2000, seed=17):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        v.append(roc_auc_score(y[k], s[k]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# --------------------------------------------------------------- exposure registry
REGISTRY = [
    # cohort, partition, n, training, architecture_sel, preproc_sel, calibration,
    # threshold_sel, descriptive_eval, evidence
    ("TBX11K", "train", 5941, "YES", "YES", "YES", "no", "no", "YES",
     "every head in the project is fitted here; provisional_c01.csv split=train"),
    ("TBX11K", "development", 659, "no", "YES (early stopping, variant choice)", "YES",
     "no", "YES (all thresholds)", "YES",
     "p2_33, p2_36, dadb run all early-stop and threshold here"),
    ("TBX11K", "calibration", 899, "no", "no", "no", "YES", "YES", "no",
     "bundle.json thresholds_fitted_on='calibration'"),
    ("TBX11K", "internal_test", 901, "no", "no", "no", "no", "no", "NO - never read",
     "recorded untouched in vietnam_eval.json; no artifact scores it"),
    ("Vietnam", "TUNE", 1352, "no", "YES (pooling grid + aggregator)", "no",
     "YES (scanner offsets, Platt)", "YES (scanner thresholds)", "YES",
     "mil_sweep.json selected_on='vietnam TUNE half only'; frozen_parameters.json"),
    ("Vietnam", "REPORT", 1356, "no", "no", "no", "no", "no",
     "YES - read at least 4 times across runs", "mil_final.json, task_report_sealed.py, "
     "dadb-era re-reads; original read-once status is spent"),
    ("Shenzhen", "all", 662, "no", "YES (exploratory pooling comparison)",
     "YES (negative composition, masking)", "no", "no", "YES",
     "hard_negatives.json and learning_curve.json both score it"),
    ("Montgomery", "all", 138, "no", "YES (exploratory pooling comparison)",
     "YES (negative composition, masking)", "no", "no", "YES", "same"),
    ("DA", "all", 72, "no", "no", "no", "no", "no", "YES (once, 2026-10-21)",
     "dadb_focal_pooling run; informed no configuration decision"),
    ("DB", "all", 104, "no", "no", "no", "no", "no", "YES (once, 2026-10-21)", "same"),
    ("CheXpert OOD pools", "lateral / non-medical", None, "no", "no", "no",
     "YES (kNN OOD threshold)", "YES (OOD threshold)", "YES",
     "radino_ood_*.npz; bundle.json ood_threshold_rule"),
    ("RAD-DINO checkpoint", "pre-training + checkpoint selection", None,
     "YES (backbone, frozen)", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN",
     "model card does not name the checkpoint-selection validation sets; exposure to any "
     "public TB benchmark cannot be excluded"),
]


def write_registry():
    with io.open(RUN / "tables/cohort_exposure_registry.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "partition", "n", "used_for_training",
                    "used_for_architecture_selection", "used_for_preprocessing_selection",
                    "used_for_calibration", "used_for_threshold_selection",
                    "used_for_descriptive_evaluation", "evidence"])
        w.writerows(REGISTRY)
    print("wrote tables/cohort_exposure_registry.csv (%d rows)" % len(REGISTRY))


# --------------------------------------------------------------- training label structure
def write_label_structure():
    tb = rd(TBX_LABELS)
    c01 = rd(C01)
    split = {r["image_id"]: r["split"] for r in c01}
    rows = []
    cnt = Counter((r["tbx11k_folder"], r["tbx11k_category"]) for r in tb)
    for (folder, cat), n in sorted(cnt.items()):
        ids = [r["image_id"] for r in tb
               if r["tbx11k_folder"] == folder and r["tbx11k_category"] == cat]
        bysplit = Counter(split.get(i, "not_in_c01") for i in ids)
        is_neg = cat in ("healthy", "sick_non_tb")
        rows.append([
            folder, cat, n,
            "negative" if is_neg else "positive",
            "YES" if cat == "healthy" else "no",
            "YES" if cat == "sick_non_tb" else "no",
            bysplit.get("train", 0), bysplit.get("development", 0),
            bysplit.get("calibration", 0), bysplit.get("internal_test", 0),
            "TBX11K archive folder structure (imgs/health, imgs/sick, imgs/tb); "
            "category from the archive's own annotation, not from filenames",
        ])
    with io.open(RUN / "tables/available_training_label_structure.csv", "w",
                 encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "category", "n_total", "class", "in_current_negatives",
                    "excluded_from_current_negatives", "n_train", "n_development",
                    "n_calibration", "n_internal_test", "documentation"])
        w.writerows(rows)
    print("wrote tables/available_training_label_structure.csv")
    for r in rows:
        print("  %-7s %-18s n=%-5d %-8s in_neg=%-4s train=%-5d" %
              (r[0], r[1], r[2], r[3], r[4], r[6]))
    return rows


# --------------------------------------------------------------- negative spectrum
def negative_spectrum():
    meta = {r["pid"]: r for r in rd(VN_META)}
    sc = rd(VN_SCORES)
    pid = np.array([r["pid"] for r in sc])
    y = np.array([int(r["xpert_result"]) for r in sc])
    s = np.array([float(r["score_baseline"]) for r in sc])
    half = np.array([r["split_half"] for r in sc])
    scan = np.array([int(r["rad_system"]) for r in sc])
    lunit = np.array([float(meta[p]["lunit_3901"]) for p in pid])

    past = np.array([int(meta[p]["past_tb"]) for p in pid])
    dm = np.array([1 if meta[p]["dm"] == "1" else 0 for p in pid])
    dm_unk = np.array([1 if meta[p]["dm"] == "99" else 0 for p in pid])
    sym = np.array([sum(int(meta[p][k]) for k in ("c", "f", "wl", "ns")) for p in pid])
    contact = np.array([int(meta[p]["contact"]) for p in pid])
    age = np.array([int(meta[p]["age_group"]) for p in pid])

    out = {"n": int(len(y)), "n_tb": int(y.sum()),
           "note": "descriptive only; Vietnam is an exposed cohort. Score is the "
                   "focal-pooling (12x12 max) arm, provenance verified in a prior run.",
           "variables_source": "data/raw/vietnam/README.md (Dryad)"}

    # --- the central question: do abnormal-but-not-TB negatives score high?
    groups = {
        "negative, no past TB": (y == 0) & (past == 0),
        "negative, PAST TB": (y == 0) & (past == 1),
        "negative, diabetes": (y == 0) & (dm == 1),
        "negative, 0 symptoms": (y == 0) & (sym == 0),
        "negative, 3+ symptoms": (y == 0) & (sym >= 3),
        "negative, TB contact": (y == 0) & (contact == 1),
        "negative, age >=55": (y == 0) & (age == 2),
        "ALL negatives": (y == 0),
        "ALL TB positives": (y == 1),
    }
    rows = []
    for name, m in groups.items():
        if m.sum() < 5:
            continue
        rows.append([name, int(m.sum()),
                     round(float(np.median(s[m])), 4),
                     round(float(np.percentile(s[m], 75)), 4),
                     round(float(np.percentile(s[m], 90)), 4),
                     round(float(np.mean(s[m] >= 2.4766)), 4),
                     round(float(np.median(lunit[m])), 4),
                     round(float(np.mean(lunit[m] >= np.quantile(lunit[y == 0], 0.70))), 4)])
    out["score_by_clinical_group"] = rows

    # --- the clinically meaningful discrimination task: active TB vs PRIOR TB
    tasks = {}
    for name, negmask in (("active TB vs negatives WITHOUT past TB", (y == 0) & (past == 0)),
                          ("active TB vs negatives WITH past TB", (y == 0) & (past == 1)),
                          ("active TB vs all negatives", (y == 0))):
        m = negmask | (y == 1)
        yy, ss, ll = y[m], s[m], lunit[m]
        if yy.sum() < 5 or (yy == 0).sum() < 5:
            continue
        lo, hi = boot_ci(yy, ss)
        llo, lhi = boot_ci(yy, ll)
        tasks[name] = {"n": int(m.sum()), "n_tb": int(yy.sum()),
                       "n_neg": int((yy == 0).sum()),
                       "ours_auroc": round(float(roc_auc_score(yy, ss)), 4),
                       "ours_ci95": [round(lo, 4), round(hi, 4)],
                       "lunit_auroc": round(float(roc_auc_score(yy, ll)), 4),
                       "lunit_ci95": [round(llo, 4), round(lhi, 4)]}
    out["discrimination_by_negative_spectrum"] = tasks

    # --- prevalence of each negative subtype, which bounds how much it can matter
    out["negative_composition"] = {
        "n_negatives": int((y == 0).sum()),
        "past_tb": int(((y == 0) & (past == 1)).sum()),
        "past_tb_pct": round(float(((y == 0) & (past == 1)).mean() * 100 /
                                   max((y == 0).mean(), 1e-9)), 2),
        "diabetes": int(((y == 0) & (dm == 1)).sum()),
        "diabetes_unknown": int(((y == 0) & (dm_unk == 1)).sum()),
        "three_or_more_symptoms": int(((y == 0) & (sym >= 3)).sum()),
        "zero_symptoms": int(((y == 0) & (sym == 0)).sum()),
    }
    out["past_tb_among_positives"] = int(((y == 1) & (past == 1)).sum())

    json.dump(out, io.open(RUN / "tables/vietnam_negative_spectrum.json", "w",
                           encoding="utf-8"), indent=2)
    with io.open(RUN / "tables/vietnam_negative_spectrum.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "median_score", "p75_score", "p90_score",
                    "referral_rate_at_fixed_threshold", "median_lunit",
                    "lunit_referral_rate"])
        w.writerows(rows)

    print("\nVietnam score distribution by clinical group (focal-pooling arm):")
    print("  %-28s %6s %9s %9s %9s %9s" % ("group", "n", "median", "p90", "refer", "lunit_med"))
    for r in rows:
        print("  %-28s %6d %9.3f %9.3f %9.3f %9.3f" % (r[0], r[1], r[2], r[4], r[5], r[6]))
    print("\nDiscrimination by negative spectrum:")
    for k, v in tasks.items():
        print("  %-42s n=%-5d tb=%-4d ours %.4f [%.4f, %.4f] | Lunit %.4f"
              % (k, v["n"], v["n_tb"], v["ours_auroc"], v["ours_ci95"][0],
                 v["ours_ci95"][1], v["lunit_auroc"]))
    print("\nnegative composition:", json.dumps(out["negative_composition"]))
    print("past TB among Xpert positives:", out["past_tb_among_positives"])
    return out


if __name__ == "__main__":
    write_registry()
    write_label_structure()
    negative_spectrum()
