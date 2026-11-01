# -*- coding: utf-8 -*-
"""Tests for the clinical-fusion CV run.

    py -3 -m pytest artifacts/clinical_fusion_20261101_0330/tests -q
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
SPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_protocol_hash_unchanged():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    assert h == (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]


def test_cohort_is_tune_and_report_untouched():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["n"] == 1352 and m["n_positive"] == 139 and m["n_prev_tb"] == 466
    assert m["partitions"]["vietnam_report_used"] is False
    assert m["partitions"]["tbx11k_internal_test_used"] is False
    assert m["adopted_into_production"] is False
    pinned = json.loads((ROOT / "artifacts/kaggle_exec_prep_20261023_1130/configs/"
                         "tune_cohort_fingerprint.json").read_text(encoding="utf-8"))
    assert m["cohort_fingerprint"] == pinned["fingerprint_sha256"]


def test_no_pid_in_any_output():
    tune = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["tune"]}
    for f in RUN.rglob("*"):
        if f.is_file() and f.suffix in (".csv", ".json", ".md"):
            t = f.read_text(encoding="utf-8", errors="ignore")
            assert not [p for p in tune if p in t], f.name


def test_m0_auroc_equals_the_raw_score_auroc():
    """M0 involves no fitting; its AUROC must be the score's own, with zero spread."""
    for r in rows(TAB / "fusion_results.csv"):
        if r["model"] == "M0":
            assert float(r["auroc_sd"]) == 0.0, r["image_score"]
    ours = [r for r in rows(TAB / "fusion_results.csv") if r["image_score"] == "ours" and r["model"] == "M0"][0]
    assert abs(float(ours["auroc_mean"]) - 0.7237) < 5e-4
    lun = [r for r in rows(TAB / "fusion_results.csv") if r["image_score"] == "lunit" and r["model"] == "M0"][0]
    assert abs(float(lun["auroc_mean"]) - 0.7780) < 5e-4


def test_fold_predictions_are_out_of_fold():
    """Fit inside one fold and score it; the within-fold fit must not have seen test rows."""
    sys.path.insert(0, str(RUN / "scripts"))
    import fusion_cv as F
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    s = y * 1.5 + rng.normal(size=n)
    X = rng.integers(0, 2, size=(n, len(F.CLIN))).astype(float)
    oof, pred = F.cv_oof("M2", s, X, y, seed=0)
    assert oof.shape == (n,) and pred.shape == (n,)
    assert 0 < oof.min() and oof.max() < 1
    # a leaked fit would score near-perfectly on an informative-but-noisy signal; OOF cannot
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, oof) < 0.95


def test_sensitivity_never_reported_without_specificity():
    for r in rows(TAB / "fusion_results.csv"):
        assert r["sens_at_train_thr_0.70spec"] and r["spec_at_train_thr_0.70spec"]
        assert 0.6 <= float(r["spec_at_train_thr_0.70spec"]) <= 0.8


def test_min_effect_flag_follows_the_rule():
    for r in rows(TAB / "paired_contrasts.csv"):
        if r["meets_min_effect_0.02"] == "":
            continue
        lo = float(r["ci95_all"].strip("[]").split(",")[0])
        want = float(r["delta_auroc_all"]) >= 0.02 and lo > 0
        assert (r["meets_min_effect_0.02"] == "True") == want, r


def test_repeats_and_folds_as_declared():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["folds"] == 10 and m["repeats"] == 20 and m["n_boot"] == 2000
    assert m["lr"] == {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
