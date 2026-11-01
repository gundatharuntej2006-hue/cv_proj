# -*- coding: utf-8 -*-
"""Tests for the Phase 2 task-definition experiment.

    py -3 -m pytest artifacts/realworld_generalization_20261021_1015/tests -q
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(RUN / "scripts"))

MET = json.loads((RUN / "tables/phase2_metrics.json").read_text(encoding="utf-8"))
PRED = list(csv.DictReader(io.open(RUN / "tables/phase2_predictions.csv",
                                   encoding="utf-8-sig")))
ARMS = ["H_baseline_current", "L_latent_negative", "S_sick_negatives", "X_both_changes"]


def test_preregistration_hash_matches_the_recorded_value():
    """The prereg must be byte-identical to what was hashed before scoring."""
    rec = (RUN / "tables/prereg_hash.txt").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(
        (RUN / "02_EXPERIMENT_PREREGISTRATION.md").read_bytes()).hexdigest()
    assert actual == rec, "the preregistration was edited after it was hashed"
    assert MET["decision"]["prereg_sha256"] == rec


def test_no_vietnam_label_entered_head_fitting():
    """Every arm's training rows are TBX11K categories only."""
    valid = {"active_tb", "latent_tb", "active_latent_tb", "healthy", "sick_non_tb"}
    for arm, info in MET["arms"].items():
        assert set(info["positives"]) | set(info["negatives"]) <= valid, arm
        assert set(info["positives"]) & set(info["negatives"]) == set(), \
            "%s has a category in both classes" % arm


def test_arm_definitions_are_the_four_preregistered_ones():
    assert sorted(MET["arms"]) == sorted(ARMS)
    a = MET["arms"]
    assert a["H_baseline_current"]["n_train"] == 3241
    assert a["L_latent_negative"]["n_train"] == 3241
    assert a["S_sick_negatives"]["n_train"] == 5941
    assert a["X_both_changes"]["n_train"] == 5941
    # L and X move latent_tb from the positive to the negative side
    assert "latent_tb" in a["H_baseline_current"]["positives"]
    assert "latent_tb" in a["L_latent_negative"]["negatives"]
    assert "latent_tb" in a["X_both_changes"]["negatives"]


def test_internal_test_partition_was_not_read():
    """TBX11K internal_test is the project's only unexposed partition."""
    src = (RUN / "scripts/phase2_experiment.py").read_text(encoding="utf-8")
    assert "internal_test" in src          # only in the assertion that excludes it
    assert 'c.split == "train"' in src and 'c.split == "development"' in src
    assert 'split == "internal_test"' not in src


def test_every_arm_scored_the_same_cases():
    n = len(PRED)
    assert n == 2708
    assert len({r["pid"] for r in PRED}) == n, "duplicate pid"
    for a in ARMS:
        vals = np.array([float(r["score_" + a]) for r in PRED])
        assert np.isfinite(vals).all(), a
        # scores are written to 6 decimal places, so a handful of exact ties is a CSV
        # rounding artefact rather than array reuse. Array reuse is caught separately by
        # test_no_two_arms_share_a_score_array. Anything beyond a few ties would mean a
        # degenerate head, so the bound is deliberately tight.
        assert len(np.unique(vals)) >= n - 5, (a, len(np.unique(vals)))


def test_no_two_arms_share_a_score_array():
    S = {a: np.array([float(r["score_" + a]) for r in PRED]) for a in ARMS}
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            assert not np.array_equal(S[a], S[b]), "%s and %s identical" % (a, b)


def test_primary_endpoint_subgroup_is_defined_as_preregistered():
    """active TB, plus Xpert-negative participants with past_tb == 1."""
    y = np.array([int(r["xpert_result"]) for r in PRED])
    past = np.array([int(r["past_tb"]) for r in PRED])
    m = (y == 1) | ((y == 0) & (past == 1))
    p = MET["endpoints"]["PRIMARY active_TB_vs_pastTB_negatives"]
    assert p["n"] == int(m.sum()) == 1130
    assert p["n_pos"] == int(y[m].sum()) == 280
    assert p["n_neg"] == 850


def test_metrics_recompute_from_the_saved_predictions():
    from sklearn.metrics import roc_auc_score
    y = np.array([int(r["xpert_result"]) for r in PRED])
    past = np.array([int(r["past_tb"]) for r in PRED])
    m = (y == 1) | ((y == 0) & (past == 1))
    p = MET["endpoints"]["PRIMARY active_TB_vs_pastTB_negatives"]
    for a in ARMS:
        s = np.array([float(r["score_" + a]) for r in PRED])
        assert abs(roc_auc_score(y[m], s[m]) - p["arms"][a]["auroc"]) < 1e-9, a


def test_the_decision_rule_was_applied_as_written():
    d = MET["decision"]
    p = MET["endpoints"]["PRIMARY active_TB_vs_pastTB_negatives"]
    dx = p["arms"]["X_both_changes"]["vs_H"]
    assert d["minimum_effect_size"] == 0.05
    assert d["criterion_1_effect_at_least_min"] == (dx["delta"] >= 0.05)
    assert d["criterion_2_ci_excludes_zero"] == dx["excludes_zero"]
    assert d["VERDICT"] in ("ACCEPT", "REJECT", "INCONCLUSIVE")
    # with criterion 1 and 2 both failing and the effect small, it must not read ACCEPT
    assert d["VERDICT"] != "ACCEPT"


def test_attribution_arithmetic_is_consistent():
    """X is L plus S, so their effects should approximately sum."""
    p = MET["endpoints"]["PRIMARY active_TB_vs_pastTB_negatives"]["arms"]
    l = p["L_latent_negative"]["vs_H"]["delta"]
    s = p["S_sick_negatives"]["vs_H"]["delta"]
    x = p["X_both_changes"]["vs_H"]["delta"]
    assert abs((l + s) - x) < 0.01, "arm X is not explained by L and S (%.4f vs %.4f)" % (
        l + s, x)


def test_baseline_arm_reproduces_the_published_vietnam_figure():
    g = MET["endpoints"]["GUARDRAIL active_TB_vs_all_negatives"]["arms"]
    assert abs(g["H_baseline_current"]["auroc"] - 0.7240) < 0.01


def test_past_tb_is_a_documented_variable_not_an_inferred_one():
    readme = (ROOT / "data/raw/vietnam/README.md").read_text(encoding="utf-8")
    assert "past_tb (past history of TB)" in readme
    assert "rad_system" in readme and "xpert_result" in readme
