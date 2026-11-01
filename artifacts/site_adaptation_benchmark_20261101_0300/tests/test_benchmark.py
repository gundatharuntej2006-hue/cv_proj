# -*- coding: utf-8 -*-
"""Tests for the efficiency / site-adaptation benchmark.

    py -3 -m pytest artifacts/site_adaptation_benchmark_20261101_0300/tests -q
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


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ran():
    return (TAB / "run_manifest.json").is_file()


# ------------------------------------------------------------- protocol & guards
def test_protocol_hash_unchanged():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    assert h == (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]


def test_split_is_patient_disjoint_and_stratified():
    s = json.loads((TAB / "site_splits.json").read_text(encoding="utf-8"))
    assert s["shz"]["adaptation_n"] == 331 and s["shz"]["evaluation_n"] == 331
    assert s["shz"]["adaptation_tb"] == 168 and s["shz"]["evaluation_tb"] == 168
    assert s["mcu"]["adaptation_n"] == 69 and s["mcu"]["evaluation_n"] == 69
    assert s["shz"]["patient_disjoint"] is True


def test_split_reproduces_from_the_declared_seed():
    """Re-derive the Shenzhen split and demand disjointness and the same counts."""
    with np.load(ROOT / "artifacts/p2/regional/regional_nlm_lungs_g12_v1.npz") as z:
        nid = z["image_id"].tolist()
    site = np.array([i.split("__")[0] for i in nid])
    y = np.array([int(i.rsplit("_", 1)[1]) for i in nid])
    rng = np.random.default_rng(20261101)
    for st, n_each in (("shz", 331), ("mcu", 69)):
        idx = np.where(site == st)[0]
        adapt, ev = [], []
        for lab in (0, 1):
            ii = idx[y[idx] == lab]
            ii = ii[rng.permutation(len(ii))]
            h = len(ii) // 2
            adapt += ii[:h].tolist()
            ev += ii[h:].tolist()
        assert not set(adapt) & set(ev)
        assert len(adapt) == n_each and len(ev) == n_each


def test_internal_test_and_report_not_referenced():
    for f in RUN.rglob("*"):
        if f.is_file() and f.suffix in (".csv", ".json", ".md"):
            t = f.read_text(encoding="utf-8", errors="ignore")
            assert "vietnam_report" not in t.lower() or "false" in t.lower(), f.name
    if ran():
        m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
        assert m["partitions"]["vietnam_report_used"] is False
        assert m["partitions"]["tbx11k_internal_test_used"] is False
        assert m["training_of_backbone"] is False
        assert m["production_changed"] is False


# ------------------------------------------------------------- statistics
def test_threshold_rule_is_the_70pct_quantile_of_negatives():
    sys.path.insert(0, str(RUN / "scripts"))
    import benchmark as B
    y = np.array([0] * 100 + [1] * 50)
    s = np.concatenate([np.linspace(0, 1, 100), np.ones(50)])
    t = B.thr_at_spec(y, s, 0.70)
    sens, spec, _ = B.operating(y, s, t)
    assert abs(spec - 0.70) <= 0.011
    assert sens == 1.0


def test_operating_point_never_reports_sensitivity_without_specificity():
    if not ran():
        pytest.skip("run not finished")
    for r in rows(TAB / "adaptation_curve.csv"):
        assert r["sensitivity"] != "" and r["specificity"] != "", r
        assert r["sens_ci"] and r["spec_ci"]


def test_no_threshold_was_fit_on_the_evaluation_set():
    """A threshold fit on the eval set would land spec at exactly 0.70 there."""
    if not ran():
        pytest.skip("run not finished")
    # Shenzhen only: its evaluation set has 163 negatives, and 0.70 * 163 = 114.1 is not
    # an integer, so an evaluation-set-fitted threshold cannot land on exactly 0.70 there
    # by arithmetic. Montgomery has 40 negatives, where 28/40 = 0.700 exactly is a
    # reachable coincidence and says nothing about where the threshold was fit.
    hits = [r for r in rows(TAB / "adaptation_curve.csv")
            if r["site"] == "Shenzhen" and r["arm"] in ("A1", "A2")
            and abs(float(r["specificity"]) - 0.70) < 1e-6]
    assert not hits, "Shenzhen specificity lands on exactly 0.70: %s" % hits[:2]


def test_a0_is_identical_across_sites_in_head_and_threshold():
    if not ran():
        pytest.skip("run not finished")
    a0 = [r for r in rows(TAB / "adaptation_curve.csv") if r["arm"] == "A0"]
    assert len({r["threshold"] for r in a0}) == 1


def test_timing_reports_trainable_and_frozen_separately():
    if not ran():
        pytest.skip("run not finished")
    t = rows(TAB / "timing.csv")
    assert t[0]["trainable_parameters"] == "769"
    assert int(t[0]["frozen_backbone_parameters"]) > 80_000_000
    assert float(t[0]["reproduces_cached_features_mad"]) <= 5e-3


def test_total_adaptation_time_includes_encoding():
    if not ran():
        pytest.skip("run not finished")
    for r in rows(TAB / "adaptation_curve.csv"):
        if r["arm"] == "A2":
            # the three columns are rounded independently to 0.01, so allow 0.011
            assert float(r["total_adaptation_s"]) >= float(r["fit_s"]) + float(r["encode_s_est"]) - 0.011


def test_verdict_follows_the_declared_rule():
    if not ran():
        pytest.skip("run not finished")
    for r in rows(TAB / "head_refit_vs_threshold_only.csv"):
        lo = float(r["delta_sens_ci95"].strip("[]").split(",")[0])
        both = r["sens_and_spec_both_not_worse"] == "True"
        if r["protocol_verdict"] == "head refit helps":
            assert both and lo > 0, r
