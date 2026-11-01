# -*- coding: utf-8 -*-
"""Audit A1 tests: hash, gate, metric arithmetic, the reading rule, and that no recalibration is saved.

    py -3 -m pytest artifacts/probability_calibration_audit_20261101_1100/tests -q
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1]
TAB = RUN / "tables"
sys.path.insert(0, str(RUN / "scripts"))


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_protocol_hash_unchanged():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    assert h == (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]


def test_reproduction_gate_passed():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    g = m["reproduction_gate"]
    assert abs(g["v3_raw_tune_auroc"] - 0.7585) <= 0.0005
    assert abs(g["v2_raw_tune_auroc"] - 0.7237) <= 0.0005
    assert m["v3_head_sha256"].startswith("7f61a082")


def test_ece_and_cox_on_synthetic_calibrated_data():
    import calib_audit as A
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert A.ece(y, p) < 0.02
    c = A.cox(y, p)
    assert abs(c["slope"] - 1) < 0.05 and abs(c["intercept"]) < 0.05
    # a shifted output is detected as an intercept
    c2 = A.cox(y, A.sigmoid(A.logit(p) + 1.0))
    assert c2["intercept"] < -0.9


def test_reading_rule_applied_as_written():
    import calib_audit as A
    for r in rows(TAB / "calibration_summary.csv"):
        ok = (A.SLOPE_OK[0] <= float(r["slope"]) <= A.SLOPE_OK[1]
              and A.INTERCEPT_OK[0] <= float(r["intercept"]) <= A.INTERCEPT_OK[1]
              and float(r["ece"]) <= A.ECE_OK)
        assert r["probability_shaped"] == str(ok)


def test_reliability_bins_cover_cohort():
    rel = rows(TAB / "reliability_bins.csv")
    for out in {r["output"] for r in rel}:
        assert sum(int(r["n"]) for r in rel if r["output"] == out) == 1352


def test_nothing_saved_for_serving():
    import inspect
    import calib_audit as A
    src = inspect.getsource(A)
    assert "np.savez" not in src and "pickle" not in src and "joblib" not in src
    assert not list(TAB.glob("*.npz")) and not list(TAB.glob("*.pkl"))
    for b in ("bundle_v2_regional", "bundle_v3_candidate"):
        bj = json.loads((RUN.parents[1] / "artifacts/p2" / b / "bundle.json").read_text(encoding="utf-8"))
        assert "calibration_map" not in json.dumps(bj).lower()
