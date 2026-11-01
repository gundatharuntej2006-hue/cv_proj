# -*- coding: utf-8 -*-
"""The label-free calibration guard fails closed: no offsets without a site reference, below the
batch minimum, or when the batch's questionnaire composition is off; offsets are plain
mean-centering when everything passes.

    py -3 -m pytest tests/test_site_calibration.py -q
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_cls import site_calibration as SC                            # noqa: E402

REF = {"past_tb": 0.35, "c": 0.5}


def batch(n, past_rate, seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-1.5, 2.0, n).tolist()
    clin = [{"past_tb": int(rng.random() < past_rate), "c": int(rng.random() < 0.5), "sex": 1} for _ in range(n)]
    return scores, clin


def test_refuses_without_site_reference():
    s, c = batch(100, 0.35, 0)
    r = SC.propose({1: s}, {1: c}, site_reference=None)
    assert r["applied"] is False and r["offsets"] == {}
    assert any("site reference" in x for x in r["reasons"])
    assert "EXPERIMENTAL" in r["status"] and "not prospectively validated" in r["status"]


def test_refuses_small_batch():
    s, c = batch(24, 0.35, 1)
    r = SC.propose({1: s}, {1: c}, site_reference=REF)
    assert r["applied"] is False and "24 < 25" in r["per_scanner"][1]["reasons"][0]


def test_refuses_composition_shift():
    """The R25 failure case: a previous-TB-only batch must not calibrate."""
    s, c = batch(100, 1.0, 2)
    r = SC.propose({1: s}, {1: c}, site_reference=REF)
    assert r["applied"] is False
    assert any("past_tb" in x for x in r["per_scanner"][1]["reasons"])
    assert any("frozen uncorrected model" in x for x in r["reasons"])


def test_one_bad_scanner_refuses_all():
    s1, c1 = batch(100, 0.35, 3)
    s2, c2 = batch(100, 1.0, 4)
    r = SC.propose({1: s1, 2: s2}, {1: c1, 2: c2}, site_reference=REF)
    assert r["applied"] is False and r["offsets"] == {}
    assert r["per_scanner"][1]["ok"] is True and r["per_scanner"][2]["ok"] is False


def test_offsets_are_mean_centering_when_guard_passes():
    s1, c1 = batch(100, 0.35, 5)
    s2, c2 = batch(60, 0.35, 6)
    r = SC.propose({1: s1, 2: s2}, {1: c1, 2: c2}, site_reference=REF)
    assert r["applied"] is True
    assert abs(r["offsets"][1] + np.mean(s1)) < 1e-3 and abs(r["offsets"][2] + np.mean(s2)) < 1e-3
    corrected, used = SC.apply(0.5, 1, r["offsets"])
    assert used and abs(corrected - (0.5 + r["offsets"][1])) < 1e-9
    raw, used = SC.apply(0.5, 3, r["offsets"])
    assert not used and raw == 0.5


def test_missing_clinical_rows_refuses():
    s, _ = batch(100, 0.35, 7)
    r = SC.propose({1: s}, None, site_reference=REF)
    assert r["applied"] is False and "composition not checked" in r["per_scanner"][1]["reasons"][0]
