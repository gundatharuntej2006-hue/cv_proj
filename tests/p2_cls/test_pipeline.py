# -*- coding: utf-8 -*-
"""End-to-end inference pipeline. The properties that matter clinically:

  - a case can be stopped at any stage, and a stopped case returns NO probability
    (never a number a clinician might act on);
  - a rejected image means 'send to a human', not 'not TB';
  - a batch always completes, because one bad file must not empty a worklist;
  - the detector link rule can only ever move NOT_TB up to REFER.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from p2_cls import pipeline as P

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/p2/bundle"
needs_bundle = pytest.mark.skipif(
    not (BUNDLE / "bundle.json").exists(),
    reason="run scripts/p2_build_demo_bundle.py first")


# ---------------------------------------------------------------- triage rule (no bundle)
def test_triage_three_way():
    assert P.triage(0.95, 0.3, 0.8)["category"] == "TB"
    assert P.triage(0.50, 0.3, 0.8)["category"] == "REFER"
    assert P.triage(0.10, 0.3, 0.8)["category"] == "NOT_TB"


def test_link_rule_only_lifts_not_tb():
    # fires on NOT_TB
    r = P.triage(0.10, 0.3, 0.8, n_lesions=2)
    assert r["category"] == "REFER" and r["link_fired"] and r["pre_link"] == "NOT_TB"
    # never touches TB or REFER
    assert P.triage(0.95, 0.3, 0.8, n_lesions=5)["category"] == "TB"
    assert not P.triage(0.95, 0.3, 0.8, n_lesions=5)["link_fired"]
    assert P.triage(0.50, 0.3, 0.8, n_lesions=5)["category"] == "REFER"
    assert not P.triage(0.50, 0.3, 0.8, n_lesions=5)["link_fired"]


def test_link_rule_cannot_create_a_tb_call():
    for p in (0.0, 0.1, 0.29, 0.5, 0.79):
        assert P.triage(p, 0.3, 0.8, n_lesions=99)["category"] != "TB"


def test_abstained_threshold_sends_to_human_not_to_not_tb():
    """If no conformal threshold could be fitted the gate must refer, never clear a case."""
    r = P.triage(0.01, None, None)
    assert r["category"] == "REFER" and "review" in r["reason"]


# ---------------------------------------------------------------- standardise
def test_standardise_reports_unreadable_rather_than_raising(tmp_path):
    img, err = P.standardise(tmp_path / "nope.png")
    assert img is None and "unreadable" in err


def test_standardise_resizes_to_512(tmp_path):
    from PIL import Image
    p = tmp_path / "small.png"
    Image.fromarray(np.full((128, 200), 120, np.uint8)).save(p)
    img, err = P.standardise(p)
    assert err is None and img.shape == (512, 512)


# ---------------------------------------------------------------- full pipeline
@pytest.fixture(scope="module")
def bundle():
    return P.load_bundle()


@needs_bundle
def test_missing_file_is_rejected_at_decode(bundle, tmp_path):
    r = P.run_case(tmp_path / "absent.png", bundle)
    assert r["status"] == "REJECTED" and r["stage"] == "decode"
    assert r["analysis"] is None


@needs_bundle
def test_blank_image_is_rejected_at_quality(bundle, tmp_path):
    from PIL import Image
    p = tmp_path / "blank.png"
    Image.fromarray(np.zeros((512, 512), np.uint8)).save(p)
    r = P.run_case(p, bundle)
    assert r["status"] == "REJECTED" and r["stage"] == "quality"
    assert "blank" in r["reasons"] and r["analysis"] is None


@needs_bundle
def test_a_rejected_case_never_carries_a_probability(bundle, tmp_path):
    """The clinically important invariant: no number to act on when the case was stopped."""
    from PIL import Image
    p = tmp_path / "blank.png"
    Image.fromarray(np.zeros((512, 512), np.uint8)).save(p)
    r = P.run_case(p, bundle)
    assert r["analysis"] is None
    assert "p_tb_calibrated" not in json.dumps(r)


@needs_bundle
def test_a_real_chest_xray_is_analysed(bundle):
    xr = sorted((ROOT / "data/raw/tbx11k/TBX11K/imgs/health").glob("*.png"))[:3]
    if not xr:
        pytest.skip("TBX11K not present")
    ok = [P.run_case(p, bundle) for p in xr]
    assert any(r["status"] == "ANALYSED" for r in ok), "no healthy CXR made it through"
    for r in ok:
        if r["status"] == "ANALYSED":
            a = r["analysis"]
            assert 0.0 <= a["p_tb_calibrated"] <= 1.0
            assert a["triage"] in ("TB", "REFER", "NOT_TB")
            assert isinstance(a["screen_positive"], bool)


@needs_bundle
def test_non_medical_image_is_flagged_for_review_not_silently_dropped(bundle):
    """A non-radiograph must be flagged — but flagging is now additive, not suppressive.

    This previously asserted the case was REJECTED at the ood stage. That behaviour was
    removed deliberately: the same code path deferred 21 tuberculosis patients on a
    screening cohort, 7 of whom the classifier would have correctly referred
    (docs/p2_gate_operational_audit.md).

    The cost of the change, pinned here so it stays visible: a genuinely alien image now
    receives a meaningless probability, so it must at least carry the flag.
    """
    ph = sorted((ROOT / "data/raw/ood/non_medical_commons").glob("*.jpg"))[:2]
    if not ph:
        pytest.skip("non-medical pool not present")
    for p in ph:
        r = P.run_case(p, bundle)
        if r["status"] == "REJECTED":
            # decode or quality may legitimately catch it first; both are input-integrity
            assert r["stage"] in ("decode", "quality")
            assert r["analysis"] is None
            continue
        assert r["ood"]["is_ood"] is True, "a non-radiograph must still score as OOD"
        assert r["review_required"] is True
        assert "out_of_distribution" in r["review_reasons"]


@needs_bundle
def test_legacy_blocking_policy_still_rejects_non_medical(bundle):
    """The old behaviour stays reachable for reproducing prior experiments."""
    ph = sorted((ROOT / "data/raw/ood/non_medical_commons").glob("*.jpg"))[:1]
    if not ph:
        pytest.skip("non-medical pool not present")
    r = P.run_case(ph[0], bundle, ood_policy="blocking")
    assert r["status"] == "REJECTED"
    assert r["stage"] in ("ood", "decode", "quality")


@needs_bundle
def test_batch_completes_despite_a_bad_file(bundle, tmp_path):
    good = sorted((ROOT / "data/raw/tbx11k/TBX11K/imgs/health").glob("*.png"))[:2]
    if not good:
        pytest.skip("TBX11K not present")
    paths = [good[0], tmp_path / "missing.png", good[1]]
    res = P.run_batch(paths, bundle)
    assert len(res) == 3, "a worklist must never be truncated by one bad file"
    assert sum(r["status"] == "ANALYSED" for r in res) >= 1


@needs_bundle
def test_worklist_puts_tb_first_and_rejected_last(bundle):
    fake = [
        {"image": "a", "status": "ANALYSED", "analysis": {"triage": "NOT_TB", "p_tb_calibrated": 0.01}},
        {"image": "b", "status": "REJECTED", "analysis": None},
        {"image": "c", "status": "ANALYSED", "analysis": {"triage": "TB", "p_tb_calibrated": 0.99}},
        {"image": "d", "status": "ANALYSED", "analysis": {"triage": "REFER", "p_tb_calibrated": 0.5}},
    ]
    order = [r["image"] for r in P.worklist(fake)]
    assert order == ["c", "d", "a", "b"]


@needs_bundle
def test_result_is_json_serialisable(bundle):
    """A frontend must be able to render it without importing torch."""
    xr = sorted((ROOT / "data/raw/tbx11k/TBX11K/imgs/health").glob("*.png"))[:1]
    if not xr:
        pytest.skip("TBX11K not present")
    json.dumps(P.run_case(xr[0], bundle))
