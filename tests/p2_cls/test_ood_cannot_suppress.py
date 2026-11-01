# -*- coding: utf-8 -*-
"""Safety regression: the OOD score must never cancel a referral.

The gate used to return before the classifier ran, so an out-of-distribution flag silently
prevented a referral from ever existing. The operational audit priced that on a screening
cohort: 21 tuberculosis patients deferred, 7 of whom the classifier would have correctly
referred, and a threefold deferral disparity between scanners that carry no information
about TB status (docs/p2_gate_operational_audit.md).

These tests exist so that behaviour cannot come back. Each one states the harm it prevents.
They use a fake bundle so they run in milliseconds and do not depend on trained weights.
"""
import numpy as np
import pytest

from p2_cls import pipeline as P


# ---------------------------------------------------------------- fakes
class _Bank:
    """OOD bank whose score is dictated by the test."""

    def __init__(self, value):
        self.value = value

    def score(self, emb):
        return np.array([self.value], dtype=np.float32)


class _Embedder:
    def embed(self, images):
        return np.ones((len(images), 8), dtype=np.float32)


def _bundle(ood_score, p_tb=0.99, thr=0.70, t_lower=0.30, t_upper=0.80, policy=None):
    meta = {"model": {"model_id": "fake", "head_type": "binary"}, "variant": "full_clahe0",
            "provisional": True, "temperature": 1.0, "t_lower": t_lower, "t_upper": t_upper,
            "ood_threshold": thr}
    if policy:
        meta["ood_policy"] = policy
    return {"meta": meta, "bank": _Bank(ood_score), "_embedder": _Embedder(),
            "entry": None, "_fake_p": p_tb}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    """Stub the embedder and classifier; these tests are about control flow, not weights."""
    monkeypatch.setattr(P, "_embedder", lambda b: b["_embedder"])
    import p2_cls.registry as R
    monkeypatch.setattr(R, "classify_embeddings",
                        lambda entry, emb: np.array([0.99], dtype=np.float32))


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(40, 210, size=(512, 512), dtype=np.uint8)


# ---------------------------------------------------------------- the core guarantee
def test_a_high_ood_score_does_not_cancel_a_positive_referral():
    """THE regression this file exists for. A flagged case is still referred."""
    r = P.run_stages(_img(), _bundle(ood_score=0.99))          # far above threshold 0.70
    assert r["status"] == "ANALYSED", "an OOD flag must not stop the classifier running"
    assert r["analysis"] is not None, "a flagged case must still carry its probability"
    assert r["ood"]["is_ood"] is True, "the flag itself must still be reported"
    assert r["analysis"]["screen_positive"] is True, "the referral must survive the flag"
    assert r["review_required"] is True
    assert "out_of_distribution" in r["review_reasons"]


def test_the_flag_is_additive_not_substitutive():
    """Same image, same classifier, only the OOD score differs: the decision must match."""
    low = P.run_stages(_img(), _bundle(ood_score=0.10))
    high = P.run_stages(_img(), _bundle(ood_score=0.99))
    assert low["analysis"]["triage"] == high["analysis"]["triage"]
    assert low["analysis"]["p_tb_calibrated"] == high["analysis"]["p_tb_calibrated"]
    assert low["review_required"] is False and high["review_required"] is True


def test_a_flagged_referral_keeps_its_place_in_the_worklist():
    """Demoting a flagged case would reintroduce suppression through ordering."""
    flagged_tb = {"image": "flagged", "status": "ANALYSED", "review_required": True,
                  "analysis": {"triage": "TB", "p_tb_calibrated": 0.99,
                               "requires_additional_review": True}}
    clean_not_tb = {"image": "clean", "status": "ANALYSED", "review_required": False,
                    "analysis": {"triage": "NOT_TB", "p_tb_calibrated": 0.01,
                                 "requires_additional_review": False}}
    order = [r["image"] for r in P.worklist([clean_not_tb, flagged_tb])]
    assert order[0] == "flagged", "a flagged TB case must still sort above a clean NOT_TB"


# ---------------------------------------------------------------- not-negative guarantees
def test_a_classifier_negative_is_not_called_healthy_merely_for_passing_the_gate():
    """Passing the OOD gate is not evidence of health; the vocabulary must not imply it."""
    r = P.run_stages(_img(), _bundle(ood_score=0.10, t_lower=0.999, t_upper=0.9999))
    assert r["ood"]["is_ood"] is False
    assert r["analysis"]["triage"] in ("NOT_TB", "REFER", "TB")
    assert "healthy" not in str(r).lower(), "no code path may assert health"
    assert "negative" not in str(r.get("review_reasons", [])), \
        "a gate pass must not be recorded as a negative finding"


def test_an_unreadable_input_is_non_diagnostic_and_never_tb_negative(tmp_path):
    """Input integrity is separate from model OOD and must not yield a TB verdict."""
    r = P.run_case(tmp_path / "missing.png", _bundle(ood_score=0.1))
    assert r["status"] == "REJECTED" and r["stage"] == "decode"
    assert r["analysis"] is None, "an unreadable file must carry no probability at all"
    import json
    assert "p_tb" not in json.dumps(r), "no TB score may leak onto an unreadable input"


def test_a_quality_failure_is_non_diagnostic_and_distinct_from_ood():
    """A blank image is an input-integrity failure, not a learned-feature OOD event."""
    r = P.run_stages(np.zeros((512, 512), np.uint8), _bundle(ood_score=0.99))
    assert r["status"] == "REJECTED" and r["stage"] == "quality"
    assert r["analysis"] is None
    assert "blank" in r["reasons"]
    assert r.get("ood") is None, "quality failures short-circuit before any OOD scoring"


# ---------------------------------------------------------------- diagnostic preservation
def test_the_ood_score_is_still_recorded_for_audit():
    r = P.run_stages(_img(), _bundle(ood_score=0.83))
    assert r["ood"]["score"] == pytest.approx(0.83, abs=1e-3)
    assert r["ood"]["threshold"] == 0.70
    assert r["ood"]["is_ood"] is True


def test_an_abstaining_threshold_flags_nothing():
    """fit_threshold returns None below the conformal minimum; that must flag nothing."""
    r = P.run_stages(_img(), _bundle(ood_score=0.99, thr=None))
    assert r["ood"]["abstained"] is True and r["ood"]["is_ood"] is False
    assert r["review_required"] is False
    assert r["analysis"] is not None


# ---------------------------------------------------------------- legacy escape hatch
def test_blocking_policy_is_available_for_reproducing_prior_experiments():
    r = P.run_stages(_img(), _bundle(ood_score=0.99), ood_policy="blocking")
    assert r["status"] == "REJECTED" and r["stage"] == "ood"
    assert r["analysis"] is None


def test_blocking_is_not_the_default():
    assert P.run_stages(_img(), _bundle(ood_score=0.99))["status"] == "ANALYSED"
    assert P.run_stages(_img(), _bundle(ood_score=0.99))["ood_policy"] == "advisory"


def test_an_unknown_policy_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="advisory"):
        P.run_stages(_img(), _bundle(ood_score=0.99), ood_policy="reject_everything")
