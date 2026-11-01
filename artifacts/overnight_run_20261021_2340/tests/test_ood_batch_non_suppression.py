# -*- coding: utf-8 -*-
"""Gap found during the overnight audit: batch-level OOD non-suppression was untested.

`tests/p2_cls/test_ood_cannot_suppress.py` covers `run_stages` thoroughly, and
`tests/p2_cls/test_pipeline.py` calls `run_batch` once, but nothing asserted that a
*worklist* preserves the guarantee. `run_batch` swallows per-case exceptions so a failure
on one image cannot stop the rest -- which is correct for availability, and is exactly the
path that could silently turn a flagged case into an ERROR row with no referral and no one
noticing, because the batch still completes.

These tests live in the overnight run directory, not in `tests/`, because the runbook for
this run forbids modifying the production test suite. Promote the file into
`tests/p2_cls/` if the team wants it enforced in CI.

Run with:
    py -3 -m pytest artifacts/overnight_run_20261021_2340/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import pipeline as P                                      # noqa: E402


class _Bank:
    def __init__(self, value):
        self.value = value

    def score(self, emb):
        return np.full(len(emb), self.value, dtype=np.float32)


class _Embedder:
    def embed(self, images):
        return np.ones((len(images), 8), dtype=np.float32)


def _bundle(ood_score, policy=None):
    meta = {"model": {"model_id": "fake", "head_type": "binary"}, "variant": "full_clahe0",
            "provisional": True, "temperature": 1.0, "t_lower": 0.30, "t_upper": 0.80,
            "ood_threshold": 0.70}
    if policy:
        meta["ood_policy"] = policy
    return {"meta": meta, "bank": _Bank(ood_score), "_embedder": _Embedder(), "entry": None}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(P, "_embedder", lambda b: b["_embedder"])
    import p2_cls.registry as R
    monkeypatch.setattr(R, "classify_embeddings",
                        lambda entry, emb: np.array([0.99], dtype=np.float32))


@pytest.fixture()
def images(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):
        p = tmp_path / ("case%d.png" % i)
        Image.fromarray(rng.integers(40, 210, (512, 512), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_every_flagged_case_in_a_batch_is_still_analysed(images):
    """A whole worklist of out-of-distribution images must still be scored, not dropped."""
    results = P.run_batch(images, _bundle(ood_score=0.99))
    assert len(results) == len(images)
    assert [r["status"] for r in results] == ["ANALYSED"] * len(images), \
        "no case may be REJECTED or ERRORed by the advisory gate"
    assert all(r["analysis"] is not None for r in results)
    assert all(r["review_required"] is True for r in results)
    assert all(r["analysis"]["screen_positive"] is True for r in results), \
        "the referral must survive the flag for every case in the batch"


def test_a_batch_decision_does_not_depend_on_the_ood_score(images):
    """Identical images, two gate scores: the triage column must be identical."""
    low = P.run_batch(images, _bundle(ood_score=0.10))
    high = P.run_batch(images, _bundle(ood_score=0.99))
    assert [r["analysis"]["triage"] for r in low] == \
           [r["analysis"]["triage"] for r in high]
    assert [r["analysis"]["p_tb_calibrated"] for r in low] == \
           [r["analysis"]["p_tb_calibrated"] for r in high]


def test_an_unread_image_is_rejected_at_decode_and_not_confused_with_ood(images):
    """The batch completing is not the same as every case having been read.

    An unreadable file and an out-of-distribution image are both reasons not to trust a
    result, but conflating them is how the original harm stayed hidden. The `stage` field
    is what keeps them apart, so it is asserted here, not only the status.
    """
    bad = list(images) + [images[0].parent / "does_not_exist.png"]
    results = P.run_batch(bad, _bundle(ood_score=0.10))
    assert len(results) == len(bad)
    unread = [r for r in results if r["status"] == "REJECTED"]
    assert len(unread) == 1
    assert unread[0]["stage"] == "decode", "an unread image is a decode failure, not an OOD one"
    assert "out_of_distribution" not in unread[0]["reasons"]
    assert unread[0]["analysis"] is None, "an unread image must carry no analysis"
    screened = [r for r in results if r.get("analysis")]
    assert len(screened) == len(images), \
        "a rejected row must not be countable as a completed screen"


def test_batch_honours_an_explicit_blocking_policy_without_changing_the_default(images):
    """Reproducing the old harmful behaviour must be opt-in, per case and per batch."""
    default = P.run_batch(images, _bundle(ood_score=0.99))
    blocking = P.run_batch(images, _bundle(ood_score=0.99), ood_policy="blocking")
    assert all(r["status"] == "ANALYSED" for r in default)
    assert all(r["status"] == "REJECTED" for r in blocking)
    assert all(r["analysis"] is None for r in blocking)


def test_the_worklist_of_a_fully_flagged_batch_is_not_empty(images):
    """The end-to-end harm: a clinician opening the worklist must still see the cases."""
    results = P.run_batch(images, _bundle(ood_score=0.99))
    ordered = P.worklist(results)
    assert len(ordered) == len(images)
    assert all(r["analysis"] is not None for r in ordered)
