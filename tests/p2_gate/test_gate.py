# -*- coding: utf-8 -*-
"""R2's gate half. The shape and the reason vocabulary are frozen by contract C09; the
rules behind them are now the real D07/D13 ones when a bundle is present.

The vocabulary and shape tests run on the forced stub, because they are about the contract
rather than the model and must stay fast enough for every push — the real gate costs a ViT
forward pass per image. The real-gate tests skip themselves when there is no bundle, and
cover the two behaviours that are easy to get backwards: an abstaining threshold must flag
nothing, and a quality failure must not spend a forward pass.
"""
import numpy as np
import pytest

from p2_gate import api as gate_api


@pytest.fixture(autouse=True)
def _stub(monkeypatch, request):
    """Force the stub for everything except the tests marked `real_gate`."""
    if "real_gate" in request.keywords:
        return
    monkeypatch.setenv("TB_P2_GATE_FORCE_STUB", "1")
    gate_api._CACHE.pop(str(gate_api.BUNDLE_DIR), None)


needs_bundle = pytest.mark.skipif(
    not (gate_api.BUNDLE_DIR / "bank.npz").exists(),
    reason="no OOD bank at artifacts/p2/bundle/")
real_gate = pytest.mark.real_gate


def gray(value, h=512, w=512):
    return np.full((h, w), value, dtype=np.uint8)


def noisy(seed=0, h=512, w=512):
    return np.random.default_rng(seed).integers(40, 210, size=(h, w), dtype=np.uint8)


# ------------------------------------------------------------------ the frozen contract
def test_result_has_exactly_the_c09_fields():
    r = gate_api.gate(noisy())
    for f in ["quality_pass", "reasons", "ood_score", "ood_threshold", "gate_pass"]:
        assert hasattr(r, f)
    assert isinstance(r.reasons, list)
    assert r.gate_pass == (r.quality_pass and r.ood_score <= r.ood_threshold)
    assert r.is_stub is True


def test_reasons_come_only_from_the_contract_vocabulary():
    for image in [gray(0), gray(255), gray(10), noisy(), np.zeros((512, 200), np.uint8)]:
        for reason in gate_api.gate(image).reasons:
            assert reason in gate_api.QUALITY_REASONS


def test_a_reason_outside_the_contract_is_refused_not_passed_on():
    """A rule added upstream must not leak a fifth value into Person 4's parquet."""
    with pytest.raises(ValueError, match="outside the C09 contract"):
        gate_api._clean(["blank", "wrong_orientation"])


def test_blank_images_are_rejected():
    assert "blank" in gate_api.gate(gray(0)).reasons
    assert "blank" in gate_api.gate(gray(128)).reasons
    assert not gate_api.gate(gray(0)).quality_pass


def test_exposure_extremes_are_rejected():
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(12, 6, (512, 512)), 0, 255).astype(np.uint8)
    bright = np.clip(rng.normal(250, 6, (512, 512)), 0, 255).astype(np.uint8)
    assert "under_exposed" in gate_api.gate(dark).reasons
    assert "over_exposed" in gate_api.gate(bright).reasons


def test_a_heavily_cropped_image_is_rejected():
    assert "heavily_cropped" in gate_api.gate(noisy(h=512, w=200)).reasons


def test_an_ordinary_image_passes_quality():
    r = gate_api.gate(noisy(seed=7))
    assert r.quality_pass and r.reasons == []


def test_the_same_image_always_scores_the_same():
    a, b = gate_api.gate(noisy(seed=3)), gate_api.gate(noisy(seed=3))
    assert a.ood_score == b.ood_score and a.gate_pass == b.gate_pass


def test_three_channel_input_is_accepted():
    g = noisy(seed=5)
    assert gate_api.gate(np.stack([g] * 3, axis=-1)).quality_pass == gate_api.gate(g).quality_pass


# ------------------------------------------------------------------ the real gate
@real_gate
@needs_bundle
def test_the_real_gate_does_not_claim_to_be_a_stub():
    assert gate_api.is_stub() is False


@real_gate
@needs_bundle
def test_a_quality_failure_costs_no_forward_pass(monkeypatch):
    """Rejecting a blank image must not embed it: the score would be meaningless and the
    forward pass is the expensive part of the gate."""
    def explode():
        raise AssertionError("the gate embedded an image it had already rejected")
    monkeypatch.setattr(gate_api, "_embedder", explode)
    r = gate_api.gate(gray(0))
    assert r.gate_pass is False and r.quality_pass is False
    assert "blank" in r.reasons and r.is_stub is False
    assert np.isnan(r.ood_score), "a rejected image must not carry a usable OOD score"


@real_gate
@needs_bundle
def test_a_precomputed_embedding_is_used_instead_of_re_embedding(monkeypatch):
    def explode():
        raise AssertionError("the gate ignored the embedding it was given")
    monkeypatch.setattr(gate_api, "_embedder", explode)
    bank, _thr = gate_api._bundle()
    emb = bank.bank[:1].astype(np.float32)         # in distribution by definition
    r = gate_api.gate(noisy(seed=11), embedding=emb)
    assert r.quality_pass is True and r.is_stub is False
    assert r.ood_score >= 0.0


@real_gate
@needs_bundle
def test_an_abstaining_threshold_flags_nothing_rather_than_everything(monkeypatch):
    """The conformal threshold does not exist below 1/alpha - 1 calibration images. A gate
    that cannot calibrate must not reject every case — see docs/p2_ood_gate_finding.md."""
    bank, _thr = gate_api._bundle()
    monkeypatch.setitem(gate_api._CACHE, str(gate_api.BUNDLE_DIR), (bank, None))
    far = np.ones((1, bank.bank.shape[1]), np.float32)     # deliberately far from the bank
    far /= np.linalg.norm(far)
    r = gate_api.gate(noisy(seed=12), embedding=far)
    assert r.ood_abstained is True
    assert r.gate_pass is True, "an uncalibrated gate must not reject the case"
    assert r.ood_threshold is None


@real_gate
@needs_bundle
def test_a_bank_point_is_not_flagged_as_out_of_distribution():
    bank, thr = gate_api._bundle()
    if thr is None:
        pytest.skip("this bundle's threshold abstained")
    r = gate_api.gate(noisy(seed=13), embedding=bank.bank[:1].astype(np.float32))
    assert r.gate_pass is True and r.ood_score <= thr
