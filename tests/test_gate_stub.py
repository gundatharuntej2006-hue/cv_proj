# -*- coding: utf-8 -*-
"""P2-02: the R2 gate stub. Shape and vocabulary are frozen; the rules are not.

The cases below are the ones P4's `run_case` fixtures need (P4-02): a rejected image
must be rejected for a reason drawn from the C09 vocabulary, and the same image must
always produce the same result.
"""
import numpy as np

from p2_gate import api_r2 as gate_api


def gray(value, h=512, w=512):
    return np.full((h, w), value, dtype=np.uint8)


def noisy(seed=0, h=512, w=512):
    return np.random.default_rng(seed).integers(40, 210, size=(h, w), dtype=np.uint8)


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
