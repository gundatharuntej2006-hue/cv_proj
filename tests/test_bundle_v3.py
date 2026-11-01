# -*- coding: utf-8 -*-
"""Candidate v3 bundle: weights are the audited R22 head, thresholds are development-only,
the backbone is pinned, fusion follows the logistic formula, and nothing claims promotion.

    py -3 -m pytest tests/test_bundle_v3.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_cls import pipeline_v2 as P2                                 # noqa: E402

BUNDLE = ROOT / "artifacts/p2/bundle_v3_candidate"
R22_HEAD = ROOT / "artifacts/fusion_on_stacked_head_20261101_0800/tables/X_noNLM_head.npz"


@pytest.fixture(scope="module")
def bundle():
    if not (BUNDLE / "head.npz").is_file():
        pytest.skip("bundle_v3 not built")
    return P2.load_bundle_v2(BUNDLE)


def test_weights_are_the_audited_r22_head(bundle):
    if not R22_HEAD.is_file():
        pytest.skip("R22 head absent")
    with np.load(R22_HEAD) as z:
        assert np.array_equal(z["W"].astype(np.float32), bundle["W"])
        assert np.array_equal(z["b"].astype(np.float32), bundle["b"])


def test_backbone_pinned_and_not_rad_dino(bundle):
    m = bundle["meta"]
    assert m["backbone"] == "microsoft/rad-dino-maira-2@a6132efb9268859545e9f68be18a76a6473f174a"
    assert m["bundle_version"] == 3 and m["model"]["target"].startswith("active_tb_label")


def test_not_promoted_and_exposure_stated(bundle):
    m = bundle["meta"]
    assert "NOT PROMOTED" in m["status"] and m["provisional"] is True
    assert "exposed" in m["known_external"]["note"].lower()
    assert "report" in m["known_external"]["note"].lower()


def test_thresholds_on_development_only(bundle):
    t = bundle["meta"]["thresholds"]
    assert "development" in t["fitted_on"].lower() and "never vietnam" in t["fitted_on"].lower()
    assert t["logit_at_spec_0.90"] > t["logit_at_spec_0.70"]


def test_gates_recorded(bundle):
    k = bundle["meta"]["known_external"]
    assert abs(k["vietnam_tune_auroc_raw"] - 0.7585) < 5e-4
    assert abs(k["tbx11k_calibration_auroc_vs_active"] - 0.9466) < 5e-4


def test_fusion_matches_the_logistic_formula(bundle):
    f = bundle["fusion"]
    assert "scanner" in f["status"].lower()
    enc, err = P2.encode_clinical({"sex": 0, "age_group": 1, "c": 1, "f": 0, "wl": 1, "ns": 0, "contact": 1,
                                   "past_tb": 0, "dm": 0, "rad_system": 3})
    assert err is None and enc["scan1"] == 0 and enc["scan2"] == 0
    score = -1.0
    r, err = P2.fuse(score, enc, bundle)
    assert err is None
    z = (score - f["score_mean"]) / f["score_sd"]
    lg = f["intercept"] + f["image_score_std"] * z + sum(f["clinical"][k] * enc[k] for k in f["features_order"][1:])
    assert abs(r["logit"] - lg) < 1e-3


def test_embedder_selection_by_backbone():
    """The dispatcher must route v3 to the HF embedder and v2 to RAD-DINO, without loading either."""
    v3 = {"meta": {"backbone": "microsoft/rad-dino-maira-2@a6132efb9268859545e9f68be18a76a6473f174a"}}
    v2 = {"meta": {"backbone": "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936"}}
    for bnd, want_hf in ((v3, True), (v2, False)):
        model_id, revision = bnd["meta"]["backbone"].split("@")
        is_rad_dino = model_id == "microsoft/rad-dino" and revision.startswith("110cbc18")
        assert is_rad_dino is (not want_hf)
