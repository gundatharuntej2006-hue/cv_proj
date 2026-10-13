# -*- coding: utf-8 -*-
"""P2-02 acceptance: the five mocks Person 2 publishes must validate by Day 3.

These are P2's own checks. Person 4's `tbcore` validators are the authority; when they
land, whichever of these duplicates them is deleted rather than kept in parallel.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from p2_cls import contracts as K

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "mock"
pytestmark = pytest.mark.skipif(
    not (MOCK / "mock_manifest_p2.json").exists(),
    reason="run `py -3 scripts/p2_02_make_mocks.py` first")


@pytest.fixture(scope="module")
def manifest():
    return json.load(open(MOCK / "mock_manifest_p2.json", encoding="utf-8"))


# ---------------------------------------------------------------------------- C03
def test_c03_columns_and_vocabulary():
    df = pd.read_csv(MOCK / "data/splits/ood_heldout_v1.csv")
    assert list(df.columns) == K.C03_COLUMNS
    assert set(df.ood_type) <= set(K.C03_OOD_TYPES)
    assert set(df.role) <= set(K.C03_ROLES)
    assert df.image_id.is_unique
    assert df.licence.notna().all() and (df.licence != "").all()


def test_c03_viewhead_pools_are_disjoint_and_lateral_only():
    df = pd.read_csv(MOCK / "data/splits/ood_heldout_v1.csv")
    view = df[df.role.str.startswith("viewhead")]
    assert (view.ood_type == "lateral_cxr").all()
    counts = view.groupby("image_id").role.nunique()
    assert counts.max() == 1, "an image may sit in only one view-head pool"
    assert set(view.role) == {"viewhead_train", "viewhead_dev", "viewhead_test"}


def test_c03_mock_sources_cannot_be_mistaken_for_real_ones():
    """A mock licence must never be copyable into the committed C03 file."""
    df = pd.read_csv(MOCK / "data/splits/ood_heldout_v1.csv")
    assert (df.licence == "MOCK-NOT-A-LICENCE").all()
    assert df.source.str.startswith("mock_").all()


# ---------------------------------------------------------------------------- C07
@pytest.mark.parametrize("variant", ["crop_m0.10_clahe0", "full_clahe0"])
def test_c07_keys_shapes_and_norms(variant):
    z = np.load(MOCK / ("artifacts/p2/emb/radino_tbx11k_%s_v1.npz" % variant))
    assert sorted(z.files) == sorted(K.C07_KEYS)
    assert z["emb"].dtype == np.float16
    assert z["emb"].shape[1] == K.EMB_DIM            # 768, confirmed in P2-01
    assert z["emb"].shape[0] == z["image_id"].shape[0]
    assert bool(z["l2_normalised"]) is True
    assert str(z["variant"]) == variant
    norms = np.linalg.norm(z["emb"].astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=2e-3), "l2_normalised claims unit vectors"
    assert len(set(z["image_id"].tolist())) == z["image_id"].shape[0]


# ---------------------------------------------------------------------------- C08
@pytest.mark.parametrize("split", ["development", "calibration", "internal_test"])
def test_c08_schema_and_invariants(split):
    t = pq.read_table(MOCK / ("artifacts/p2/scores/scores_tbx11k_%s_v1.parquet" % split))
    assert t.schema.names == K.C08_SCHEMA.names
    for f in K.C08_SCHEMA:
        assert t.schema.field(f.name).type == f.type, f.name
    df = t.to_pandas()
    assert K.check_c08_invariants(df) == []
    assert set(df["head"]) <= set(K.C08_HEADS)
    assert set(df.model_id) == {m["model_id"] for m in K.REGISTRY}
    assert (df["split"] == split).all()


def test_c08_registry_covers_every_comparison_p2_13_must_ship():
    """primary, alternate head, full-image variant, CLAHE alternate, EfficientNet-B3."""
    ids = {m["model_id"] for m in K.REGISTRY}
    assert sum(m["is_primary"] for m in K.REGISTRY) == 1
    assert any(m["head"] == "three_class" for m in K.REGISTRY)
    assert any(m["margin"] is None for m in K.REGISTRY)          # the full-image row
    assert any(m["clahe"] for m in K.REGISTRY)
    assert any(i.startswith("effb3") for i in ids)


def test_c08_scores_separate_the_classes(manifest):
    """A mock nobody can learn from is no use to P4. The primary model's AUC must be in
    the region the generator claims, not 0.5."""
    df = pq.read_table(MOCK / "artifacts/p2/scores/scores_tbx11k_calibration_v1.parquet").to_pandas()
    sp = df[df.is_primary]
    y = sp.image_id.str.startswith("tb").astype(int).to_numpy()
    r = pd.Series(sp.p_tb_raw.to_numpy()).rank().to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    assert 0.80 < auc < 0.97
    assert abs(auc - manifest["known_answers"]["primary_model_auc"]["calibration"]) < 1e-3


def test_c08_temperature_matches_the_recorded_answer_key(manifest):
    """The mock is overconfident on purpose so P4-11 has something to correct. Assert
    against what the files contain, which the generator measured, not against the design
    value: each split is a finite draw and drifts from it by up to about 0.2."""
    from p2_02_make_mocks import fit_temperature
    for split, per_model in manifest["known_answers"]["recovered_temperature"].items():
        df = pq.read_table(
            MOCK / ("artifacts/p2/scores/scores_tbx11k_%s_v1.parquet" % split)).to_pandas()
        for model_id, expected in per_model.items():
            sp = df[df.model_id == model_id]
            y = sp.image_id.str.startswith("tb").astype(float).to_numpy()
            got = fit_temperature(sp.logit_tb.to_numpy().astype(float), y)
            assert abs(got - expected) < 1e-2, (split, model_id, got, expected)
            assert got > 1.2, "scores must be overconfident, or calibration has nothing to do"


# ---------------------------------------------------------------------------- C09
@pytest.mark.parametrize("name", ["gate_tbx11k_development_v1", "gate_tbx11k_calibration_v1",
                                  "gate_ood_ood_eval_v1"])
def test_c09_schema_and_vocabulary(name):
    t = pq.read_table(MOCK / ("artifacts/p2/gate/%s.parquet" % name))
    assert t.schema.names == K.C09_SCHEMA.names
    for f in K.C09_SCHEMA:
        assert t.schema.field(f.name).type == f.type, f.name
    df = t.to_pandas()
    assert (df.gate_pass == (df.quality_pass & df.ood_pass)).all()
    assert (df.ood_pass == (df.ood_score_k5 <= df.ood_threshold_k5)).all()
    for r in df.loc[~df.quality_pass, "quality_reasons"]:
        assert r in K.C09_QUALITY_REASONS
    assert (df.loc[df.quality_pass, "quality_reasons"] == "").all()
    assert df.image_id.is_unique


def test_c09_development_meets_the_d13_criterion(manifest):
    """D13: development pass rate 95.0% +- 0.2. The mock has to satisfy its own criterion,
    otherwise P4 cannot tell a broken threshold from a broken mock."""
    df = pq.read_table(MOCK / "artifacts/p2/gate/gate_tbx11k_development_v1.parquet").to_pandas()
    assert abs(100 * df.ood_pass.mean() - 95.0) <= 0.2
    assert 100 * (~df.quality_pass).mean() <= 1.0            # P2-05 criterion


def test_c09_ood_images_mostly_fail_the_gate():
    df = pq.read_table(MOCK / "artifacts/p2/gate/gate_ood_ood_eval_v1.parquet").to_pandas()
    assert df.gate_pass.mean() < 0.5
    dev = pq.read_table(MOCK / "artifacts/p2/gate/gate_tbx11k_development_v1.parquet").to_pandas()
    assert df.ood_score_k5.mean() > dev.ood_score_k5.mean() * 1.5


# ---------------------------------------------------------------------------- C17
def test_c17_columns_threshold_and_direction():
    df = pd.read_csv(MOCK / "artifacts/p2/dedup/near_dup_candidates_v1.csv")
    assert list(df.columns) == K.C17_COLUMNS
    assert (df.cosine >= df.tau).all(), "every candidate must meet tau"
    assert (df.tau == K.TAU_DEFAULT).all()
    order = {"train": 0, "development": 1, "calibration": 2, "internal_test": 3}
    assert all(order[a] <= order[b] for a, b in zip(df.split_a, df.split_b)), \
        "column a must hold the earlier split, so P1 knows which copy to keep"
    assert (df.split_a != df.split_b).all(), "same-split pairs cannot leak across splits"


def test_c17_contains_the_case_dedup_exists_for():
    """H1: a train image that reappears in internal_test, and a pair sitting exactly on
    tau so P1 must decide >= against > rather than discover it in Week 8."""
    df = pd.read_csv(MOCK / "artifacts/p2/dedup/near_dup_candidates_v1.csv")
    assert ((df.split_a == "train") & (df.split_b == "internal_test")).any()
    assert (df.cosine == K.TAU_DEFAULT).any(), "no pair sits on the boundary"


def test_c17_agrees_with_the_c07_embeddings_it_came_from():
    """The candidate list is a claim about C07. Recompute it and check, or the two mocks
    can drift apart and P1 chases a phantom."""
    z = np.load(MOCK / "artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz")
    ids = list(z["image_id"])
    idx = {v: i for i, v in enumerate(ids)}
    x = z["emb"].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    df = pd.read_csv(MOCK / "artifacts/p2/dedup/near_dup_candidates_v1.csv")
    for r in df.itertuples():
        c = float(x[idx[r.image_id_a]] @ x[idx[r.image_id_b]])
        assert abs(c - r.cosine) < 1e-3, (r.image_id_a, r.image_id_b, c, r.cosine)
