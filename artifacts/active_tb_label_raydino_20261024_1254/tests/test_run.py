# -*- coding: utf-8 -*-
"""Tests for the active-TB label-alignment run.

    py -3 -m pytest artifacts/active_tb_label_raydino_20261024_1254/tests -q
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
CALEMB = RUN / "outputs/regional_tbx11k_calibration_lungs_g12_v1.npz"

TB = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE = ("active_tb", "active_latent_tb")


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def c01():
    return rows(C01)


# ----------------------------------------------------------------- target mapping
def test_target_mapping_is_deterministic(c01):
    """Both targets are pure functions of the source category, with no silent cases."""
    seen = {}
    for r in c01:
        tb = 1 if r["category"] in TB else 0
        act = 1 if r["category"] in ACTIVE else 0
        prev = seen.setdefault(r["category"], (tb, act))
        assert prev == (tb, act), "category %s maps two ways" % r["category"]
    assert set(seen) == {"healthy", "sick_non_tb", "active_tb", "latent_tb",
                         "active_latent_tb"}


def test_stored_tb_label_matches_the_derived_one(c01):
    for r in c01:
        assert int(r["tb_label"]) == (1 if r["category"] in TB else 0), r["image_id"]


def test_the_mismatch_set_is_exactly_the_latent_images(c01):
    """140 under C01, not 139.

    139 trainval images carry *only* ObsoletePulmonaryTuberculosis boxes. C01 adds
    `tb1199`, the one TB image with no lesion box at all, which the published train
    counts (473/104/23) can only place in the latent class. The v1 label CSV disagrees
    and calls it active -- see PHASE0_STATE_AND_PROVENANCE.md section 4.
    """
    mism = [r for r in c01 if (r["category"] in TB) != (r["category"] in ACTIVE)]
    assert len(mism) == 140
    assert {r["category"] for r in mism} == {"latent_tb"}
    assert sum(r["_tbx11k_boxcats"] == "ObsoletePulmonaryTuberculosis" for r in mism) == 139
    assert [r["image_id"] for r in mism if not r["_tbx11k_boxcats"]] == ["tb1199"]


def test_every_target_class_is_accounted_for(c01):
    n = {}
    for r in c01:
        n[r["category"]] = n.get(r["category"], 0) + 1
    assert sum(n.values()) == 8400
    assert n["healthy"] == 3800 and n["sick_non_tb"] == 3800
    assert n["active_tb"] + n["latent_tb"] + n["active_latent_tb"] == 800


def test_coco_categories_declare_the_active_inactive_split():
    """The decision gate rests on the dataset's own documentation, so check it is there."""
    p = ROOT / "data/raw/tbx11k/TBX11K/annotations/json/TBX11K_train.json"
    if not p.exists():
        pytest.skip("TBX11K annotations not present on this machine")
    cats = {c["id"]: c["name"] for c in json.loads(p.read_text(encoding="utf-8"))["categories"]}
    assert cats[1] == "ActiveTuberculosis"
    assert cats[2] == "ObsoletePulmonaryTuberculosis"


# ----------------------------------------------------------------- split guards
def test_split_sizes_are_what_the_protocol_declares(c01):
    n = {}
    for r in c01:
        n[r["split"]] = n.get(r["split"], 0) + 1
    assert n == {"train": 5941, "development": 659, "calibration": 899,
                 "internal_test": 901}


def test_splits_are_disjoint(c01):
    by = {}
    for r in c01:
        by.setdefault(r["image_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by.values())
    assert len(by) == 8400


def test_internal_test_is_absent_from_every_embedding_the_run_loads(c01):
    forbidden = {r["image_id"] for r in c01 if r["split"] == "internal_test"}
    assert len(forbidden) == 901
    for p in (ROOT / "artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz", CALEMB):
        if not p.exists():
            pytest.skip("%s absent" % p.name)
        with np.load(p) as z:
            assert not (forbidden & set(z["image_id"].tolist())), p.name


def test_no_prediction_file_contains_an_internal_test_image(c01):
    forbidden = {r["image_id"] for r in c01 if r["split"] == "internal_test"}
    d = RUN / "outputs/predictions"
    if not d.exists():
        pytest.skip("predictions not written yet")
    for p in d.glob("calibration_arm*.csv"):
        ids = {r["image_id"] for r in rows(p)}
        assert len(ids) == 899
        assert not (ids & forbidden), p.name


def test_no_vietnam_report_participant_was_scored():
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json")
                    .read_text(encoding="utf-8"))
    report = {int(i) for i in sp["report"]}
    d = RUN / "outputs/predictions"
    if not d.exists():
        pytest.skip("predictions not written yet")
    for p in d.glob("vietnam_tune_arm*.csv"):
        pids = {int(r["pid"]) for r in rows(p)}
        assert len(pids) == 1352
        assert not (pids & report), p.name


# ----------------------------------------------------------------- feature cache
def test_the_calibration_embedding_reproduced_the_frozen_cache():
    p = RUN / "outputs/embed_calibration.json"
    if not p.exists():
        pytest.skip("embedding report absent")
    v = json.loads(p.read_text(encoding="utf-8"))["verification"]
    assert v["shape_match"] is True
    assert v["max_abs_diff"] <= v["tolerance"]
    assert v["mean_cosine_to_cache"] >= 0.9999


def test_calibration_embedding_shape_and_normalisation():
    if not CALEMB.exists():
        pytest.skip("calibration embedding absent")
    with np.load(CALEMB) as z:
        a, ids = z["reg"], z["image_id"].tolist()
    assert a.shape == (899, 144, 768)
    assert len(set(ids)) == 899
    n = np.linalg.norm(a[:8].astype(np.float32), axis=2)
    assert np.allclose(n, 1.0, atol=5e-3), "region vectors are not L2-normalised"


# ----------------------------------------------------------------- metric consistency
def test_protocol_hash_still_matches():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    assert h == (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]


def test_manifest_records_that_the_locked_partitions_were_not_read():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["internal_test_loaded"] is False
    assert m["vietnam_report_loaded"] is False
    assert "internal_test" not in " ".join(m["partitions_accessed"])


def test_seed_mean_score_equals_the_mean_of_the_per_seed_scores():
    d = RUN / "outputs/predictions"
    if not d.exists():
        pytest.skip("predictions not written yet")
    for p in d.glob("calibration_arm*.csv"):
        for r in rows(p):
            per = [float(v) for k, v in r.items() if k.startswith("score_seed_")
                   and k != "score_seed_mean"]
            assert abs(float(r["score_seed_mean"]) - sum(per) / len(per)) < 1e-4


def test_reported_auroc_matches_the_preserved_predictions():
    """Recompute the headline AUROC from the per-case files, independently of the run."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from p2_cls import selection as SEL
    d = RUN / "outputs/predictions"
    if not d.exists():
        pytest.skip("predictions not written yet")
    rep = {m["arm"]: float(m["cal_auroc_vs_active"]) for m in rows(TAB / "metrics_by_arm.csv")}
    for arm, want in rep.items():
        rs = rows(d / ("calibration_arm%s.csv" % arm))
        y = np.array([int(r["active_tb_label"]) for r in rs])
        p = np.array([float(r["score_seed_mean"]) for r in rs])
        assert abs(SEL.auc(y, p) - want) < 5e-4, arm


def test_the_verdict_follows_from_the_four_declared_criteria():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    p = m["primary_endpoint"]
    passed = (p["meets_min_effect_0.05"] and p["excludes_zero"]
              and p["halfwidth_within_0.05"] and p["sensitivity_guardrail_ok"])
    assert m["verdict"] == ("ACCEPT" if passed
                            else ("INCONCLUSIVE" if not p["halfwidth_within_0.05"]
                                  else "REJECT"))


def test_prevalence_is_reported_alongside_auprc():
    for m in rows(TAB / "metrics_by_arm.csv"):
        assert 0.0 < float(m["cal_prevalence_active"]) < 1.0
        assert float(m["cal_auprc_vs_active"]) > float(m["cal_prevalence_active"])


def test_threshold_was_fitted_on_development_not_on_the_evaluation_split():
    """A threshold fitted on calibration would give ~0.90 specificity there by
    construction; fitted on development it does not land exactly on 0.90."""
    for m in rows(TAB / "metrics_by_arm.csv"):
        assert abs(float(m["cal_specificity"]) - 0.90) > 1e-6
