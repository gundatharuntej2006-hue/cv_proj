# -*- coding: utf-8 -*-
"""Focused tests for the DA/DB focal-pooling evaluation.

    py -3 -m pytest artifacts/dadb_focal_pooling_20261021_0905/tests -q
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(RUN / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "data/raw/tbx11k/TBX11K/imgs/extra/da+db"
MAN = json.loads((RUN / "tables/experiment_manifest.json").read_text(encoding="utf-8"))
MET = json.loads((RUN / "tables/metrics.json").read_text(encoding="utf-8"))
PRED = list(csv.DictReader(io.open(RUN / "tables/predictions.csv", encoding="utf-8-sig")))
INV = list(csv.DictReader(io.open(RUN / "tables/case_inventory.csv", encoding="utf-8-sig")))


def truth():
    out = {}
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            out[p.stem] = (1 if p.name[0] == "p" else 0,
                           "DA" if p.suffix.lower() == ".png" else "DB")
    return out


# ------------------------------------------------------------------ cohort + alignment
def test_cohort_is_176_with_the_published_composition():
    t = truth()
    assert len(t) == 176
    da = [k for k, v in t.items() if v[1] == "DA"]
    db = [k for k, v in t.items() if v[1] == "DB"]
    assert (len(da), sum(t[k][0] for k in da)) == (72, 22)
    assert (len(db), sum(t[k][0] for k in db)) == (104, 52)


def test_every_arm_scored_the_same_cases_in_the_same_order():
    arms = sorted({r["arm"] for r in PRED})
    assert len(arms) == 3
    orders = [[r["case_id"] for r in PRED if r["arm"] == a] for a in arms]
    assert all(o == orders[0] for o in orders)
    assert len(orders[0]) == 176
    assert len(set(orders[0])) == 176


def test_labels_and_sites_match_the_canonical_source():
    t = truth()
    for r in PRED:
        assert int(r["label"]) == t[r["case_id"]][0], r["case_id"]
        assert r["site"] == t[r["case_id"]][1], r["case_id"]


def test_regional_cache_ids_match_the_source_ordering():
    import common as C
    ids = C.read_ids(RUN / "tables/regional_dadb_lungs_g12_v1.npz")
    expect = [p.stem for p in sorted(SRC.rglob("*"))
              if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    assert ids == expect


def test_global_embedding_row_count_matches_the_manifest():
    with np.load(ROOT / "artifacts/p2/external/radino_dadb_lungs_only_v1.npz") as z:
        assert z["emb"].shape == (176, 768)
        assert "image_id" not in z          # positional alignment, proven by the test above


# ------------------------------------------------------------------ shape guards
def test_regional_features_are_exactly_12x12():
    """A silent 18x18-against-12x12 mismatch cost an earlier run; pinned here."""
    import zipfile
    with zipfile.ZipFile(RUN / "tables/regional_dadb_lungs_g12_v1.npz") as z, \
            z.open("reg.npy") as f:
        v = np.lib.format.read_magic(f)
        shape, _fortran, dtype = np.lib.format._read_array_header(f, v)
    assert shape == (176, 144, 768), shape
    assert np.dtype(dtype) == np.float16
    assert MAN["configuration"]["region_grid"] == 12
    assert MAN["configuration"]["regions"] == 144


def test_the_committed_grid_constant_is_still_12():
    import p2_31_regional_mil as RM
    assert RM.GRID == 12


# ------------------------------------------------------------------ no-tuning guarantees
def test_manifest_declares_no_dadb_tuning():
    g = MAN["no_tuning_guarantee"]
    assert g["dadb_labels_used_to_fit_any_head"] is False
    assert g["dadb_labels_used_to_set_any_threshold"] is False
    assert g["dadb_used_to_select_seed_preprocessing_or_checkpoint"] is False
    assert g["offset_correction_applied"] is False
    assert g["rad_dino_retrained_or_finetuned"] is False
    assert g["pooling_sweep_rerun"] is False


def test_thresholds_come_from_tbx11k_development_not_dadb():
    thr = MAN["thresholds_from_tbx11k_development_only"]
    assert set(thr) == {"A_primary_5seed", "A_secondary_seed17", "B_regional_max_12x12"}
    # the seed-17 global threshold must match the value p2_36 recorded, to 5 decimals
    assert abs(thr["A_secondary_seed17"] - 0.697045) < 1e-5
    assert abs(thr["B_regional_max_12x12"] - (-3.336755)) < 1e-4


def test_arms_reproduce_the_published_tbx11k_development_aucs():
    d = MAN["tbx11k_dev_auroc"]
    assert abs(d["A_p2_36_mean"] - 0.9570) < 0.001
    assert abs(d["B_regional"] - 0.9368) < 0.001
    assert d["equivalence_passed"] is True


def test_streaming_was_proved_equivalent_before_bulk_use():
    e = MAN["streaming_equivalence"]
    assert e["pct_identical"] == 100.0
    assert e["max_abs_difference"] == 0.0


# ------------------------------------------------------------------ metric integrity
def test_metrics_recompute_from_the_saved_predictions():
    from sklearn.metrics import roc_auc_score
    rows = {a: [r for r in PRED if r["arm"] == a] for a in {r["arm"] for r in PRED}}
    pooled = [b for b in MET["results"]["primary_all_176"] if b["group"] == "POOLED"][0]
    for a, rs in rows.items():
        y = np.array([int(r["label"]) for r in rs])
        s = np.array([float(r["score"]) for r in rs])
        assert abs(roc_auc_score(y, s) - pooled["arms"][a]["auroc"]) < 1e-9, a


def test_no_duplicate_ids_and_no_non_finite_scores():
    i = MET["integrity"]
    assert i["duplicates"] == 0
    assert all(v == 0 for v in i["non_finite_scores"].values())
    assert i["labels_rederived_from_filename_match"] is True
    assert i["published_counts_match"] is True


def test_no_two_arms_share_a_score_array():
    assert not any(MET["integrity"]["arm_arrays_identical"].values())


def test_paired_bootstrap_file_reproduces_the_reported_interval():
    rows = list(csv.DictReader(io.open(RUN / "tables/paired_bootstrap.csv",
                                       encoding="utf-8-sig")))
    d = np.array([float(r["delta_auroc_B_minus_A_primary_pooled_all176"]) for r in rows])
    pooled = [b for b in MET["results"]["primary_all_176"] if b["group"] == "POOLED"][0]
    p = pooled["paired_B_minus_A_primary"]
    assert len(d) == p["n_resamples_used"]
    assert abs(np.percentile(d, 2.5) - p["ci95_lo"]) < 1e-9
    assert abs(np.percentile(d, 97.5) - p["ci95_hi"]) < 1e-9


# ------------------------------------------------------------------ the sanity-fail case
def test_the_sanity_failure_is_identified_label_blind_and_is_the_only_one():
    fails = [r for r in INV if r["sanity_verdict"] == "FAIL"]
    assert len(fails) == 1
    f = fails[0]
    assert f["case_id"] == "px4"
    assert f["sanity_reasons"] == "components"
    assert int(f["n_lung_components"]) == 1
    # the rule is label-blind by construction: it takes the mask and nothing else
    import inspect
    import p1_build_c04_c05 as C05
    assert list(inspect.signature(C05.roi_from_mask).parameters) == ["m"]
    assert set(C05.roi_from_mask(np.zeros((512, 512), np.uint8))) >= {"sanity_pass"}


def test_all_176_are_included_in_the_primary_and_175_in_the_sensitivity():
    assert sum(int(r["included_primary"]) for r in INV) == 176
    assert sum(int(r["included_sensitivity"]) for r in INV) == 175
    key = [k for k in MET["results"] if k.startswith("sensitivity")][0]
    assert [b for b in MET["results"][key] if b["group"] == "POOLED"][0]["n"] == 175


def test_excluding_the_sanity_failure_does_not_change_the_conclusion():
    a = [b for b in MET["results"]["primary_all_176"]
         if b["group"] == "POOLED"][0]["paired_B_minus_A_primary"]
    key = [k for k in MET["results"] if k.startswith("sensitivity")][0]
    b = [x for x in MET["results"][key]
         if x["group"] == "POOLED"][0]["paired_B_minus_A_primary"]
    assert a["ci_excludes_zero"] == b["ci_excludes_zero"]
    assert abs(a["delta"] - b["delta"]) < 0.01


# ------------------------------------------------------------------ nothing outside changed
def test_upstream_artifacts_are_unchanged():
    for rel, digest in MAN["files"].items():
        p = ROOT / rel
        if not p.exists():
            p = Path(rel)
        assert p.exists(), rel
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        assert h.hexdigest() == digest, "%s changed during this run" % rel
