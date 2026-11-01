# -*- coding: utf-8 -*-
"""Tests for case alignment, label mapping, unchanged baseline artifacts and the formula.

Scoring was not performed (Phase 0 precondition 6 failed), so the "unchanged baseline
scores" test is implemented as what it can be: an assertion that the committed DA/DB
artifacts still hash to the values recorded in this run's manifest, i.e. that nothing in
this run touched them.

    py -3 -m pytest artifacts/dadb_offset_validation_20261021_0255/tests -q
"""
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(RUN / "scripts"))

import verify_preconditions as V                                     # noqa: E402

REPORT = json.loads((RUN / "tables/precondition_report.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ case alignment
def test_case_count_reconciles_with_the_published_figure():
    cases = V.dadb_cases()
    assert len(cases) == 176, "cohort size does not reconcile with dadb_third_site.json"


def test_case_ids_are_unique():
    cases = V.dadb_cases()
    ids = [c["case_id"] for c in cases]
    assert len(set(ids)) == len(ids)


def test_ordering_is_deterministic_and_reproducible():
    """The npz stores no image_id, so the sorted-path order IS the alignment key."""
    a = [c["case_id"] for c in V.dadb_cases()]
    b = [c["case_id"] for c in V.dadb_cases()]
    assert a == b
    paths = sorted(p for p in V.SRC.rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    assert a == [p.stem for p in paths], "order differs from the committed images() order"


def test_embeddings_have_one_row_per_case_and_no_stored_ids():
    import numpy as np
    n = len(V.dadb_cases())
    for v in ("full", "lungs_only", "border_only"):
        with np.load(ROOT / ("artifacts/p2/external/radino_dadb_%s_v1.npz" % v)) as z:
            assert z["emb"].shape == (n, 768)
            assert "image_id" not in z, \
                "if ids were stored, alignment should use them rather than file order"


# ------------------------------------------------------------------ label mapping
def test_label_mapping_matches_the_committed_rule():
    """p -> 1, anything else -> 0, taken from the filename, as in p2_26."""
    for c in V.dadb_cases():
        assert c["label"] == (1 if c["filename"][0] == "p" else 0)
    chars = Counter(c["filename"][0] for c in V.dadb_cases())
    assert set(chars) == {"n", "p"}, "unexpected filename prefix: %s" % dict(chars)


def test_per_site_counts_match_the_published_result_file():
    pub = json.loads(
        (ROOT / "artifacts/p2/external/dadb_third_site.json").read_text(encoding="utf-8"))
    bysite = {r["site"]: (r["n"], r["n_tb"]) for r in pub["results"] if r["variant"] == "full"}
    cases = V.dadb_cases()
    for site in ("DA", "DB"):
        n = sum(1 for c in cases if c["site"] == site)
        k = sum(c["label"] for c in cases if c["site"] == site)
        assert (n, k) == bysite[site], site
    assert (len(cases), sum(c["label"] for c in cases)) == bysite["DA+DB"]


def test_site_mapping_follows_the_file_suffix():
    for c in V.dadb_cases():
        assert c["site"] in ("DA", "DB")
    assert {c["site"] for c in V.dadb_cases()} == {"DA", "DB"}


# ------------------------------------------------------------------ baseline unchanged
def test_committed_dadb_artifacts_are_unchanged_by_this_run():
    for name, rec in REPORT["protocol_sources"].items():
        p = ROOT / rec["path"]
        assert p.exists(), name
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        assert h.hexdigest() == rec["sha256"], "%s changed during this run" % name


def test_dadb_does_not_overlap_the_training_or_development_splits():
    assert REPORT["provenance"]["overlap_with_provisional_c01"] == 0
    assert REPORT["provenance"]["overlap_with_core_trainval_labels"] == 0


# ------------------------------------------------------------------ the correction formula
def test_the_frozen_offsets_are_keyed_to_vietnam_scanners_only():
    keys = set(REPORT["frozen_offsets"]["parameters"])
    assert keys == {"1", "2", "3"}, keys
    assert "rad_system" in REPORT["frozen_offsets"]["keyed_by"]


def test_no_dadb_site_maps_onto_a_frozen_offset_group():
    """DA and DB are not Vietnam rad_system values; guessing a mapping is forbidden."""
    sites = {c["site"] for c in V.dadb_cases()}
    assert not (sites & set(REPORT["frozen_offsets"]["parameters"]))


def test_the_dadb_estimator_requires_labels_which_disqualifies_it():
    d = REPORT["dadb_estimator_in_NEXT_STEP"]
    assert d["requires_dadb_non_tb_labels"] is True
    assert d["fits_offsets_on_dadb"] is True


def test_the_offset_formula_is_rank_preserving_within_a_site():
    """Pinned because the whole interpretation depends on it: a per-site additive shift
    cannot reorder cases inside that site, so within-site AUROC is invariant and only the
    pooled figure can move."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    s = rng.standard_normal(120)
    site = rng.integers(0, 2, 120)
    c = s.copy()
    for g, off in ((0, 1.37), (1, -2.04)):
        c[site == g] -= off
    for g in (0, 1):
        m = site == g
        assert abs(roc_auc_score(y[m], s[m]) - roc_auc_score(y[m], c[m])) < 1e-12


# ------------------------------------------------------------------ the stop itself
def test_no_scores_were_produced():
    assert REPORT["scores_computed"] is False
    rows = list(csv.DictReader(io.open(RUN / "tables/dadb_case_inventory.csv",
                                       encoding="utf-8-sig")))
    assert len(rows) == 176
    assert all(r["baseline_score"] == "NOT_SCORED" for r in rows)
    assert all(r["corrected_score"] == "NOT_SCORED" for r in rows)


def test_the_verdict_is_not_evaluable():
    assert REPORT["precondition_6_application_at_inference"]["verdict"].startswith(
        "NOT EVALUABLE")
