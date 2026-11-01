# -*- coding: utf-8 -*-
"""Synthetic tests for the Dryad-rights / CLEAR concept audit. No weights, no images.

    py -3 -m pytest artifacts/dryad_rights_clear_audit_20261023_0900/tests -q
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(RUN / "src"))

import run_concept_audit as A                                        # noqa: E402


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    """Correctness of the analysis does not need 10,000 replicates; the real run does."""
    monkeypatch.setattr(A, "N_BOOT", 400)
    yield

MAP = json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------ split guards
def test_tune_only_loader_returns_exactly_the_tune_cohort():
    d = A.load_tune_only()
    split = json.loads(A.VN_SPLIT.read_text(encoding="utf-8"))
    assert len(d["pid"]) == len(split["tune"]) == 1352
    assert set(d["pid"]) == {str(x) for x in split["tune"]}


def test_no_report_pid_can_reach_the_analysis_set():
    d = A.load_tune_only()
    report = {str(x) for x in json.loads(A.VN_SPLIT.read_text(encoding="utf-8"))["report"]}
    assert not (set(d["pid"]) & report)


def _write_scores(tmp_path, rows, name="scores.csv"):
    import csv
    p = tmp_path / name
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return p


def _tune_rows():
    import csv
    split = json.loads(A.VN_SPLIT.read_text(encoding="utf-8"))
    tune = {str(x) for x in split["tune"]}
    real = list(csv.DictReader(io.open(A.VN_SCORES, encoding="utf-8-sig")))
    return [r for r in real if str(r["pid"]) in tune], split


def test_report_contamination_is_structurally_impossible_not_merely_checked():
    """A REPORT pid cannot reach the analysis set because selection is an allow-list.

    The loader selects `pid in tune` rather than excluding REPORT, so contamination is
    prevented by construction. The residual intersection assertion in the loader is
    therefore belt-and-braces and can never fire — which is why the two tests below
    exercise the guards that CAN fire.
    """
    import inspect
    src = inspect.getsource(A.load_tune_only)
    assert 'in tune' in src, "selection must be an allow-list against TUNE"
    d = A.load_tune_only()
    report = {str(x) for x in json.loads(A.VN_SPLIT.read_text(encoding="utf-8"))["report"]}
    assert not (set(d["pid"]) & report)


def test_loader_hard_fails_if_a_tune_patient_is_missing(monkeypatch, tmp_path):
    rows, _ = _tune_rows()
    monkeypatch.setattr(A, "VN_SCORES", _write_scores(tmp_path, rows[:-1]))
    with pytest.raises(A.SplitViolation, match="expected"):
        A.load_tune_only()


def test_loader_hard_fails_on_a_duplicate_patient(monkeypatch, tmp_path):
    rows, _ = _tune_rows()
    rows = rows[:-1] + [dict(rows[0])]          # drop one, duplicate another
    monkeypatch.setattr(A, "VN_SCORES", _write_scores(tmp_path, rows))
    with pytest.raises(A.SplitViolation, match="duplicate"):
        A.load_tune_only()


def test_loader_hard_fails_if_the_split_manifest_itself_overlaps(monkeypatch, tmp_path):
    split = json.loads(A.VN_SPLIT.read_text(encoding="utf-8"))
    split["report"] = split["report"] + [split["tune"][0]]
    bad = tmp_path / "bad_split.json"
    bad.write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(A, "VN_SPLIT", bad)
    with pytest.raises(A.SplitViolation, match="overlap"):
        A.load_tune_only()


def test_internal_test_is_never_referenced_in_any_source_file():
    for p in (RUN / "src").glob("*.py"):
        src = p.read_text(encoding="utf-8")
        for tok in ('"internal_test"', "'internal_test'", "internal_test/"):
            assert tok not in src, "%s references %s" % (p.name, tok)


def test_no_image_or_weight_in_the_bundle():
    bad = [p.relative_to(RUN).as_posix() for p in RUN.rglob("*")
           if p.is_file() and p.suffix.lower() in
           (".png", ".jpg", ".dcm", ".pt", ".pth", ".npz", ".safetensors", ".pickle")]
    assert not bad, "bundle must be code-only; found %s" % bad


def test_no_per_case_table_is_written_to_the_bundle():
    for p in (RUN / "tables").glob("*.csv"):
        n = sum(1 for _ in io.open(p, encoding="utf-8")) - 1
        assert n < 100, "%s has %d rows - looks per-case, not aggregate" % (p.name, n)


# ------------------------------------------------------------ concept mapping
def test_only_exact_vocabulary_entries_are_used():
    primary = [c for c in MAP["mapping"] if c["concept_idx"] is not None]
    assert len(primary) == MAP["n_primary"] == 9
    for c in primary:
        assert c["exact_vocab_entry"] == c["concept"]
        assert "EXACT" in c["status"]


def test_concepts_without_an_exact_entry_are_excluded_not_approximated():
    excluded = [c for c in MAP["mapping"] if c["concept_idx"] is None]
    assert {c["concept"] for c in excluded} == {"cavity", "nodule", "reticular opacity"}
    for c in excluded:
        assert "excluded" in c["status"].lower()
    assert "not approximated" in MAP["rule"]


def test_concept_indices_are_in_range_and_unique():
    idx = [c["concept_idx"] for c in MAP["mapping"] if c["concept_idx"] is not None]
    assert len(set(idx)) == len(idx)
    assert all(0 <= i < 368294 for i in idx)


def test_pinned_revision_is_a_commit_not_a_branch():
    rev = MAP["pinned_revision"]
    assert len(rev) == 40 and all(ch in "0123456789abcdef" for ch in rev)
    assert rev != "main"


# ------------------------------------------------------------ analysis correctness
def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    past = (rng.random(n) < 0.35).astype(int)
    d = {"pid": [str(i) for i in range(n)], "y": y, "past_tb": past,
         "rad_system": rng.integers(1, 4, n), "age_group": rng.integers(1, 3, n),
         "sex": rng.integers(0, 2, n), "symptoms": rng.integers(0, 5, n)}
    return d, rng


def test_analysis_recovers_a_planted_association_and_a_null():
    d, rng = _synthetic()
    n = len(d["y"])
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    d["clf_score"] = 2.0 * signal + rng.normal(0, 0.5, n)
    S = np.column_stack([signal, noise])
    rows, incr = A.analyse(d, S, ["planted", "null"])
    neg = [r for r in rows if r["group"] == "negatives (all)"]
    planted = [r for r in neg if r["concept"] == "planted"][0]
    nul = [r for r in neg if r["concept"] == "null"][0]
    assert planted["spearman_rho"] > 0.6 and planted["ci95_lo"] > 0
    assert abs(nul["spearman_rho"]) < 0.2 and nul["ci95_lo"] < 0 < nul["ci95_hi"]


def test_bh_adjustment_is_applied_and_monotone():
    d, rng = _synthetic(seed=1)
    n = len(d["y"])
    d["clf_score"] = rng.normal(size=n)
    S = rng.normal(size=(n, 5))
    rows, _ = A.analyse(d, S, list("abcde"))
    fam = [r for r in rows if r["group"] == "negatives (all)"]
    assert all(r["p_bh_adjusted"] is not None for r in fam)
    assert all(r["p_bh_adjusted"] >= r["p_raw"] - 1e-12 for r in fam)
    assert all(0 <= r["p_bh_adjusted"] <= 1 for r in fam)
    other = [r for r in rows if r["group"] != "negatives (all)"]
    assert all(r["p_bh_adjusted"] is None for r in other)


def test_incremental_r2_is_zero_for_a_pure_noise_concept():
    d, rng = _synthetic(seed=2)
    n = len(d["y"])
    d["clf_score"] = rng.normal(size=n)
    S = rng.normal(size=(n, 1))
    _, incr = A.analyse(d, S, ["noise"])
    assert abs(incr[0]["delta_r2"]) < 0.02


def test_groups_too_small_are_skipped_not_reported_as_zero():
    d, rng = _synthetic(n=60, seed=3)
    d["clf_score"] = rng.normal(size=60)
    rows, _ = A.analyse(d, np.random.default_rng(0).normal(size=(60, 1)), ["x"])
    for r in rows:
        assert r["n"] >= 20


def test_case_level_resampling_is_appropriate_here():
    """Vietnam is one image per patient, so case-level == patient-level."""
    d = A.load_tune_only()
    assert len(set(d["pid"])) == len(d["pid"])


# ------------------------------------------------------------ data integrity
def test_tune_cohort_has_no_missing_values_in_the_fields_used():
    d = A.load_tune_only()
    for k in ("y", "past_tb", "rad_system", "age_group", "sex", "symptoms", "clf_score"):
        v = np.asarray(d[k])
        assert len(v) == 1352
        assert np.isfinite(v.astype(float)).all(), k


def test_label_and_history_values_are_in_their_documented_domains():
    d = A.load_tune_only()
    assert set(np.unique(d["y"])) <= {0, 1}
    assert set(np.unique(d["past_tb"])) <= {0, 1}
    assert set(np.unique(d["rad_system"])) <= {1, 2, 3}
