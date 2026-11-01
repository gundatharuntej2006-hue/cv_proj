# -*- coding: utf-8 -*-
"""Tests for the consolidation / B0-B1 forensic run.

    py -3 -m pytest artifacts/consolidation_b0_b1_forensic_20261101_0123/tests -q
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
PKG = ROOT / "artifacts/kaggle_exec_prep_20261023_1130"
PRIOR = ROOT / "artifacts/b1_clear_mechanism_20261031_2328"
PRED = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/predictions"
SPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VOCAB = Path(r"D:\hf-cache\clear_vocab\mimic_concepts.csv")
VOCAB_SHA = "93e975b722422b64f18b57161ac9894fe30ac7a04cc68b7aa8a2e01f6d0fc439"


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def conf():
    return json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))


def ran():
    return (TAB / "b0_b1_concept_associations.csv").is_file()


# ---------------------------------------------------------------- partition guards
def test_no_report_pid_anywhere_in_this_run():
    report = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["report"]}
    assert len(report) == 1356
    for f in RUN.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".csv", ".json", ".md", ".txt", ".py"):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        hits = [p for p in report if p in text]
        assert not hits, "%s mentions REPORT pid(s) %s" % (f.name, hits[:3])


def test_no_tune_pid_is_written_into_this_directory():
    tune = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["tune"]}
    assert len(tune) == 1352
    for f in RUN.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".csv", ".json", ".md", ".txt"):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        hits = [p for p in tune if p in text]
        assert not hits, "%s leaks pid(s) %s" % (f.name, hits[:3])


def test_no_tbx11k_artifact_is_referenced_as_an_input():
    if not (TAB / "provenance_inputs.csv").is_file():
        pytest.skip("provenance not written")
    for r in rows(TAB / "provenance_inputs.csv"):
        assert "internal_test" not in r["path"]
        assert "tbx11k" not in r["path"].lower() or "label" not in r["path"].lower()


def test_scratch_was_removed():
    if not (TAB / "metric_reproduction.json").is_file():
        pytest.skip("run not finished")
    assert not Path(r"D:\tb-triage-scratch\consolidation_forensic_20261101_0123").exists()
    assert not list(RUN.rglob("*.npy"))
    assert not list(RUN.rglob("_shards"))
    assert not (RUN / "configs/_shadow").exists(), "the shadow config must be cleaned up"


def test_no_table_is_long_enough_to_be_per_case():
    if not ran():
        pytest.skip("run not finished")
    for f in TAB.glob("*.csv"):
        assert len(rows(f)) < 900, "%s has %d rows" % (f.name, len(rows(f)))


# ---------------------------------------------------------------- mapping
def test_the_nine_primary_indices_are_identical_across_all_three_runs():
    orig = {c["concept"]: c["concept_idx"]
            for c in json.loads((PKG / "configs/concept_indices.json")
                                .read_text(encoding="utf-8"))["mapping"]
            if c["concept_idx"] is not None}
    prev = {c["concept"]: c["concept_idx"]
            for c in json.loads((PRIOR / "configs/concept_indices.json")
                                .read_text(encoding="utf-8"))["mapping"]
            if c["concept_idx"] is not None}
    mine = {c["concept"]: c["concept_idx"] for c in conf()["audited"]}
    assert len(orig) == 9
    for k, v in orig.items():
        assert prev[k] == v and mine[k] == v, k


def test_the_family_reconciles_as_9_11_17():
    c = conf()
    assert len(c["audited"]) == 11
    assert len([a for a in c["audited"] if a["role"] == "primary"]) == 9
    assert len([a for a in c["audited"] if a["role"] == "negative control"]) == 2
    assert len(c["diagnostic_probes"]) == 6
    assert len(c["bh_family"]) == 9
    assert set(c["bh_family"]) == {a["concept"] for a in c["audited"] if a["role"] == "primary"}


def test_every_declared_index_re_derives_from_the_pinned_vocabulary():
    if not VOCAB.is_file():
        pytest.skip("vocabulary absent")
    assert hashlib.sha256(VOCAB.read_bytes()).hexdigest() == VOCAB_SHA
    idx = {}
    for r in rows(VOCAB):
        idx.setdefault(r["concept"].strip().lower(), []).append(int(r["concept_idx"]))
    for c in conf()["audited"] + conf()["diagnostic_probes"]:
        assert idx.get(c["concept"].lower()) == [c["concept_idx"]], c["concept"]


def test_consolidation_is_the_bare_word_and_is_unique():
    if not VOCAB.is_file():
        pytest.skip("vocabulary absent")
    n = sum(1 for r in rows(VOCAB) if r["concept"].strip().lower() == "consolidation")
    assert n == 1
    by = {int(r["concept_idx"]): r["concept"] for r in rows(VOCAB)}
    assert by[123742] == "consolidation"


def test_probes_are_excluded_from_the_bh_family():
    c = conf()
    names = {p["concept"] for p in c["diagnostic_probes"]}
    assert not (names & set(c["bh_family"]))
    if not ran():
        pytest.skip("run not finished")
    for r in rows(TAB / "b0_b1_concept_associations.csv"):
        if r.get("is_diagnostic_probe") == "True" or r.get("is_negative_control") == "True":
            assert not r.get("p_bh_adjusted"), r["concept"]


def test_bh_covers_exactly_the_nine_primary_concepts_per_arm():
    if not ran():
        pytest.skip("run not finished")
    fam = set(conf()["bh_family"])
    rs = rows(TAB / "b0_b1_concept_associations.csv")
    for arm in {r["arm"] for r in rs}:
        adj = [r for r in rs if r["arm"] == arm and r.get("p_bh_adjusted")]
        assert len(adj) == 9, (arm, len(adj))
        assert {r["concept"] for r in adj} == fam
        assert all(r["group"] == "negatives (all)" for r in adj)
        for r in adj:
            assert float(r["p_bh_adjusted"]) >= float(r["p_raw"]) - 1e-12


# ---------------------------------------------------------------- alignment
def test_arm_files_align_by_id_and_carry_identical_labels():
    a, b = rows(PRED / "vietnam_tune_armA.csv"), rows(PRED / "vietnam_tune_armB1.csv")
    assert [r["pid"] for r in a] == [r["pid"] for r in b]
    assert [r["xpert_result"] for r in a] == [r["xpert_result"] for r in b]
    assert len({r["pid"] for r in a}) == 1352
    assert sum(int(r["xpert_result"]) for r in a) == 139


def test_arm_lookup_is_keyed_on_pid_not_position():
    sys.path.insert(0, str(RUN / "scripts"))
    import b1_image_side as M
    pids = [r["pid"] for r in rows(PRED / "vietnam_tune_armA.csv")]
    straight, _ = M.arm_scores("A", pids)
    rev, _ = M.arm_scores("A", list(reversed(pids)))
    assert np.allclose(straight, rev[::-1])
    assert not np.allclose(straight, rev)
    with pytest.raises(SystemExit):
        M.arm_scores("A", pids[:-1] + ["9999999"])


def test_b0_is_production_identical():
    if not (TAB / "metric_reproduction.json").is_file():
        pytest.skip("run not finished")
    b = json.loads((TAB / "metric_reproduction.json").read_text(encoding="utf-8"))
    ident = b["b0_identity_with_production"]
    assert ident["spearman"] == 1.0 and ident["pearson"] == 1.0
    assert ident["max_abs_difference"] <= 1e-05


# ---------------------------------------------------------------- statistics
def test_r2_matches_an_independent_ols_implementation():
    sys.path.insert(0, str(RUN / "scripts"))
    import b1_image_side as M
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(250, 6)), rng.normal(size=250)
    assert abs(M.r2(X, y) - LinearRegression().fit(X, y).score(X, y)) < 1e-10


def test_reproduction_targets_were_met():
    if not (TAB / "metric_reproduction.json").is_file():
        pytest.skip("run not finished")
    b = json.loads((TAB / "metric_reproduction.json").read_text(encoding="utf-8"))
    assert b["reproduction"]["all_passed"] is True, b["reproduction"]


def test_the_prior_consolidation_values_are_what_we_claim_they_were():
    """Read the previous run's committed file, not a summary of it."""
    prev = {(r["concept"], r["group"]): float(r["spearman_rho"])
            for r in rows(PRIOR / "tables/aggregate_results.csv") if r["arm"] == "B0"}
    assert abs(prev[("consolidation", "negatives (all)")] - (-0.3000)) < 5e-4
    assert abs(prev[("scarring", "negatives (all)")] - 0.5661) < 5e-4
    old = {(r["concept"], r["group"]): float(r["spearman_rho"])
           for r in rows(ROOT / "artifacts/concept_audit_result_20261024_0130"
                              "/outputs/tables/concept_associations.csv")}
    assert abs(old[("consolidation", "positives (all)")] - (-0.3849)) < 5e-4


def test_negation_pairs_come_only_from_existing_vocabulary_entries():
    if not (TAB / "negation_sensitivity.csv").is_file():
        pytest.skip("semantics not run")
    if not VOCAB.is_file():
        pytest.skip("vocabulary absent")
    idx = {}
    for r in rows(VOCAB):
        idx.setdefault(r["concept"].strip().lower(), []).append(int(r["concept_idx"]))
    for r in rows(TAB / "negation_sensitivity.csv"):
        assert idx.get(r["negated_phrase"].lower()) == [int(r["negated_idx"])]
        assert idx.get(r["positive_phrase"].lower()) == [int(r["positive_idx"])]


def test_run_recorded_no_training_and_no_production_change():
    if not (TAB / "metric_reproduction.json").is_file():
        pytest.skip("run not finished")
    b = json.loads((TAB / "metric_reproduction.json").read_text(encoding="utf-8"))
    assert b["training_performed"] is False
    assert b["partitions"]["vietnam_report_used"] is False
    assert b["partitions"]["tbx11k_internal_test_used"] is False
    assert b["partitions"]["report_pids_in_cohort"] == 0
    assert b["cohort"]["n"] == 1352 and b["cohort"]["n_positive"] == 139
    assert b["cohort"]["fingerprint_sha256"] == json.loads(
        (PKG / "configs/tune_cohort_fingerprint.json").read_text(encoding="utf-8")
    )["fingerprint_sha256"]


def test_upstream_artifacts_were_not_modified():
    if not (TAB / "provenance_inputs.csv").is_file():
        pytest.skip("provenance not written")
    for r in rows(TAB / "provenance_inputs.csv"):
        if r["sha256"] == "ABSENT":
            continue
        p = Path(r["path"])
        if not p.is_absolute():
            p = ROOT / r["path"]
        if not p.is_file() or p.stat().st_size > 50_000_000:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                h.update(c)
        assert h.hexdigest() == r["sha256"], r["path"]
