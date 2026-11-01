# -*- coding: utf-8 -*-
"""Tests for the B1 x CLEAR mechanism run.

    py -3 -m pytest artifacts/b1_clear_mechanism_20261031_2328/tests -q
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
PRED = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/predictions"
SPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VOCAB = Path(r"D:\hf-cache\clear_vocab\mimic_concepts.csv")

sys.path.insert(0, str(PKG / "src"))


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tables_exist():
    return (TAB / "aggregate_results.csv").is_file()


# ------------------------------------------------------------- TUNE-only enforcement
def test_no_report_pid_in_either_arm_prediction_file():
    report = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["report"]}
    tune = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["tune"]}
    assert len(report) == 1356 and len(tune) == 1352
    for f in ("vietnam_tune_armA.csv", "vietnam_tune_armB1.csv"):
        pids = {r["pid"] for r in rows(PRED / f)}
        assert len(pids) == 1352, f
        assert not (pids & report), f
        assert pids == tune, f


def test_the_loader_refuses_a_cohort_that_is_not_tune():
    """The split guard is live, not decorative: feeding it REPORT pids must raise."""
    import concept_audit as A
    import paths as P
    P._ASSET_DIRS = [RUN / "configs"] + list(P._ASSET_DIRS)
    p = P.resolve(prefer="repo")
    orig = Path(p["split"]).read_text(encoding="utf-8")
    sp = json.loads(orig)
    swapped = dict(sp)
    swapped["tune"] = sp["report"][:1352]          # same size, wrong cohort
    import io
    import json as _j
    real_open = io.open

    def fake_open(path, *a, **k):
        if str(path) == str(p["split"]):
            return io.StringIO(_j.dumps(swapped))
        return real_open(path, *a, **k)

    io.open = fake_open
    try:
        with pytest.raises(A.SplitViolation):
            A.load_tune_only(p)
    finally:
        io.open = real_open


def test_report_images_sit_beside_the_approved_ones_and_are_still_not_read():
    """The repo layout exposes all 2,708 images. Only the 1,352 approved are opened."""
    import concept_audit as A
    import paths as P
    P._ASSET_DIRS = [RUN / "configs"] + list(P._ASSET_DIRS)
    p = P.resolve(prefer="repo")
    have = P.image_ids(p)
    assert len(have) == 2708, "expected the full cohort on disk"
    d = A.load_tune_only(p)
    missing, extra = A.check_images(p, d["pid"])
    assert missing == []
    assert len(extra) == 1356, "the REPORT images are present but unapproved"
    report = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["report"]}
    assert set(extra) == report


def test_preflight_records_the_locked_partitions_as_untouched():
    if not (TAB / "preflight.json").is_file():
        pytest.skip("run not executed yet")
    pre = json.loads((TAB / "preflight.json").read_text(encoding="utf-8"))
    assert pre["partitions"]["vietnam_report_used"] is False
    assert pre["partitions"]["tbx11k_internal_test_used"] is False
    assert pre["partitions"]["report_pids_in_cohort"] == 0
    assert pre["cohort"]["n"] == 1352
    assert pre["cohort"]["fingerprint_sha256"] == \
        json.loads((PKG / "configs/tune_cohort_fingerprint.json")
                   .read_text(encoding="utf-8"))["fingerprint_sha256"]


# ------------------------------------------------------------- no per-case export
def test_no_output_file_contains_a_patient_identifier():
    """Every written table must be aggregate. A pid appearing anywhere is a leak."""
    if not tables_exist():
        pytest.skip("run not executed yet")
    pids = {str(i) for i in json.loads(SPLIT.read_text(encoding="utf-8"))["tune"]}
    for f in sorted(RUN.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".csv", ".json", ".md", ".txt"):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        hits = [p for p in pids if p in text]
        assert not hits, "%s contains pid(s) %s" % (f.name, hits[:3])


def test_no_per_case_array_was_written_into_the_run_directory():
    if not tables_exist():
        pytest.skip("run not executed yet")
    assert not list(RUN.rglob("*.npy")), "per-case shards must not live here"
    assert not list(RUN.rglob("_shards")), "shard directory must not live here"
    for f in RUN.rglob("*.csv"):
        n = len(rows(f))
        assert n < 400, "%s has %d rows; a 1,352-row table would be per-case" % (f.name, n)


def test_scratch_directory_was_cleaned_up():
    if not (TAB / "preflight.json").is_file():
        pytest.skip("run not executed yet")
    scratch = Path(r"D:\tb-triage-scratch\b1_clear_mechanism_20261031_2328")
    assert not scratch.exists(), "scratch still holds per-case similarities"


# ------------------------------------------------------------- ID alignment
def test_the_two_arms_are_row_aligned_by_id_not_by_position():
    a = rows(PRED / "vietnam_tune_armA.csv")
    b = rows(PRED / "vietnam_tune_armB1.csv")
    assert [r["pid"] for r in a] == [r["pid"] for r in b]
    assert [r["xpert_result"] for r in a] == [r["xpert_result"] for r in b]
    assert sum(int(r["xpert_result"]) for r in a) == 139


def test_arm_scores_are_reordered_by_id_when_the_file_order_differs():
    """load_arm_scores must key on pid. Shuffle the file order and demand the same vector."""
    sys.path.insert(0, str(RUN / "scripts"))
    import run_mechanism as M
    a = rows(PRED / "vietnam_tune_armA.csv")
    pids = [r["pid"] for r in a]
    straight, _ = M.load_arm_scores("A", pids)
    shuffled = list(reversed(pids))
    rev, _ = M.load_arm_scores("A", shuffled)
    assert np.allclose(straight, rev[::-1])
    assert not np.allclose(straight, rev), "the test itself would be vacuous"


def test_a_missing_pid_is_a_blocker_not_a_silent_drop():
    sys.path.insert(0, str(RUN / "scripts"))
    import run_mechanism as M
    pids = [r["pid"] for r in rows(PRED / "vietnam_tune_armA.csv")]
    with pytest.raises(SystemExit):
        M.load_arm_scores("A", pids[:-1] + ["9999999"])


# ------------------------------------------------------------- concept mapping
def test_the_nine_pinned_concepts_are_copied_verbatim():
    mine = json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))
    orig = json.loads((PKG / "configs/concept_indices.json").read_text(encoding="utf-8"))
    by = {c["concept"]: c for c in mine["mapping"]}
    for c in orig["mapping"]:
        assert by[c["concept"]]["concept_idx"] == c["concept_idx"], c["concept"]
        assert by[c["concept"]]["exact_vocab_entry"] == c["exact_vocab_entry"]


def test_the_excluded_concepts_are_still_excluded_and_not_approximated():
    mine = json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))
    excluded = {c["concept"] for c in mine["mapping"] if c["concept_idx"] is None}
    assert excluded == {"cavity", "nodule", "reticular opacity"}


def test_every_index_re_derives_from_the_pinned_vocabulary():
    if not VOCAB.is_file():
        pytest.skip("vocabulary not cached locally")
    assert hashlib.sha256(VOCAB.read_bytes()).hexdigest() == \
        "93e975b722422b64f18b57161ac9894fe30ac7a04cc68b7aa8a2e01f6d0fc439"
    idx = {}
    with open(VOCAB, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            idx.setdefault(r["concept"].strip().lower(), []).append(int(r["concept_idx"]))
    mine = json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))
    for c in mine["mapping"]:
        if c["concept_idx"] is None:
            assert c["exact_vocab_entry"] is None
            assert c["concept"].lower() not in idx, \
                "%s has an exact entry after all" % c["concept"]
            continue
        got = idx.get(c["exact_vocab_entry"].strip().lower())
        assert got == [c["concept_idx"]], (c["concept"], got, c["concept_idx"])


def test_the_negative_controls_are_exact_entries_and_outside_the_bh_family():
    mine = json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))
    ctrl = [c for c in mine["mapping"] if c.get("role")]
    assert [c["concept"] for c in ctrl] == ["scoliosis", "degenerative changes"]
    for c in ctrl:
        assert c["status"].startswith("EXACT")
        assert c["concept_idx"] is not None
        assert c["concept"] not in mine["bh_family"]
    assert len(mine["bh_family"]) == 9


def test_bh_adjustment_covers_exactly_the_nine_concept_family():
    if not tables_exist():
        pytest.skip("run not executed yet")
    fam = set(json.loads((RUN / "configs/concept_indices.json")
                         .read_text(encoding="utf-8"))["bh_family"])
    rs = rows(TAB / "aggregate_results.csv")
    for arm in {r["arm"] for r in rs}:
        adj = [r for r in rs if r["arm"] == arm and r["p_bh_adjusted"]]
        assert len(adj) == 9, arm
        assert {r["concept"] for r in adj} == fam
        assert all(r["group"] == "negatives (all)" for r in adj)
        for r in adj:
            assert float(r["p_bh_adjusted"]) >= float(r["p_raw"]) - 1e-12
    controls = {r["concept"] for r in rs if r["is_negative_control"] == "True"}
    assert controls == {"scoliosis", "degenerative changes"}
    assert not [r for r in rs if r["is_negative_control"] == "True" and r["p_bh_adjusted"]]


# ------------------------------------------------------------- paired bootstrap
def test_paired_bootstrap_preserves_pairing():
    """Shared indices: differencing two identical score vectors must give exactly zero."""
    sys.path.insert(0, str(RUN / "scripts"))
    import run_mechanism as M
    rng = np.random.default_rng(0)
    n = 300
    d = {"y": np.zeros(n, int), "past_tb": rng.integers(0, 2, n),
         "rad_system": rng.integers(1, 4, n), "age_group": rng.integers(0, 4, n),
         "sex": rng.integers(0, 2, n), "symptoms": rng.integers(0, 5, n)}
    S = rng.normal(size=(n, 2))
    s = S[:, 0] * 0.7 + rng.normal(size=n) * 0.5
    M.PAIRED_BOOT = 200
    out = M.paired_differences(d, S, ["a", "b"], s, s.copy(), log=lambda *_: None)
    for r in out:
        assert r["delta_rho_B1_minus_B0"] == 0.0
        assert r["delta_rho_ci95_lo"] == 0.0 and r["delta_rho_ci95_hi"] == 0.0
        assert r["delta_incr_r2_ci95_lo"] == 0.0 and r["delta_incr_r2_ci95_hi"] == 0.0


def test_paired_bootstrap_detects_a_real_difference():
    sys.path.insert(0, str(RUN / "scripts"))
    import run_mechanism as M
    rng = np.random.default_rng(1)
    n = 600
    d = {"y": np.zeros(n, int), "past_tb": rng.integers(0, 2, n),
         "rad_system": rng.integers(1, 4, n), "age_group": rng.integers(0, 4, n),
         "sex": rng.integers(0, 2, n), "symptoms": rng.integers(0, 5, n)}
    c = rng.normal(size=n)
    S = c.reshape(-1, 1)
    s0 = c * 1.2 + rng.normal(size=n) * 0.4          # strongly tied to the concept
    s1 = c * 0.1 + rng.normal(size=n) * 1.0          # barely tied
    M.PAIRED_BOOT = 500
    r = M.paired_differences(d, S, ["c"], s0, s1, log=lambda *_: None)[0]
    assert r["delta_rho_B1_minus_B0"] < -0.3
    assert r["delta_rho_excludes_zero"] is True
    assert r["delta_rho_ci95_hi"] < 0


def test_resampling_unit_is_the_patient():
    if not (TAB / "preflight.json").is_file():
        pytest.skip("run not executed yet")
    pre = json.loads((TAB / "preflight.json").read_text(encoding="utf-8"))
    assert "patient" in pre["statistics"]["resampling_unit"]
    assert pre["statistics"]["paired_n_boot"] == 10000
    assert pre["statistics"]["paired_seed"] == 20261031


# ------------------------------------------------------------- integrity
def test_manifest_hash_still_matches():
    h = hashlib.sha256((RUN / "ANALYSIS_MANIFEST.md").read_bytes()).hexdigest()
    assert h == (TAB / "manifest_hash.txt").read_text(encoding="utf-8").split()[0]


def test_the_audited_package_was_not_modified():
    before = {}
    for line in (TAB / "protected_hashes_before.txt").read_text(encoding="utf-8").splitlines():
        h, _, p = line.partition(" ")
        before[p.strip().lstrip("*")] = h
    for p, h in before.items():
        f = Path(p)
        if not f.is_absolute():
            f = ROOT / p
        if not f.is_file():
            continue
        assert hashlib.sha256(f.read_bytes()).hexdigest() == h, p


def test_reproduction_gate_passed_if_the_run_executed():
    f = TAB / "reproduction_gate.csv"
    if not f.is_file():
        pytest.skip("run not executed yet")
    rs = rows(f)
    assert len(rs) == 6
    assert all(r["pass"] == "True" for r in rs), \
        [r["quantity"] for r in rs if r["pass"] != "True"]
