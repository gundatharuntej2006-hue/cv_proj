# -*- coding: utf-8 -*-
"""Tests for the stacked-levers run: guards, gates, statistics, partitions.

    py -3 -m pytest artifacts/stacked_levers_20261101_0700/tests -q
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
sys.path.insert(0, str(RUN / "scripts"))


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_protocol_hash_unchanged():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    assert h == (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]


def test_both_gates_passed():
    g = {r["arm"]: r for r in rows(TAB / "reproduction_gate.csv")}
    assert g["A"]["pass"] == "True" and abs(float(g["A"]["here"]) - 0.9498) < 5e-4
    assert g["MB1"]["pass"] == "True" and abs(float(g["MB1"]["here"]) - 0.9497) < 5e-4


def test_nlm_cache_is_maira2_on_the_same_images():
    p = RUN / "cache" / "regional_D2_nlm_lungs_g12.npz"
    if not p.is_file():
        pytest.skip("cache absent")
    with np.load(p) as z, np.load(ROOT / "artifacts/p2/regional/regional_nlm_lungs_g12_v1.npz") as r:
        assert z["reg"].shape == (800, 144, 768)
        assert set(z["image_id"].tolist()) == set(r["image_id"].tolist())
        X = z["reg"][:16].astype(np.float32)
    assert np.allclose(np.linalg.norm(X, axis=2), 1.0, atol=5e-3)


def test_primary_verdict_follows_the_rule():
    import stacked as T
    p = [r for r in rows(TAB / "paired_differences.csv") if r["primary"] == "True"]
    assert len(p) == 1 and p[0]["contrast"] == "X - A"
    lo, hi = [float(v) for v in p[0]["cal_ci"].strip("[]").split(",")]
    assert p[0]["verdict"] == T.verdict(float(p[0]["cal_delta"]), lo, hi)


def test_manifest_records_additivity_and_encoder():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["encoder_X"]["revision"].startswith("a6132efb")
    assert abs(m["additivity"]["sum_of_single_lever_tune_deltas"] - 0.0468) < 1e-3
    d = [r for r in rows(TAB / "paired_differences.csv") if r["contrast"] == "X - A"][0]
    assert abs(float(d["tune_delta_EXPLORATORY"]) - m["additivity"]["observed_X_minus_A_tune"]) < 1e-6


def test_seed_spread_is_small():
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows(TAB / "per_seed.csv"):
        by[r["arm"]].append(float(r["cal_auroc_vs_active"]))
    assert set(by) == {"A", "MB1", "X", "X_noNLM", "X_tb"} and all(len(v) == 5 for v in by.values())
    assert all(max(v) - min(v) < 0.006 for v in by.values())


def test_partitions_and_no_pid():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["partitions"]["vietnam_report_used"] is False
    assert m["partitions"]["tbx11k_internal_test_used"] is False
    assert m["production_changed"] is False
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    pids = {str(i) for i in sp["tune"]} | {str(i) for i in sp["report"]}
    for f in RUN.rglob("*.csv"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        assert not [p for p in pids if p in t], f.name
