# -*- coding: utf-8 -*-
"""Tests for the multi-source run: guards, gate, statistics, partitions.

    py -3 -m pytest artifacts/multisource_head_20261101_0500/tests -q
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

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


def test_nlm_label_parser():
    import multisource as M
    assert M.nlm_label("shz__CHNCXR_0001_0") == 0
    assert M.nlm_label("mcu__MCUCXR_0100_1") == 1
    with pytest.raises(SystemExit):
        M.nlm_label("shz__CHNCXR_0001_x")


def test_training_sets_as_protocol():
    t = json.loads((TAB / "training_sets.json").read_text(encoding="utf-8"))
    assert t["tbx11k_train"] == 3241 and t["development"] == 659 and t["calibration"] == 899
    assert t["nlm"]["shz"] == {"n": 662, "tb": 336} and t["nlm"]["mcu"] == {"n": 138, "tb": 58}
    assert t["nlm_in_development"] is False


def test_reproduction_gate_passed():
    g = {r["arm"]: r for r in rows(TAB / "reproduction_gate.csv")}
    assert g["A"]["pass"] == "True" and g["B1"]["pass"] == "True"
    assert abs(float(g["A"]["here"]) - 0.9498) < 5e-4
    assert abs(float(g["B1"]["here"]) - 0.9460) < 5e-4


def test_primary_verdict_follows_the_rule():
    import multisource as M
    p = [r for r in rows(TAB / "paired_differences.csv") if r["primary"] == "True"][0]
    assert p["contrast"] == "M - A"
    lo, hi = [float(v) for v in p["cal_ci"].strip("[]").split(",")]
    assert p["verdict"] == M.verdict(float(p["cal_delta"]), lo, hi)


def test_verdict_rule():
    import multisource as M
    assert M.verdict(0.03, 0.01, 0.05) == "ACCEPT"
    assert M.verdict(0.03, -0.01, 0.07) == "REJECT"
    assert M.verdict(0.01, 0.005, 0.015) == "REJECT"
    assert M.verdict(0.03, -0.05, 0.11) == "INCONCLUSIVE"


def test_seed_spread_is_small():
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows(TAB / "per_seed.csv"):
        by[r["arm"]].append(float(r["cal_auroc_vs_active"]))
    assert set(by) == {"A", "M", "B1", "MB1"} and all(len(v) == 5 for v in by.values())
    assert all(max(v) - min(v) < 0.005 for v in by.values())


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
