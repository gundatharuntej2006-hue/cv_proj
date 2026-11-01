# -*- coding: utf-8 -*-
"""Tests for the clinical-field ablation: hash, gates, arm definitions, carry rule, partitions.

    py -3 -m pytest artifacts/clinical_field_ablation_20261101_0900/tests -q
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

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


def test_gates_passed():
    g = rows(TAB / "reproduction_gate.csv")
    assert len(g) == 3 and all(r["pass"] == "True" for r in g)


def test_arm_definitions_follow_the_protocol():
    import ablation as A
    arms = A.arms()
    assert len(arms) == 1 + 8 + 2 + 5
    assert arms["without dm"] == [c for c in A.CLIN if c not in ("dm", "dm_unknown")]
    assert arms["without scanner"] == [c for c in A.CLIN if c not in ("scan1", "scan2")]
    assert arms["questionnaire only"] == arms["without scanner"]
    assert arms["image + past_tb"] == ["past_tb"]


def test_questionnaire_only_equals_without_scanner():
    r = {x["arm"]: x for x in rows(TAB / "ablation_results.csv")}
    assert r["questionnaire only"]["auroc_mean"] == r["without scanner"]["auroc_mean"]


def test_carry_flag_follows_the_rule():
    for r in rows(TAB / "paired_vs_full.csv"):
        if not r["contrast"].startswith("without"):
            continue
        lo, hi = [float(v) for v in r["ci"].strip("[]").split(",")]
        want = float(r["delta"]) <= -0.01 and hi < 0
        assert r["field_carries_gain"] == str(want), r["contrast"]


def test_only_scanner_carries():
    carry = [r["contrast"] for r in rows(TAB / "paired_vs_full.csv") if r["field_carries_gain"] == "True"]
    assert carry == ["without scanner - full"]


def test_partitions_and_no_pid():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["partitions"]["vietnam_report_used"] is False
    assert m["partitions"]["tbx11k_internal_test_used"] is False
    assert m["n"] == 1352 and m["positives"] == 139
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    pids = {str(i) for i in sp["tune"]} | {str(i) for i in sp["report"]}
    for f in RUN.rglob("*.csv"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        assert not [p for p in pids if p in t], f.name
