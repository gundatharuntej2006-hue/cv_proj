# -*- coding: utf-8 -*-
"""Tests for the fusion-on-X run: hash, gates, verdict rule, fingerprint, partitions.

    py -3 -m pytest artifacts/fusion_on_stacked_head_20261101_0800/tests -q
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_protocol_hash_unchanged():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    assert h == (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]


def test_all_four_gates_passed():
    g = rows(TAB / "reproduction_gate.csv")
    assert len(g) == 4 and all(r["pass"] == "True" for r in g)
    by = {r["check"]: float(r["here"]) for r in g}
    assert abs(by["X_noNLM raw TUNE AUROC"] - 0.7585) < 1e-3
    assert abs(by["ours M2 (R16)"] - 0.7809) < 1e-3


def test_cohort_fingerprint_is_the_pinned_one():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    pinned = json.loads((ROOT / "artifacts/kaggle_exec_prep_20261023_1130/configs/tune_cohort_fingerprint.json")
                        .read_text(encoding="utf-8"))["fingerprint_sha256"]
    assert m["cohort_fingerprint"] == pinned and m["n"] == 1352 and m["positives"] == 139


def test_primary_verdict_follows_the_rule():
    p = [r for r in rows(TAB / "paired_contrasts.csv") if r["primary"] == "True"]
    assert len(p) == 1 and p[0]["contrast"] == "ours_X M2 - ours M2"
    lo, hi = [float(v) for v in p[0]["ci"].strip("[]").split(",")]
    d = float(p[0]["delta"])
    want = "INCONCLUSIVE" if (hi - lo) / 2 > 0.05 else ("ACCEPT" if (d >= 0.02 and lo > 0) else "REJECT")
    assert p[0]["verdict"] == want


def test_fusion_models_are_m0_and_m2_only():
    r = rows(TAB / "fusion_results.csv")
    assert {x["model"] for x in r} == {"M0", "M2"} and {x["image_score"] for x in r} == {"ours", "ours_X", "lunit"}
    assert all(x["n_repeats"] == "20" and x["folds"] == "10" for x in r)


def test_head_file_has_five_seeds():
    with np.load(TAB / "X_noNLM_head.npz") as z:
        assert z["W"].shape == (5, 1, 768) and z["b"].shape == (5, 1)


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
