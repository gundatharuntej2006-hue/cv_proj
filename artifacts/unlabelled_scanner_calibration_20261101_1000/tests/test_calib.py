# -*- coding: utf-8 -*-
"""Tests for the label-free calibration run: hash, group disjointness, affine invariance,
verdict rule, label hiding, partitions.

    py -3 -m pytest artifacts/unlabelled_scanner_calibration_20261101_1000/tests -q
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

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


def test_groups_are_disjoint_and_stratified():
    import calib as C
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10 + [0] * 45 + [1] * 5)
    sc = np.array([1] * 100 + [2] * 50)
    pool, evl = C.split_groups(y, sc, rng)
    assert not set(pool) & set(evl) and len(pool) + len(evl) == 150
    assert y[pool].sum() == 7 and y[evl].sum() == 8          # floor halves per scanner x label


def test_unlabelled_arms_never_read_labels():
    import calib as C
    rng = np.random.default_rng(1)
    s = rng.normal(size=300)
    sc = np.repeat([1, 2, 3], 100)
    batches = {k: np.where(sc == k)[0][:20] for k in (1, 2, 3)}
    out_c, _ = C.correct("C", s, sc, batches, y=None)
    out_d, _ = C.correct("D", s, sc, batches, y=None)
    for k, bi in batches.items():
        assert abs(out_c[bi].mean()) < 1e-9 and abs(out_d[bi].mean()) < 1e-9
        assert abs(out_d[bi].std() - 1.0) < 1e-6


def test_within_scanner_auroc_unchanged_by_affine_arms():
    r = {(x["arm"], x["n_per_scanner"]): x for x in rows(TAB / "arms_by_size.csv")}
    for sc in (1, 2, 3):
        vals = {r[(a, "100")]["within_auroc_s%d" % sc] for a in ("A", "C", "D")}
        assert len(vals) == 1, sc


def test_verdict_follows_the_pre_registered_rule():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    rule, v = m["rule"], m["verdict"]
    for arm in ("C", "D"):
        want = (v[arm]["gain_vs_A_n100"] >= rule["min_fraction_of_B_gain"] * v[arm]["B_gain_vs_A_n100"]
                and v[arm]["worst_scanner2_sens_move"] <= rule["max_sens_move"]
                and v[arm]["worst_scanner2_burden_move"] <= rule["max_burden_move"])
        assert v[arm]["usable"] is want


def test_composition_table_complete():
    t = rows(TAB / "composition_shift_n100.csv")
    assert {(x["condition"], x["arm"]) for x in t} == {(c, a) for c in
            ("natural", "prevalence 0", "prevalence 0.30", "previous TB only", "no previous TB") for a in "ABCD"}


def test_partitions_and_no_pid():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["partitions"]["vietnam_report_used"] is False
    assert m["partitions"]["tbx11k_internal_test_used"] is False
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    pids = {str(i) for i in sp["tune"]} | {str(i) for i in sp["report"]}
    for f in list(RUN.rglob("*.csv")) + list(RUN.rglob("*.json")):
        t = f.read_text(encoding="utf-8", errors="ignore")
        assert not [p for p in pids if p in t], f.name
