# -*- coding: utf-8 -*-
"""Tests for the box-supervised run: geometry, labels, guards, statistics.

    py -3 -m pytest artifacts/box_supervised_head_20261101_0400/tests -q
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


def test_bins_match_adaptive_avg_pool2d():
    """The hand-written 37->12 bins must equal what torch's adaptive pooling uses."""
    import torch
    import box_supervised as B
    x = torch.zeros(1, 1, 37, 37)
    for i in range(37):
        x[0, 0, i, :] = i                    # row index as value
    y = torch.nn.functional.adaptive_avg_pool2d(x, 12)[0, 0, :, 0]
    for r, (s, e) in enumerate(B.BINS):
        assert abs(float(y[r]) - np.mean(range(s, e))) < 1e-5, (r, s, e)


def test_box_to_region_mapping_edges():
    import box_supervised as B
    # a box covering the whole image touches every region
    assert B.box_to_regions([0, 0, 512, 512], 512, 512) == set(range(144))
    # a tiny box at the origin lands only in region (0,0)
    assert B.box_to_regions([0, 0, 5, 5], 512, 512) == {0}
    # a tiny box at the far corner lands only in region (11,11)
    assert B.box_to_regions([507, 507, 5, 5], 512, 512) == {143}
    # non-512 originals are refused rather than silently scaled
    with pytest.raises(SystemExit):
        B.box_to_regions([0, 0, 5, 5], 1024, 1024)


def test_region_labels_follow_the_protocol_table():
    import box_supervised as B
    boxes = {"tb_a": [(1, [0, 0, 40, 40], 512, 512)],                 # active only
             "tb_o": [(2, [0, 0, 40, 40], 512, 512)],                 # obsolete only
             "tb_m": [(1, [0, 0, 40, 40], 512, 512), (2, [470, 470, 40, 40], 512, 512)]}
    Y, n_act, n_obs = B.region_labels(["h", "tb_a", "tb_o", "tb_m", "tb_none"],
                                      ["healthy", "active_tb", "latent_tb", "active_latent_tb", "active_tb"], boxes)
    assert (Y[0] == 0).all()                                   # healthy: all 0
    assert Y[1, 0] == 1 and np.isnan(Y[1, 143])                # active box -> +1; rest masked
    assert Y[2, 0] == 0 and np.isnan(Y[2, 143])                # obsolete-only -> 0; rest masked
    assert Y[3, 0] == 1 and Y[3, 143] == 0                     # mixed: active +1, obsolete 0
    assert np.isnan(Y[4]).all()                                # TB without boxes: all masked


def test_category_3_is_refused():
    import box_supervised as B
    with pytest.raises(SystemExit):
        B.region_labels(["x"], ["active_tb"], {"x": [(3, [0, 0, 5, 5], 512, 512)]})


def test_trainer_with_lambda_zero_reproduced_production():
    g = {r["arm"]: r for r in rows(TAB / "reproduction_gate.csv")}
    assert g["A"]["pass"] == "True" and g["B1"]["pass"] == "True"
    assert abs(float(g["A"]["here"]) - 0.9498) < 5e-4


def test_primary_verdict_follows_the_rule():
    p = [r for r in rows(TAB / "paired_differences.csv") if r["primary"] == "True"][0]
    lo, hi = [float(v) for v in p["cal_ci"].strip("[]").split(",")]
    hw = (hi - lo) / 2
    d = float(p["cal_delta"])
    want = "INCONCLUSIVE" if hw > 0.05 else ("ACCEPT" if (d >= 0.02 and lo > 0) else "REJECT")
    assert p["verdict"] == want


def test_localisation_baselines_are_below_the_model():
    L = {(r["arm"], r["metric"]): float(r["hit_rate"]) for r in rows(TAB / "localisation.csv")
         if r["hit_rate"]}
    assert L[("A", "top-1 hit")] > L[("baseline: random lung region", "top-1 hit")]
    assert L[("baseline: random lung region", "top-1 hit")] > L[("baseline: random region", "top-1 hit")]
    assert L[("A", "top-1 hit")] >= 0.9


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
