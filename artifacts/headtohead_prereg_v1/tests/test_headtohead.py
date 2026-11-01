# -*- coding: utf-8 -*-
"""The head-to-head scorer: protocol hash guard, manifest validation, decision rule, cluster
resampling, and the one-shot lock — all on synthetic data, no model loaded.

    py -3 -m pytest artifacts/headtohead_prereg_v1/tests -q
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
import headtohead as H                                               # noqa: E402


def test_protocol_hash_matches_recorded():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    rec = RUN / "protocol_hash.txt"
    if not rec.is_file():
        pytest.skip("protocol not yet hashed")
    assert rec.read_text(encoding="utf-8").split()[0] == h


def test_decision_rule():
    assert H.decision(0.01).startswith("SUPERIOR")
    assert H.decision(-0.01).startswith("NON-INFERIOR")
    assert H.decision(-0.031).startswith("NOT NON-INFERIOR")
    assert H.NI_MARGIN == 0.03 and H.MIN_CASES == 400


def test_manifest_validation_rejects_bad_files(tmp_path):
    p = tmp_path / "c.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pid", "image", "xpert_result"])
        w.writeheader()
        w.writerow({"pid": "a", "image": "a.png", "xpert_result": "1"})
    with pytest.raises(SystemExit):
        H.load_rows(p)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=H.REQUIRED)
        w.writeheader()
        w.writerow(dict(zip(H.REQUIRED, ["a", "a.png", "2", "0", "s1", "0.3", "v1"])))
    with pytest.raises(SystemExit):
        H.load_rows(p)


def test_paired_bootstrap_units():
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 40)
    pa = rng.normal(size=240) + y
    pb = pa + rng.normal(scale=0.1, size=240)
    d, lo, hi, unit = H.paired(y, pa, pb, np.arange(240), np.random.default_rng(0), n=200)
    assert unit == "participant" and lo <= d <= hi
    clusters = np.repeat(np.arange(24), 10)
    d2, lo2, hi2, unit2 = H.paired(y, pa, pb, clusters, np.random.default_rng(0), n=200)
    assert unit2 == "cluster" and lo2 <= d2 <= hi2


def test_lock_refuses_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "LOCK", tmp_path / "LOCK")
    H.LOCK.write_text("scoring started", encoding="utf-8")
    monkeypatch.setattr(H, "MANIFEST", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["headtohead.py", "--run"])
    monkeypatch.setattr(H, "TAB", tmp_path / "tables")
    with pytest.raises(SystemExit) as e:
        H.main()
    assert "manifest" in str(e.value) or "LOCK" in str(e.value)


def test_no_training_code_path():
    src = (RUN / "scripts/headtohead.py").read_text(encoding="utf-8")
    for forbidden in ("optim.", "backward(", "LogisticRegression(", ".fit("):
        assert forbidden not in src, forbidden
