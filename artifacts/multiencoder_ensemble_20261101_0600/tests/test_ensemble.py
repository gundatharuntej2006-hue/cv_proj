# -*- coding: utf-8 -*-
"""Tests for the multi-encoder run: guards, gate, standardisation, statistics, partitions.

    py -3 -m pytest artifacts/multiencoder_ensemble_20261101_0600/tests -q
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


def test_encoders_pinned_in_manifest():
    m = json.loads((TAB / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["encoders"]["D2"]["revision"].startswith("a6132efb")
    assert m["encoders"]["S"]["revision"].startswith("e2ad64ee")
    assert m["rad_dino_revision"].startswith("110cbc18")


def test_caches_cover_only_allowed_partitions():
    import pandas as pd
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    forbidden = set(c01[c01.split == "internal_test"].image_id)
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    report = {str(i) for i in sp["report"]}
    want = {"tbx_train": 3241, "tbx_dev": 659, "tbx_cal": 899, "vie_tune": 1352}
    for key in ("D2", "S"):
        for name, n in want.items():
            p = RUN / "cache" / ("regional_%s_%s_lungs_g12.npz" % (key, name))
            if not p.is_file():
                pytest.skip("cache absent")
            with np.load(p) as z:
                ids = set(z["image_id"].tolist())
                assert z["reg"].shape == (n, 144, 768), (key, name)
            assert len(ids) == n and not (ids & forbidden) and not (ids & report), (key, name)


def test_region_vectors_are_unit_norm():
    p = RUN / "cache" / "regional_S_tbx_dev_lungs_g12.npz"
    if not p.is_file():
        pytest.skip("cache absent")
    with np.load(p) as z:
        X = z["reg"][:16].astype(np.float32)
    assert np.allclose(np.linalg.norm(X, axis=2), 1.0, atol=5e-3)


def test_calibration_masking_rule_matches_r12():
    """Rebuild one calibration PNG with the R12 rule and compare to what the run used."""
    import ensemble as E
    from PIL import Image
    import pandas as pd
    c01 = pd.read_csv(E.C01)
    stem = c01[c01.split == "calibration"].image_id.iloc[0]
    got = E.lungs_png(stem)
    a = np.array(Image.open(E.TBX_RAW / (stem + ".png")).convert("L"), np.uint8)
    m = np.array(Image.open(E.MASKS / (stem + ".png")).convert("L"), np.uint8) > 127
    o = a.astype(np.float32)
    o[~m] = float(o.mean())
    ref = np.clip(o, 0, 255).astype(np.uint8)
    assert np.array_equal(np.array(Image.open(got).convert("L"), np.uint8), ref)


def test_reproduction_gate_passed():
    g = rows(TAB / "reproduction_gate.csv")[0]
    assert g["arm"] == "D" and g["pass"] == "True" and abs(float(g["here"]) - 0.9498) < 5e-4


def test_standardisation_from_development_only():
    s = json.loads((TAB / "standardisation.json").read_text(encoding="utf-8"))
    assert set(s) == {"D", "D2", "S"}
    assert all(v["dev_sd"] > 0 for v in s.values())


def test_primary_verdict_follows_the_rule():
    import ensemble as E
    p = [r for r in rows(TAB / "paired_differences.csv") if r["primary"] == "True"]
    assert len(p) == 1 and p[0]["contrast"] == "E3 - D"
    lo, hi = [float(v) for v in p[0]["cal_ci"].strip("[]").split(",")]
    assert p[0]["verdict"] == E.verdict(float(p[0]["cal_delta"]), lo, hi)


def test_verdict_rule():
    import ensemble as E
    assert E.verdict(0.03, 0.01, 0.05) == "ACCEPT"
    assert E.verdict(0.03, -0.01, 0.07) == "REJECT"
    assert E.verdict(0.01, 0.005, 0.015) == "REJECT"
    assert E.verdict(0.03, -0.05, 0.11) == "INCONCLUSIVE"


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
