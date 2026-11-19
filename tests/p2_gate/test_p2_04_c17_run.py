# -*- coding: utf-8 -*-
"""C17 run requirements from P1: hash check, filter to C01, equal-sets coverage, sidecar fields.

    py -3 -m pytest tests/p2_gate/test_p2_04_c17_run.py -q
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "p2_04_dedup.py"


def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p2_04_dedup_cli", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fixture(tmp_path, n_c01=40, n_extra=7, drop_from_emb=0):
    rng = np.random.default_rng(0)
    c01_ids = ["img%03d" % i for i in range(n_c01)]
    extra_ids = ["ext%03d" % i for i in range(n_extra)]
    emb_ids = c01_ids[drop_from_emb:] + extra_ids
    e = rng.normal(size=(len(emb_ids), 16)).astype(np.float32)
    e[1] = e[0]                                    # one exact duplicate pair: img000 ~ img001
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    npz = tmp_path / "c07.npz"
    np.savez(npz, image_id=np.array(emb_ids), emb=e.astype(np.float16),
             l2_normalised=np.array(True), variant=np.array("full_clahe0"))
    splits = ["train" if i % 2 == 0 else "development" for i in range(n_c01)]
    csv = tmp_path / "c01.csv"
    pd.DataFrame({"image_id": c01_ids, "split": splits,
                  "official_split": ["test"] * n_c01}).to_csv(csv, index=False)
    dec = tmp_path / "d06.json"
    json.dump({"decision_id": "D06", "value": {"tau": 0.95, "comparison": ">="},
               "locked": False, "locked_at": None, "guide_ack": False}, open(dec, "w"))
    return npz, csv, dec


def _run(tmp_path, npz, csv, dec, *extra):
    out = tmp_path / "c17.csv"
    r = subprocess.run([sys.executable, str(SCRIPT), "--emb", str(npz), "--splits", str(csv),
                        "--decision", str(dec), "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=str(ROOT))
    return r, out


def test_restricted_run_passes_and_writes_every_sidecar_field(tmp_path):
    npz, csv, dec = _fixture(tmp_path)
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    r, out = _run(tmp_path, npz, csv, dec, "--restrict-to-c01", "--c01-sha256", sha)
    assert r.returncode == 0, r.stderr + r.stdout
    meta = json.loads(Path(str(out) + ".json").read_text())
    for k in ("c01_path", "c01_sha256", "n_c01_ids", "n_embedded", "missing_embeddings",
              "extra_embeddings", "d06_file", "tau", "comparison", "d06_locked"):
        assert k in meta, k
    assert meta["c01_sha256"] == sha
    assert meta["n_c01_ids"] == meta["n_embedded"] == 40
    assert meta["missing_embeddings"] == 0 and meta["extra_embeddings"] == 0
    assert meta["n_excluded_not_in_c01"] == 7 and meta["coverage_check"] == "PASS"
    df = pd.read_csv(out)
    assert len(df) == 1 and set(df.image_id_a) | set(df.image_id_b) == {"img000", "img001"}
    assert (df.split_a != df.split_b).all()


def test_wrong_c01_hash_stops_the_run(tmp_path):
    npz, csv, dec = _fixture(tmp_path)
    r, out = _run(tmp_path, npz, csv, dec, "--restrict-to-c01", "--c01-sha256", "0" * 64)
    assert r.returncode != 0 and "SHA-256 mismatch" in (r.stderr + r.stdout)
    assert not out.exists()


def test_c01_image_without_embedding_fails_coverage(tmp_path):
    npz, csv, dec = _fixture(tmp_path, drop_from_emb=3)
    r, out = _run(tmp_path, npz, csv, dec, "--restrict-to-c01")
    assert r.returncode != 0 and "coverage check FAILED" in (r.stderr + r.stdout)
    assert not out.exists()


def test_unrestricted_run_still_refuses_images_outside_c01(tmp_path):
    npz, csv, dec = _fixture(tmp_path)
    r, _ = _run(tmp_path, npz, csv, dec)
    assert r.returncode != 0 and "--restrict-to-c01" in (r.stderr + r.stdout)


def test_decision_without_comparison_is_refused(tmp_path):
    npz, csv, dec = _fixture(tmp_path)
    json.dump({"value": {"tau": 0.98}, "locked": True}, open(dec, "w"))
    r, _ = _run(tmp_path, npz, csv, dec, "--restrict-to-c01")
    assert r.returncode != 0 and "comparison" in (r.stderr + r.stdout)


def test_restrict_and_check_rejects_duplicate_ids():
    m = _load_script()
    with pytest.raises(SystemExit):
        m.restrict_and_check(["a", "a", "b"], np.zeros((3, 2)), ["a", "b"])
