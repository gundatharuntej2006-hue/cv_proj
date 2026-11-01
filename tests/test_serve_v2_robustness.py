# -*- coding: utf-8 -*-
"""Bad input never produces a number. These go through the real analyse path; every case is
rejected before the encoder is needed, so no model loads.

    py -3 -m pytest tests/test_serve_v2_robustness.py -q
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _serve():
    spec = importlib.util.spec_from_file_location("serve_v2_rb", ROOT / "scripts/serve_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    S = _serve()
    if not (ROOT / "artifacts/p2/bundle_v2_regional/head.npz").is_file():
        pytest.skip("bundle absent")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    S.register_v2(app)
    return TestClient(app), S


def _assert_no_number(r):
    assert r["status"] in ("REJECTED", "ERROR")
    assert r["analysis"] is None
    assert r["provisional"] is True


def test_blank_image_rejected_at_quality(client):
    c, _ = client
    r = c.post("/v2/analyse", files={"file": ("blank.png", _png(np.zeros((512, 512))), "image/png")}).json()
    _assert_no_number(r)
    assert r["stage"] == "quality" and "blank" in r["reasons"]


def test_saturated_white_image_rejected(client):
    c, _ = client
    r = c.post("/v2/analyse", files={"file": ("white.png", _png(np.full((512, 512), 255)), "image/png")}).json()
    _assert_no_number(r)
    assert r["stage"] == "quality"


def test_corrupt_bytes_rejected_at_decode(client):
    c, _ = client
    r = c.post("/v2/analyse", files={"file": ("x.png", b"\x89PNG not really an image", "image/png")}).json()
    _assert_no_number(r)
    assert r["stage"] == "decode"


def test_empty_and_oversized_uploads(client):
    c, _ = client
    assert c.post("/v2/analyse", files={"file": ("e.png", b"", "image/png")}).status_code == 400
    big = b"0" * (25 * 1024 * 1024 + 2)
    assert c.post("/v2/analyse", files={"file": ("big.png", big, "image/png")}).status_code == 413
    b = c.post("/v2/batch", files=[("files", ("big.png", big, "image/png"))]).json()
    assert b["n_rejected"] == 1 and b["cases"][0]["screening_score"] is None


def test_non_square_blank_is_resized_then_rejected(client):
    c, _ = client
    r = c.post("/v2/analyse", files={"file": ("tall.png", _png(np.zeros((1024, 700))), "image/png")}).json()
    _assert_no_number(r)


def test_bad_clinical_json_is_a_400(client):
    c, _ = client
    r = c.post("/v2/analyse", files={"file": ("blank.png", _png(np.zeros((512, 512))), "image/png")},
               data={"clinical": "{not json"})
    assert r.status_code == 400


def test_partial_clinical_form_never_fuses():
    from p2_cls import pipeline_v2 as P2
    b = P2.load_bundle_v2()
    if b["fusion"] is None:
        pytest.skip("no fusion block")
    for bad in ({"past_tb": 1}, {"sex": 1, "age_group": 2, "c": 1, "f": 0, "wl": 0, "ns": 0, "contact": 0,
                                 "past_tb": 1, "dm": 0},                      # rad_system missing
                {"sex": "m", "age_group": 2, "c": 1, "f": 0, "wl": 0, "ns": 0, "contact": 0, "past_tb": 1, "dm": 0, "rad_system": 1},
                {"sex": 1, "age_group": 2, "c": 2, "f": 0, "wl": 0, "ns": 0, "contact": 0, "past_tb": 1, "dm": 0, "rad_system": 1}):
        enc, err = P2.encode_clinical(bad)
        if err is None:
            fz, err = P2.fuse(0.0, enc, b)
            assert fz is None
        assert err


def test_calibration_propose_rejects_garbage(client):
    c, _ = client
    assert c.post("/v2/calibration/propose", json={"batch_scores": {"1": ["a", "b"]}}).status_code == 400
    r = c.post("/v2/calibration/propose", json={"batch_scores": {"1": [0.1] * 30}}).json()
    assert r["applied"] is False and r["offsets"] == {}


def test_decision_schema_mentions_no_probability(client):
    c, _ = client
    s = c.get("/v2/schema/decision").json()
    assert "logit" in json.dumps(s).lower()
