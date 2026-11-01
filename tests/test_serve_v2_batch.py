# -*- coding: utf-8 -*-
"""Batch ranking, overlay cache and the triage screen, with the analysis function stubbed so no
model loads. Ordering, rejected handling, the one-form-one-image rule and the overlay route.

    py -3 -m pytest tests/test_serve_v2_batch.py -q
"""
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_serve():
    spec = importlib.util.spec_from_file_location("serve_v2", ROOT / "scripts/serve_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _png(seed):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_analysis(score, row=3, col=7, fused=None):
    rl = np.full((12, 12), score - 3.0)
    rl[row, col] = score
    return {"image_logit": score,
            "operating_points": {"screen_positive_at_spec_0.70": score >= -3.3368, "positive_at_spec_0.90": score >= -0.9052,
                                 "thresholds_fitted_on": "TBX11K development", "logit_at_spec_0.70": -3.3368, "logit_at_spec_0.90": -0.9052},
            "region_logits": rl.round(4).tolist(), "argmax_region": {"row": row, "col": col, "logit": score},
            "per_seed_image_logit": [score] * 5, "lung_mask": {"lung_fraction": 0.31, "masked": True},
            "fusion": ({"p_fused": fused, "logit": 0.0, "status": "EXPLORATORY"} if fused is not None else None)}


@pytest.fixture
def client(monkeypatch):
    S = _load_serve()
    scores = {"a.png": -5.0, "b.png": 1.2, "c.png": -2.0}

    def fake_analyse(raw, filename="upload", clinical=None):
        if filename == "bad.png":
            return {"image": filename, "status": "REJECTED", "stage": "quality", "reasons": ["blank"], "analysis": None, "provisional": True}
        return {"image": filename, "status": "ANALYSED", "stage": "complete", "reasons": [], "provisional": True,
                "analysis": _fake_analysis(scores.get(filename, 0.0), fused=(0.42 if clinical else None)), "latency_ms": 12.3}

    monkeypatch.setattr(S, "analyse_bytes_v2", fake_analyse)
    S._V2["bundle"] = {"meta": {"model": {"model_id": "stub"}, "bundle_version": 2,
                                "thresholds": {"fitted_on": "TBX11K development"}}, "fusion": None}
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    S.register_v2(app)
    return TestClient(app), S


def test_batch_ranks_highest_first_and_lists_rejected_last(client):
    c, S = client
    files = [("files", (n, _png(i), "image/png")) for i, n in enumerate(["a.png", "b.png", "bad.png", "c.png"])]
    r = c.post("/v2/batch", files=files)
    assert r.status_code == 200, r.text
    res = r.json()
    assert (res["n"], res["n_analysed"], res["n_rejected"]) == (4, 3, 1)
    assert [x["image"] for x in res["cases"]] == ["b.png", "c.png", "a.png", "bad.png"]
    assert [x["rank"] for x in res["cases"]] == [1, 2, 3, None]
    assert res["cases"][0]["screen_positive_at_spec_0.70"] is True and res["cases"][2]["screen_positive_at_spec_0.70"] is False
    assert res["n_screen_positive"] == 2
    assert res["cases"][3]["screening_score"] is None and res["cases"][3]["reasons"] == ["blank"]
    assert "not a calibrated probability" in res["model"]["note"]


def test_overlay_key_serves_png(client):
    c, S = client
    r = c.post("/v2/batch", files=[("files", ("b.png", _png(1), "image/png"))])
    key = r.json()["cases"][0]["overlay_key"]
    o = c.get("/v2/overlay/%s" % key)
    assert o.status_code == 200 and o.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(o.content))
    assert im.size == (512, 512)
    assert c.get("/v2/overlay/doesnotexist").status_code == 404


def test_clinical_form_only_with_one_image(client):
    c, S = client
    two = [("files", ("a.png", _png(1), "image/png")), ("files", ("b.png", _png(2), "image/png"))]
    assert c.post("/v2/batch", files=two, data={"clinical": "{\"sex\": 1}"}).status_code == 400
    one = c.post("/v2/batch", files=two[:1], data={"clinical": "{\"sex\": 1}"})
    assert one.status_code == 200 and one.json()["cases"][0]["p_fused"] == 0.42


def test_overlay_cache_is_bounded(client):
    c, S = client
    S._OVERLAYS.clear()
    for i in range(S._OVERLAY_CAP + 5):
        S._remember(b"x%d" % i, _fake_analysis(0.0))
    assert len(S._OVERLAYS) == S._OVERLAY_CAP


def test_ui_page_served(client):
    c, S = client
    r = c.get("/v2/ui")
    assert r.status_code == 200 and "TB triage screen" in r.text and "not clinically validated" in r.text
