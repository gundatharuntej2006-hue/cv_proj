# -*- coding: utf-8 -*-
"""The HTTP surface the frontend reads from. What is tested here is not "does FastAPI work"
but the promises the API makes to a screen:

  - a rejected image never arrives with a probability attached;
  - every response says whether the model is provisional or a stub, so a banner can be shown;
  - one bad file in a batch returns a rejected row rather than failing the request;
  - the batch comes back in reading order, most urgent first.

The model is stubbed out with a fake bundle wherever possible so these run in milliseconds;
the two tests that need the real backbone skip themselves without one.
"""
import io

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient            # noqa: E402

from p2_serve import app as A                        # noqa: E402

BUNDLE_OK = (A.__file__ and
             __import__("pathlib").Path(A.__file__).resolve().parents[2]
             / "artifacts/p2/bundle/bundle.json").exists()
needs_bundle = pytest.mark.skipif(not BUNDLE_OK, reason="no inference bundle on disk")


def png_bytes(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def noisy(seed=0, h=512, w=512):
    return np.random.default_rng(seed).integers(40, 210, size=(h, w), dtype=np.uint8)


@pytest.fixture
def client(monkeypatch):
    """A client with the pipeline faked out: fast, and independent of the trained model."""
    fake_meta = {"model": {"model_id": "fake", "head_type": "binary"}, "variant": "full_clahe0",
                 "provisional": True, "temperature": 1.0, "t_lower": 0.3, "t_upper": 0.8,
                 "ood_threshold": 0.5, "ood_alpha": 0.05, "backbone": "fake"}
    monkeypatch.setitem(A._STATE, "bundle", {"meta": fake_meta})

    def fake_stages(img, bundle, n_lesions=0, image_label=None, t0=None):
        mean = float(np.asarray(img).mean())
        if mean < 5:                                   # stand in for the quality gate
            return {"image": image_label, "status": "REJECTED", "stage": "quality",
                    "reasons": ["blank"], "analysis": None,
                    "quality": {"pass": False, "reasons": ["blank"]},
                    "ood": None, "latency_ms": 1.0}
        return {"image": image_label, "status": "ANALYSED", "stage": "complete",
                "reasons": [], "quality": {"pass": True, "reasons": []},
                "ood": {"score": 0.1, "threshold": 0.5, "is_ood": False, "abstained": False},
                "analysis": {"p_tb_raw": 0.9, "p_tb_calibrated": 0.9, "temperature": 1.0,
                             "t_lower": 0.3, "t_upper": 0.8, "triage": "TB",
                             "pre_link": "TB", "link_fired": False,
                             "screen_positive": True},
                "latency_ms": 1.0}

    from p2_cls import pipeline as P
    monkeypatch.setattr(P, "run_stages", fake_stages)
    yield TestClient(A.create_app())
    A._STATE["bundle"] = None


# ------------------------------------------------------------------ the basics
def test_health_reports_whether_a_model_is_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_model_endpoint_exposes_provenance_and_the_provisional_flag(client):
    d = client.get("/model").json()
    assert d["provisional"] is True
    for k in ("model_id", "variant", "t_lower", "t_upper", "thresholds_fitted_on"):
        assert k in d


def test_the_decision_schema_is_published_for_the_frontend(client):
    d = client.get("/schema/decision").json()
    assert "analysis" in d and "status" in d
    assert "_null_when" in d["analysis"], "the frontend must be told when analysis is null"


# ------------------------------------------------------------------ one image
def test_an_ordinary_image_is_analysed(client):
    r = client.post("/analyse", files={"file": ("x.png", png_bytes(noisy()), "image/png")})
    d = r.json()
    assert r.status_code == 200 and d["status"] == "ANALYSED"
    assert 0.0 <= d["analysis"]["p_tb_calibrated"] <= 1.0
    assert d["analysis"]["triage"] in ("TB", "REFER", "NOT_TB")
    assert d["provisional"] is True


def test_a_rejected_image_carries_no_probability(client):
    """The invariant worth asserting twice: nothing for a clinician to act on."""
    r = client.post("/analyse",
                    files={"file": ("blank.png", png_bytes(np.zeros((512, 512), np.uint8)),
                                    "image/png")})
    d = r.json()
    assert d["status"] == "REJECTED" and d["analysis"] is None
    assert "p_tb_calibrated" not in r.text


def test_a_corrupt_file_is_a_result_not_an_error(client):
    r = client.post("/analyse", files={"file": ("bad.png", b"not an image", "image/png")})
    d = r.json()
    assert r.status_code == 200
    assert d["status"] == "REJECTED" and d["stage"] == "decode" and d["analysis"] is None


def test_an_empty_upload_is_refused(client):
    r = client.post("/analyse", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_an_oversized_upload_is_refused(client, monkeypatch):
    monkeypatch.setattr(A, "MAX_BYTES", 128)
    r = client.post("/analyse", files={"file": ("big.png", png_bytes(noisy()), "image/png")})
    assert r.status_code == 413


def test_an_unsupported_content_type_is_refused(client):
    r = client.post("/analyse", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 415


def test_every_response_states_the_provisional_and_stub_status(client):
    for name, data in (("ok.png", png_bytes(noisy(1))),
                       ("blank.png", png_bytes(np.zeros((512, 512), np.uint8))),
                       ("bad.png", b"nope")):
        d = client.post("/analyse", files={"file": (name, data, "image/png")}).json()
        assert "provisional" in d and "is_stub" in d, \
            "a screen that cannot say the model is provisional will mislead its reader"


# ------------------------------------------------------------------ batches
def test_a_batch_completes_despite_a_bad_file(client):
    files = [("files", ("a.png", png_bytes(noisy(2)), "image/png")),
             ("files", ("bad.png", b"nope", "image/png")),
             ("files", ("c.png", png_bytes(noisy(3)), "image/png"))]
    d = client.post("/analyse-batch", files=files).json()
    assert d["n"] == 3, "one bad file must not truncate a worklist"
    assert sum(r["status"] == "ANALYSED" for r in d["results"]) == 2


def test_a_batch_is_returned_in_reading_order(client):
    files = [("files", ("blank.png", png_bytes(np.zeros((512, 512), np.uint8)), "image/png")),
             ("files", ("tb.png", png_bytes(noisy(4)), "image/png"))]
    d = client.post("/analyse-batch", files=files).json()
    assert d["results"][0]["status"] == "ANALYSED"
    assert d["results"][-1]["status"] == "REJECTED", "rejected cases sort last"


def test_an_overlong_batch_is_refused(client, monkeypatch):
    monkeypatch.setattr(A, "MAX_BATCH", 2)
    files = [("files", ("%d.png" % i, png_bytes(noisy(i)), "image/png")) for i in range(3)]
    assert client.post("/analyse-batch", files=files).status_code == 413


# ------------------------------------------------------------------ guard rails
def test_a_leaked_probability_on_a_rejected_case_is_refused_not_served():
    bad = {"status": "REJECTED", "analysis": {"p_tb_calibrated": 0.99}}
    with pytest.raises(RuntimeError, match="refusing to serve"):
        A._assert_no_leaked_probability(bad)


# ------------------------------------------------------------------ the real model
@pytest.mark.slow
@needs_bundle
def test_the_real_pipeline_answers_over_http():
    A._STATE["bundle"] = None
    c = TestClient(A.create_app())
    d = c.post("/analyse", files={"file": ("x.png", png_bytes(noisy(9)), "image/png")}).json()
    assert d["status"] in ("ANALYSED", "REJECTED")
    if d["status"] == "ANALYSED":
        assert 0.0 <= d["analysis"]["p_tb_calibrated"] <= 1.0
    else:
        assert d["analysis"] is None
    A._STATE["bundle"] = None
