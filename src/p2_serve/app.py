# -*- coding: utf-8 -*-
"""A local HTTP API over the inference pipeline, for the frontend to call.

    py -3 scripts/serve.py                 # http://127.0.0.1:8000, docs at /docs

Endpoints:

    GET  /health                 is the service up, and is it running a real model?
    GET  /model                  what model, thresholds and provenance are loaded
    POST /analyse                one image (multipart file) -> one decision
    POST /analyse-batch          several images -> decisions, worklist-ordered
    GET  /schema/decision        the response shape, for the frontend to code against

Design rules, which exist because this is the surface a clinician's screen would read from:

  - **A rejected image never carries a probability.** The pipeline guarantees it; this layer
    asserts it again before responding, because a bug that leaks a number to a screen is
    worse than a 500.
  - **`provisional` and `is_stub` are always present in the response.** A demo that cannot
    tell the viewer the model is provisional is a demo that misleads them. The frontend
    should render a banner whenever either is true.
  - **A failure is a result, not an exception.** One unreadable file in a batch of forty
    returns a rejected row; it does not fail the request.
  - **Nothing is persisted.** No uploaded image is written to disk and no result is stored.
    This is a demo service on localhost, and storing patient images would need a great deal
    more than a demo service has.

It binds to 127.0.0.1 by default and has no authentication, so it must not be exposed to a
network. `scripts/serve.py --host` exists for a reason but the warning stands.
"""
import io
import time
from typing import List, Optional

import numpy as np

MAX_BYTES = 25 * 1024 * 1024          # a chest X-ray PNG is a few MB; 25 is generous
MAX_BATCH = 40
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp",
           "application/octet-stream"}

_STATE = {"bundle": None, "loaded_at": None, "n_analysed": 0}


def _bundle():
    """Loaded once on first use: the backbone load is the expensive part."""
    if _STATE["bundle"] is None:
        from p2_cls import pipeline as P
        _STATE["bundle"] = P.load_bundle()
        _STATE["loaded_at"] = time.time()
    return _STATE["bundle"]


def _decode(raw):
    """bytes -> uint8 greyscale array, or (None, reason)."""
    from PIL import Image
    try:
        a = np.array(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    except Exception as e:
        return None, "unreadable: %s" % type(e).__name__
    if min(a.shape[:2]) < 16:
        return None, "unreadable: image is %dx%d" % (a.shape[1], a.shape[0])
    return a, None


def _assert_no_leaked_probability(result):
    """The invariant worth re-checking at the boundary: a stopped case has no number."""
    if result.get("status") != "ANALYSED" and result.get("analysis") is not None:
        raise RuntimeError("a rejected case carried an analysis block; refusing to serve it")


def analyse_bytes(raw, filename="upload", n_lesions=0):
    """One image's bytes -> the decision dict the frontend renders. Never raises."""
    from p2_cls import pipeline as P
    t0 = time.perf_counter()
    meta = None
    img, err = _decode(raw)
    if err:
        return {"image": filename, "status": "REJECTED", "stage": "decode",
                "reasons": [err], "analysis": None, "quality": None, "ood": None,
                "provisional": True, "is_stub": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    b = _bundle()
    meta = b["meta"]
    try:
        if img.shape != (512, 512):
            import cv2
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        r = P.run_stages(img, b, n_lesions=n_lesions, image_label=filename)
    except Exception as e:                        # a screen must get an answer, not a 500
        return {"image": filename, "status": "ERROR", "stage": "pipeline",
                "reasons": ["%s: %s" % (type(e).__name__, e)], "analysis": None,
                "quality": None, "ood": None,
                "provisional": bool(meta.get("provisional", True)), "is_stub": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    r["image"] = filename
    r.setdefault("provisional", bool(meta.get("provisional", True)))
    r["is_stub"] = _is_stub()
    _assert_no_leaked_probability(r)
    _STATE["n_analysed"] += 1
    return r


def _is_stub():
    try:
        from p2_cls import api as cls_api
        return bool(cls_api.is_stub())
    except Exception:
        return None


def model_info():
    b = _bundle()
    m = b["meta"]
    return {
        "model_id": m["model"]["model_id"], "variant": m["variant"],
        "backbone": m.get("backbone"), "head_type": m["model"].get("head_type"),
        "provisional": bool(m.get("provisional", True)),
        "is_stub": _is_stub(),
        "temperature": m.get("temperature"),
        "temperature_fitted_on": m.get("temperature_fitted_on"),
        "t_lower": m.get("t_lower"), "t_upper": m.get("t_upper"),
        "thresholds_fitted_on": m.get("thresholds_fitted_on"),
        "ood_threshold": m.get("ood_threshold"), "ood_alpha": m.get("ood_alpha"),
        "ood_abstained": m.get("ood_threshold") is None,
        "refer_band_empty": (m.get("t_lower") is not None
                             and m.get("t_upper") is not None
                             and float(m["t_upper"]) <= float(m["t_lower"])),
        "created": m.get("created"), "note": m.get("note"),
        "dev_auc": m.get("dev_auc"),
    }


DECISION_SCHEMA = {
    "image": "str — the filename as uploaded",
    "status": "ANALYSED | REJECTED | ERROR",
    "stage": "where it stopped: decode | quality | ood | complete | pipeline",
    "reasons": "list[str] — empty when ANALYSED; C09 vocabulary for a quality rejection",
    "quality": "{pass: bool, reasons: list[str]} or null",
    "ood": "{score: float, threshold: float|null, is_ood: bool, abstained: bool} or null",
    "analysis": {
        "p_tb_raw": "float — before calibration",
        "p_tb_calibrated": "float — the number to show, 0..1",
        "temperature": "float",
        "t_lower": "float|null", "t_upper": "float|null",
        "triage": "TB | REFER | NOT_TB",
        "pre_link": "triage before the detector link rule",
        "link_fired": "bool — the rule only ever lifts NOT_TB to REFER",
        "screen_positive": "bool — TB or REFER",
        "_null_when": "status != ANALYSED. A stopped case never carries a probability.",
    },
    "provisional": "bool — true means these thresholds are not Person 4's fitted values",
    "is_stub": "bool|null — true means the head is untrained and nothing may be reported",
    "latency_ms": "float",
}


def create_app():
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="TB triage (Person 2 inference service)",
                  description=__doc__.split("Design rules")[0].strip(),
                  version="0.1.0")
    # the frontend will be served from a different port in development
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"ok": True, "model_loaded": _STATE["bundle"] is not None,
                "is_stub": _is_stub() if _STATE["bundle"] is not None else None,
                "n_analysed": _STATE["n_analysed"]}

    @app.get("/model")
    def model():
        return model_info()

    @app.get("/schema/decision")
    def schema():
        return DECISION_SCHEMA

    def _read(f: "UploadFile"):
        if f.content_type and f.content_type not in ALLOWED:
            raise HTTPException(415, "unsupported content type %r" % f.content_type)
        raw = f.file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "file larger than %d bytes" % MAX_BYTES)
        if not raw:
            raise HTTPException(400, "empty upload")
        return raw

    @app.post("/analyse")
    def analyse(file: UploadFile = File(...), n_lesions: int = 0):
        return analyse_bytes(_read(file), filename=file.filename or "upload",
                             n_lesions=n_lesions)

    @app.post("/analyse-batch")
    def analyse_batch(files: List[UploadFile] = File(...)):
        if len(files) > MAX_BATCH:
            raise HTTPException(413, "at most %d images per request" % MAX_BATCH)
        from p2_cls import pipeline as P
        out = [analyse_bytes(_read(f), filename=f.filename or "upload") for f in files]
        return {"n": len(out), "results": P.worklist(out),
                "ordering": "TB first, then REFER, then NOT_TB, rejected last"}

    return app
