# -*- coding: utf-8 -*-
"""Serve the v1 API plus the v2 regional-head routes, without editing the v1 service.

    py -3 scripts/serve_v2.py                 # http://127.0.0.1:8000, docs at /docs
    py -3 scripts/serve_v2.py --preload

Added routes (everything from src/p2_serve/app.py is still there):

    GET  /v2/model                     the regional bundle: head, thresholds, fusion status
    POST /v2/overlay                   PNG: region map shaded, decision cell boxed
    POST /v2/calibration/propose       fail-closed label-free scanner offsets (EXPERIMENTAL)
    POST /v2/analyse                   multipart `file`, optional form field `clinical` holding a
                                       JSON object with the Dryad screening-form fields
                                       {sex, age_group, c, f, wl, ns, contact, past_tb, dm,
                                        rad_system}; fusion is applied only if all are present
    GET  /v2/schema/decision           response shape

Same rules as v1: localhost only, no auth, nothing persisted, a rejected image never carries
a number, `provisional` is always present. The fusion block is EXPLORATORY and says so in
every response that uses it.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_V2 = {"bundle": None}


def _bundle():
    if _V2["bundle"] is None:
        from p2_cls import pipeline_v2 as P2
        # P2_BUNDLE selects a bundle directory (default: the v2 regional bundle); candidate v3
        # lives in artifacts/p2/bundle_v3_candidate and is not the default
        _V2["bundle"] = P2.load_bundle_v2(os.environ.get("P2_BUNDLE") or P2.BUNDLE_V2)
    return _V2["bundle"]


def analyse_bytes_v2(raw, filename="upload", clinical=None):
    import numpy as np
    from PIL import Image
    from p2_cls import pipeline_v2 as P2
    t0 = time.perf_counter()
    try:
        img = np.array(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    except Exception as e:
        return {"image": filename, "status": "REJECTED", "stage": "decode",
                "reasons": ["unreadable: %s" % type(e).__name__], "analysis": None,
                "provisional": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if img.shape != (512, 512):
        import cv2
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    try:
        r = P2.run_stages_v2(img, _bundle(), clinical=clinical, image_label=filename, t0=t0)
    except Exception as e:
        return {"image": filename, "status": "ERROR", "stage": "pipeline",
                "reasons": ["%s: %s" % (type(e).__name__, e)], "analysis": None,
                "provisional": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if r.get("status") != "ANALYSED" and r.get("analysis") is not None:
        raise RuntimeError("a rejected case carried an analysis block")
    return r


DECISION_SCHEMA_V2 = {
    "status": "ANALYSED | REJECTED | ERROR",
    "analysis.image_logit": "float - the regional head's score; higher = more TB-like",
    "analysis.operating_points": "screen_positive_at_spec_0.70 and positive_at_spec_0.90, both fitted on TBX11K development",
    "analysis.region_logits": "12x12 list - per-region logits, seed mean; max over it is the image logit",
    "analysis.argmax_region": "{row, col, logit} - the decision region, exactly",
    "analysis.fusion": "null, or {p_fused, logit, status: EXPLORATORY...} when the full clinical form was supplied, or {status: 'not applied', reason}",
    "provisional": "always true for this bundle",
}


def register_v2(app):
    from fastapi import File, Form, UploadFile, HTTPException
    from typing import Optional

    @app.get("/v2/model")
    def v2_model():
        b = _bundle()
        m = b["meta"]
        return {"model_id": m["model"]["model_id"], "bundle_version": m.get("bundle_version"),
                "backbone": m.get("backbone"), "trainable_parameters": m["model"].get("trainable_parameters"),
                "dev_auc": m.get("dev_auc"), "thresholds": m.get("thresholds"),
                "known_external": m.get("known_external"), "attribution": m.get("attribution"),
                "fusion": {"available": b["fusion"] is not None,
                           "status": b["fusion"]["status"] if b["fusion"] else None,
                           "cv_result": b["fusion"]["cv_result"] if b["fusion"] else None},
                "provisional": m.get("provisional", True), "note": m.get("note")}

    @app.get("/v2/schema/decision")
    def v2_schema():
        return DECISION_SCHEMA_V2

    @app.post("/v2/analyse")
    def v2_analyse(file: UploadFile = File(...), clinical: Optional[str] = Form(None)):
        raw = file.file.read(25 * 1024 * 1024 + 1)
        if not raw:
            raise HTTPException(400, "empty upload")
        if len(raw) > 25 * 1024 * 1024:
            raise HTTPException(413, "file too large")
        clin = None
        if clinical:
            try:
                clin = json.loads(clinical)
            except json.JSONDecodeError:
                raise HTTPException(400, "clinical must be a JSON object")
        return analyse_bytes_v2(raw, filename=file.filename or "upload", clinical=clin)

    @app.post("/v2/overlay")
    def v2_overlay(file: UploadFile = File(...)):
        """The analysed image as PNG with the 12x12 region map shaded and the decision cell boxed.
        The box is the arg-max region cell - the region whose logit IS the image logit - at
        region-cell resolution (about 43 px), not a lesion boundary."""
        from fastapi.responses import Response
        raw = file.file.read(25 * 1024 * 1024 + 1)
        if not raw or len(raw) > 25 * 1024 * 1024:
            raise HTTPException(400, "empty or oversized upload")
        r = analyse_bytes_v2(raw, filename=file.filename or "upload")
        if r.get("status") != "ANALYSED":
            raise HTTPException(422, "not analysed: %s" % "; ".join(r.get("reasons", [])))
        return Response(content=overlay_png(raw, r["analysis"]), media_type="image/png")

    @app.post("/v2/calibration/propose")
    def v2_calibration_propose(payload: dict):
        """Propose per-scanner offsets from UNLABELLED local image logits. Fails closed: refuses
        without a declared site reference, below 25 images per scanner, or when the batch's
        questionnaire composition is off. Status: EXPERIMENTAL, not prospectively validated (R25).
        Body: {"batch_scores": {scanner: [logits]}, "batch_clinical": {scanner: [encoded rows]},
               "site_reference": {field: rate}}"""
        from p2_cls import site_calibration as SC
        try:
            scores = {str(k): list(map(float, v)) for k, v in payload.get("batch_scores", {}).items()}
        except (TypeError, ValueError):
            raise HTTPException(400, "batch_scores must map scanner -> list of numbers")
        clin = payload.get("batch_clinical")
        clin = {str(k): v for k, v in clin.items()} if isinstance(clin, dict) else None
        return SC.propose(scores, clin, site_reference=payload.get("site_reference"))


def overlay_png(raw, analysis):
    """PNG bytes: greyscale image, region logits as a translucent 12x12 heat layer, decision cell boxed."""
    import numpy as np
    from PIL import Image, ImageDraw
    img = Image.open(io.BytesIO(raw)).convert("L").resize((512, 512))
    base = img.convert("RGB")
    rl = np.array(analysis["region_logits"], dtype=float)                 # 12x12
    lo, hi = float(rl.min()), float(rl.max())
    norm = (rl - lo) / (hi - lo + 1e-9)
    heat = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    hd = ImageDraw.Draw(heat)
    cell = 512 / 12.0
    for r_ in range(12):
        for c_ in range(12):
            a = int(110 * norm[r_, c_])
            hd.rectangle([c_ * cell, r_ * cell, (c_ + 1) * cell, (r_ + 1) * cell], fill=(255, 64, 0, a))
    out = Image.alpha_composite(base.convert("RGBA"), heat)
    d = ImageDraw.Draw(out)
    am = analysis["argmax_region"]
    x0, y0 = am["col"] * cell, am["row"] * cell
    d.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(0, 255, 0, 255), width=3)
    d.text((6, 6), "logit %.2f  decision cell (r%d,c%d) %.2f" % (
        analysis["image_logit"], am["row"], am["col"], am["logit"]), fill=(255, 255, 255, 255))
    buf = io.BytesIO()
    out.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--bundle", default=None, help="bundle directory (overrides P2_BUNDLE)")
    a = ap.parse_args()
    if a.bundle:
        os.environ["P2_BUNDLE"] = a.bundle
    from p2_serve.app import create_app
    app = create_app()
    register_v2(app)
    if a.preload:
        _bundle()
        from p2_cls import pipeline_v2 as P2
        P2._embedder(_V2["bundle"]); P2._segmenter(_V2["bundle"])
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
