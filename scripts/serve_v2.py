# -*- coding: utf-8 -*-
"""Serve the v1 API plus the v2 regional-head routes, without editing the v1 service.

    py -3 scripts/serve_v2.py                 # http://127.0.0.1:8000, docs at /docs
    py -3 scripts/serve_v2.py --preload

Added routes (everything from src/p2_serve/app.py is still there):

    GET  /v2/model                     the regional bundle: head, thresholds, fusion status
    POST /v2/overlay                   PNG: region map shaded, decision cell boxed
    POST /v2/batch                     several files -> ranked list by screening score, flags at the
                                       0.70-specificity point, decision cell per case; overlays cached
    GET  /v2/overlay/{key}             PNG overlay for a case returned by /v2/batch (memory only)
    GET  /v2/ui                        the triage screen: drop X-rays, rank the batch, see each decision cell
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
import pandas  # noqa: F401  loaded first on purpose: on Windows, pandas/pyarrow after torch segfaults
import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_V2 = {"bundle": None}
_OVERLAYS = collections.OrderedDict()          # key -> (raw bytes, analysis); bounded, memory only
_OVERLAY_CAP = 64


def _remember(raw, analysis):
    key = hashlib.sha256(raw).hexdigest()[:16]
    _OVERLAYS[key] = (raw, analysis)
    _OVERLAYS.move_to_end(key)
    while len(_OVERLAYS) > _OVERLAY_CAP:
        _OVERLAYS.popitem(last=False)
    return key


def rank_batch(results):
    """Analysed cases first, highest screening score first; rejected cases listed after, unscored."""
    ok = [r for r in results if r.get("status") == "ANALYSED"]
    bad = [r for r in results if r.get("status") != "ANALYSED"]
    ok.sort(key=lambda r: -r["analysis"]["image_logit"])
    rows = []
    for i, r in enumerate(ok, 1):
        a = r["analysis"]
        rows.append({"rank": i, "image": r["image"], "status": r["status"], "overlay_key": r.get("overlay_key"),
                     "screening_score": a["image_logit"],
                     "screen_positive_at_spec_0.70": a["operating_points"]["screen_positive_at_spec_0.70"],
                     "positive_at_spec_0.90": a["operating_points"]["positive_at_spec_0.90"],
                     "decision_cell": {"row": a["argmax_region"]["row"], "col": a["argmax_region"]["col"]},
                     "lung_fraction": a["lung_mask"]["lung_fraction"],
                     "p_fused": (a["fusion"] or {}).get("p_fused"), "latency_ms": r.get("latency_ms")})
    for r in bad:
        rows.append({"rank": None, "image": r["image"], "status": r["status"], "overlay_key": None,
                     "screening_score": None, "screen_positive_at_spec_0.70": None, "positive_at_spec_0.90": None,
                     "decision_cell": None, "lung_fraction": None, "p_fused": None,
                     "reasons": r.get("reasons", []), "latency_ms": r.get("latency_ms")})
    return {"n": len(results), "n_analysed": len(ok), "n_rejected": len(bad),
            "n_screen_positive": sum(1 for x in rows if x["screen_positive_at_spec_0.70"]),
            "ordering": "screening score, highest first; rejected cases unscored at the end",
            "cases": rows}


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

    @app.post("/v2/batch")
    def v2_batch(files: list[UploadFile] = File(...), clinical: Optional[str] = Form(None)):
        """Rank a batch by screening score. The optional clinical form applies to every image in the
        batch, so send it only for a single-image analysis."""
        if not files:
            raise HTTPException(400, "no files")
        if len(files) > 200:
            raise HTTPException(413, "at most 200 files per batch")
        clin = None
        if clinical:
            try:
                clin = json.loads(clinical)
            except json.JSONDecodeError:
                raise HTTPException(400, "clinical must be a JSON object")
            if len(files) > 1:
                raise HTTPException(400, "a clinical form applies to one image; send one file with it")
        out = []
        for f in files:
            raw = f.file.read(25 * 1024 * 1024 + 1)
            name = f.filename or "upload"
            if not raw or len(raw) > 25 * 1024 * 1024:
                out.append({"image": name, "status": "REJECTED", "stage": "decode",
                            "reasons": ["empty or oversized upload"], "analysis": None, "provisional": True})
                continue
            r = analyse_bytes_v2(raw, filename=name, clinical=clin)
            if r.get("status") == "ANALYSED":
                r["overlay_key"] = _remember(raw, r["analysis"])
            out.append(r)
        b = _bundle()
        res = rank_batch(out)
        res["model"] = {"model_id": b["meta"]["model"]["model_id"], "bundle_version": b["meta"].get("bundle_version"),
                        "thresholds_fitted_on": b["meta"]["thresholds"]["fitted_on"], "provisional": True,
                        "note": "screening score is the image logit, not a calibrated probability; not clinically validated"}
        return res

    @app.get("/v2/overlay/{key}")
    def v2_overlay_cached(key: str):
        from fastapi.responses import Response
        if key not in _OVERLAYS:
            raise HTTPException(404, "unknown or expired overlay key")
        raw, analysis = _OVERLAYS[key]
        return Response(content=overlay_png(raw, analysis), media_type="image/png")

    @app.get("/v2/ui")
    def v2_ui():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(UI_HTML)

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


UI_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TB triage screen</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--muted:#5b6470;--line:#d9dee5;--flag:#b42318;--ok:#067647;--accent:#1f5fbf}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:14px 20px;background:#fff;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0}
.sub{color:var(--muted);font-size:13px;margin-top:4px}main{display:grid;grid-template-columns:1fr 420px;gap:16px;padding:16px 20px}
@media(max-width:900px){main{grid-template-columns:1fr}}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.drop{border:2px dashed var(--line);border-radius:8px;padding:22px;text-align:center;color:var(--muted);cursor:pointer}
.drop.on{border-color:var(--accent);color:var(--accent)}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}tr.sel{background:#eef3fb}tr[data-k]{cursor:pointer}
.flag{color:var(--flag);font-weight:600}.clear{color:var(--ok)}.rej{color:var(--muted);font-style:italic}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}button:disabled{opacity:.5}
.form{display:grid;grid-template-columns:repeat(2,1fr);gap:6px 10px;margin-top:10px;font-size:13px}.form label{display:flex;justify-content:space-between;gap:6px}
img.ov{width:100%;border:1px solid var(--line);border-radius:6px;background:#000}.kv{font-size:13px;color:var(--muted);margin-top:8px}
.warn{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:8px 10px;font-size:13px;margin-top:10px}
</style></head><body>
<header><h1>TB triage screen &mdash; research system, not clinically validated</h1>
<div class="sub" id="model">loading model info&hellip;</div></header>
<main>
<section class="card">
  <div class="drop" id="drop">Drop chest X-rays here (PNG/JPG), or click to choose. Up to 200 per batch.</div>
  <input type="file" id="files" multiple accept="image/*" style="display:none">
  <div style="margin-top:10px;display:flex;gap:10px;align-items:center">
    <button id="go" disabled>Analyse batch</button><span id="count" class="kv"></span>
  </div>
  <details style="margin-top:10px"><summary style="cursor:pointer;font-size:13px">Screening questionnaire (single image only; fusion is exploratory)</summary>
  <div class="form" id="form">
    <label>sex (0/1)<input name="sex" size="2"></label><label>age_group (1/2)<input name="age_group" size="2"></label>
    <label>cough<input name="c" size="2"></label><label>fever<input name="f" size="2"></label>
    <label>weight loss<input name="wl" size="2"></label><label>night sweats<input name="ns" size="2"></label>
    <label>TB contact<input name="contact" size="2"></label><label>previous TB<input name="past_tb" size="2"></label>
    <label>diabetes (0/1/99)<input name="dm" size="2"></label><label>scanner (1/2/3)<input name="rad_system" size="2" value="3"></label>
  </div></details>
  <div class="warn">Scores are the model's screening logit at a development-fitted operating point; they are not probabilities and this screen is not a diagnostic device. Rejected images carry no score.</div>
  <div id="summary" class="kv"></div>
  <table id="tbl" style="display:none"><thead><tr><th>#</th><th>image</th><th>score</th><th>screen @0.70 spec</th><th>@0.90 spec</th><th>decision cell</th><th>lung</th><th>ms</th></tr></thead><tbody></tbody></table>
</section>
<aside class="card">
  <div id="ovtitle" class="kv">Select a case to see its decision cell.</div>
  <img class="ov" id="ov" alt="" style="display:none">
  <div class="kv">Green box = the region cell whose logit is the image logit (exact attribution at 12&times;12, &asymp;43&thinsp;px). Shading = region logits. Not a lesion boundary.</div>
</aside>
</main>
<script>
const $=s=>document.querySelector(s);let picked=[];
fetch('/v2/model').then(r=>r.json()).then(m=>{$('#model').textContent=`${m.model_id} (bundle v${m.bundle_version}) \u00b7 backbone ${m.backbone} \u00b7 thresholds: ${m.thresholds.fitted_on} \u00b7 provisional`}).catch(()=>{$('#model').textContent='model info unavailable'});
const drop=$('#drop'),inp=$('#files');drop.onclick=()=>inp.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('on')};drop.ondragleave=()=>drop.classList.remove('on');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('on');setFiles([...e.dataTransfer.files])};inp.onchange=()=>setFiles([...inp.files]);
function setFiles(fs){picked=fs.filter(f=>f.type.startsWith('image/')||/\\.(png|jpe?g|bmp|tiff?)$/i.test(f.name));$('#count').textContent=picked.length+' file(s) ready';$('#go').disabled=!picked.length}
function clinical(){const o={};let any=false;for(const i of $('#form').querySelectorAll('input')){if(i.value!==''){o[i.name]=Number(i.value);any=true}}return any?o:null}
$('#go').onclick=async()=>{const fd=new FormData();picked.forEach(f=>fd.append('files',f,f.name));const c=clinical();
 if(c){if(picked.length>1){alert('The questionnaire applies to one image; analyse a single file with it.');return}fd.append('clinical',JSON.stringify(c))}
 $('#go').disabled=true;$('#summary').textContent='analysing\u2026 first call loads the models (~40 s)';
 try{const r=await fetch('/v2/batch',{method:'POST',body:fd});if(!r.ok){throw new Error(await r.text())}render(await r.json())}
 catch(e){$('#summary').textContent='error: '+e.message}finally{$('#go').disabled=false}};
function render(res){$('#summary').textContent=`${res.n} uploaded \u00b7 ${res.n_analysed} analysed \u00b7 ${res.n_rejected} rejected \u00b7 ${res.n_screen_positive} screen-positive at 0.70 specificity \u2014 ${res.ordering}`;
 const tb=$('#tbl tbody');tb.innerHTML='';$('#tbl').style.display='';
 for(const c of res.cases){const tr=document.createElement('tr');
  if(c.status!=='ANALYSED'){tr.className='rej';tr.innerHTML=`<td>\u2013</td><td>${c.image}</td><td colspan="6">${c.status}: ${(c.reasons||[]).join('; ')}</td>`}
  else{tr.dataset.k=c.overlay_key;tr.innerHTML=`<td>${c.rank}</td><td>${c.image}</td><td>${c.screening_score.toFixed(3)}${c.p_fused!=null?' (fused '+c.p_fused.toFixed(3)+')':''}</td>
   <td class="${c['screen_positive_at_spec_0.70']?'flag':'clear'}">${c['screen_positive_at_spec_0.70']?'FLAG':'clear'}</td><td class="${c['positive_at_spec_0.90']?'flag':'clear'}">${c['positive_at_spec_0.90']?'FLAG':'clear'}</td>
   <td>r${c.decision_cell.row} c${c.decision_cell.col}</td><td>${(c.lung_fraction*100).toFixed(0)}%</td><td>${c.latency_ms??''}</td>`;
   tr.onclick=()=>{for(const x of tb.querySelectorAll('tr'))x.classList.remove('sel');tr.classList.add('sel');$('#ov').src='/v2/overlay/'+c.overlay_key+'?t='+Date.now();$('#ov').style.display='';$('#ovtitle').textContent=`#${c.rank} ${c.image} \u2014 score ${c.screening_score.toFixed(3)}, decision cell r${c.decision_cell.row} c${c.decision_cell.col}`}}
  tb.appendChild(tr)}
 const first=tb.querySelector('tr[data-k]');if(first)first.click()}
</script></body></html>"""


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
