"""In-memory PDF downloads for P2's logit-based v2 API (separate from C14/C15).

No probability/category is invented. Cached downloads reuse the original result,
including clinical fusion, without repeating model inference. Original filenames,
questionnaire answers and file metadata are excluded from the PDF/cache result.
"""
from __future__ import annotations

import copy
import io
import json
import math
import secrets
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from html import escape

from PIL import Image as PILImage


class ReportCache:
    def __init__(self, capacity=200):
        if capacity < 1:
            raise ValueError("Cache capacity must be positive")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def remember(self, raw, result, metadata):
        key = secrets.token_hex(16)
        clean = copy.deepcopy(result)
        clean.pop("image", None)
        clean.pop("reasons", None)  # P2 error text can contain paths/identifiers.
        clean.pop("report_key", None)
        clean.pop("overlay_key", None)
        original = None
        if result.get("status") == "ANALYSED":
            im = PILImage.open(io.BytesIO(raw)).convert("L")
            im = im.resize((512, 512), PILImage.Resampling.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            original = buf.getvalue()
        with self._lock:
            self._entries[key] = (original, clean, copy.deepcopy(metadata))
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return key

    def get(self, key):
        with self._lock:
            raw, result, meta = self._entries[key]
            return raw, copy.deepcopy(result), copy.deepcopy(meta)


def build_pdf(result, *, case_id, metadata, original=None, overlay=None):
    """Return a PDF for a v2 result. No C14 schema or calibrated-probability claim."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from .pdf_report import tbcore_texts

    disclaimer, scope = tbcore_texts()
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 11
    styles["BodyText"].spaceBefore = 0
    styles["BodyText"].spaceAfter = 0
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36,
                            topMargin=30, bottomMargin=30, title="TB screening research report",
                            author="TB screening project")
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 2))

    paragraph("TB screening research report", "Title")
    paragraph(f"Case {case_id} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    paragraph(f"Status: {result['status']} | Provisional research output", "Heading2")
    paragraph("Screening scores are logits, not calibrated probabilities. This report does not assign "
              "a calibrated TB / REFER / NOT TB category.")
    for key in ("model_id", "bundle_version", "backbone", "git_sha", "decisions_hash", "thresholds_fitted_on"):
        if metadata.get(key) is not None:
            value = json.dumps(metadata[key], ensure_ascii=True) if isinstance(metadata[key], (dict, list)) else str(metadata[key])
            paragraph(f"{key}: {value}")
    analysis = result.get("analysis")
    if result.get("status") == "ANALYSED":
        if not isinstance(analysis, dict):
            raise ValueError("Analysed report requires analysis")
        score = float(analysis["image_logit"])
        if not math.isfinite(score):
            raise ValueError("Screening logit must be finite")
        paragraph(f"Screening logit: {score:.4f}", "Heading2")
        for key, label in (("screen_positive_at_spec_0.70", "Development specificity 0.70"),
                           ("positive_at_spec_0.90", "Development specificity 0.90")):
            flag = analysis["operating_points"][key]
            if not isinstance(flag, bool):
                raise ValueError("Operating-point flags must be booleans")
            paragraph(f"{label}: {'FLAG' if flag else 'clear'}")
        pics = []
        for raw in (original, overlay):
            if raw:
                image = Image(io.BytesIO(raw))
                scale = min(210 / image.imageWidth, 190 / image.imageHeight)
                image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
                pics.append(image)
        if pics:
            table = Table([pics], colWidths=[250] * len(pics))
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(table)
        paragraph("Attention evidence, not a finding. Region shading is not a lesion boundary.")
        evidence = analysis.get("detector") or {}
        if evidence.get("status") == "available":
            paragraph(f"Lesion detector: {evidence['n_lesions']} box(es), threshold {evidence['conf_threshold']}")
            paragraph(f"Detector model: {evidence['model_id']}")
            paragraph(f"Checkpoint SHA-256: {evidence.get('checkpoint_sha256') or 'not supplied'}")
            boxes = evidence.get("boxes", [])
            paragraph("Box scores: " + (", ".join(f"{b['score']:.3f}" for b in boxes[:12]) or "none"))
            if len(boxes) > 12:
                paragraph(f"{len(boxes) - 12} further boxes shown on overlay.")
        else:
            paragraph("Lesion detector unavailable. This does not mean zero lesions.")
        referral = analysis.get("detector_referral") or {}
        paragraph("Additional detector review: " + ("unavailable" if not referral.get("available") else
                  "requested" if referral.get("additional_review") else "not requested"))
        agreement = analysis.get("detector_region_agreement") or {}
        paragraph("Predicted boxes overlap displayed region: " +
                  (str(agreement.get("any_overlap")) if agreement.get("status") == "available" else "unavailable"))
        paragraph("This overlap is exploratory agreement, not lesion accuracy or the published 95.9% result.")
        fusion = analysis.get("fusion") or {}
        paragraph("Clinical fusion: " + ("EXPLORATORY; not a validated TB probability" if fusion.get("p_fused") is not None
                                         else "not applied"))
        if result.get("latency_ms") is not None:
            paragraph(f"Analysis latency: {result['latency_ms']} ms (PDF generation excluded)")
    else:
        if analysis is not None:
            raise ValueError("Rejected/error reports must not contain analysis")
        paragraph("No model score or lesion analysis. Send for manual review.")
    paragraph(disclaimer, "Heading3")
    paragraph(scope)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 18, f"Research use | {case_id} | Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


UI_EXTENSION = r"""
<script>
const p3RenderOriginal=render;
render=function(res){
  // Upstream renderer inserts names/reasons with innerHTML. Escape those fields.
  const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  p3RenderOriginal({...res,cases:res.cases.map(c=>({...c,image:safe(c.image),reasons:(c.reasons||[]).map(safe)}))});
  const head=document.querySelector('#tbl thead tr');
  if(!head.querySelector('[data-p3]')){const th=document.createElement('th');th.dataset.p3='1';th.textContent='Lesions / PDF';head.appendChild(th)}
  const rows=document.querySelectorAll('#tbl tbody tr');
  res.cases.forEach((c,i)=>{const td=document.createElement('td');const d=c.detector;
    td.appendChild(document.createTextNode(d&&d.status==='available'?`${d.n_lesions} lesion(s) `:'detector unavailable '));
    if(c.detector_referral&&c.detector_referral.additional_review)td.appendChild(document.createTextNode('- REVIEW '));
    if(c.report_key){const a=document.createElement('a');a.href='/v2/report/'+encodeURIComponent(c.report_key);a.textContent='Download PDF';a.target='_blank';a.rel='noopener';a.onclick=e=>e.stopPropagation();td.appendChild(a)}
    rows[i].appendChild(td);
  });
};
document.querySelector('aside .kv:last-child').textContent='Green: displayed seed-mean region. Cyan: predicted lesion boxes with confidence scores. Attention evidence, not a finding.';
</script>
"""


def install(service):
    """Wrap only the P3-bound service instance; do not edit Atul's source files."""
    from fastapi import File, Form, HTTPException, UploadFile
    from fastapi.responses import Response

    cache = ReportCache()
    service.p3_reports = cache
    original_analyse, original_rank, original_register = service.analyse_bytes_v2, service.rank_batch, service.register_v2

    def metadata():
        try:
            meta = service._bundle()["meta"]
        except Exception:
            # A corrupt/rejected upload must still produce a manual-review PDF
            # when no classifier bundle is installed. Do not expose error paths.
            meta = {}
        from ._tbcore_adapter import version_info
        versions = version_info()
        return {"model_id": meta.get("model", {}).get("model_id"), "bundle_version": meta.get("bundle_version"),
                "backbone": meta.get("backbone"), "thresholds_fitted_on": meta.get("thresholds", {}).get("fitted_on"),
                "git_sha": versions.get("git_sha"), "decisions_hash": versions.get("decisions_hash")}

    def analyse(raw, filename="upload", clinical=None):
        result = original_analyse(raw, filename=filename, clinical=clinical)
        result = copy.deepcopy(result)
        result["report_key"] = cache.remember(raw, result, metadata())
        return result

    def rank(results):
        ranked = original_rank(results)
        ordered = sorted((r for r in results if r.get("status") == "ANALYSED"), key=lambda r: -r["analysis"]["image_logit"])
        ordered += [r for r in results if r.get("status") != "ANALYSED"]
        for row, source in zip(ranked["cases"], ordered):
            row["report_key"] = source.get("report_key")
        return ranked

    def pdf(key):
        try:
            raw, result, meta = cache.get(key)
        except KeyError:
            raise HTTPException(404, "Unknown or expired report; analyse the case again") from None
        overlay = service.overlay_png(raw, result["analysis"]) if raw else None
        content = build_pdf(result, case_id=key, metadata=meta, original=raw, overlay=overlay)
        return Response(content, media_type="application/pdf", headers={
            "Content-Disposition": f'attachment; filename="case_{key}_report.pdf"', "Cache-Control": "no-store"})

    # With postponed annotations, FastAPI resolves these names in module globals.
    globals().update(UploadFile=UploadFile)

    def register(app):
        original_register(app)

        @app.get("/v2/report/{key}")
        def cached_report(key: str):
            return pdf(key)

        @app.post("/v2/report")
        def uploaded_report(file: UploadFile = File(...), clinical: str | None = Form(None)):
            raw = file.file.read(25 * 1024 * 1024 + 1)
            if not raw:
                raise HTTPException(400, "Empty upload")
            if len(raw) > 25 * 1024 * 1024:
                raise HTTPException(413, "File too large")
            try:
                fields = json.loads(clinical) if clinical else None
            except ValueError:
                raise HTTPException(400, "clinical must be a JSON object") from None
            if fields is not None and not isinstance(fields, dict):
                raise HTTPException(400, "clinical must be a JSON object")
            result = service.analyse_bytes_v2(raw, filename="upload", clinical=fields)
            return pdf(result["report_key"])

    service.analyse_bytes_v2, service.rank_batch, service.register_v2 = analyse, rank, register
    if hasattr(service, "UI_HTML"):
        service.UI_HTML = service.UI_HTML.replace("</body>", UI_EXTENSION + "</body>")
