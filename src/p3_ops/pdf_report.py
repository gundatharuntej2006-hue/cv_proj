"""P3-15: per-case PDF report (ReportLab), one A4 page, from a C14 CaseResult.

Sections: header (case_id, generation time, short versions) - status banner - images (image | Grad-CAM
overlay, captioned) - results - lesions (only when the CaseResult has a detector block) - disclaimer
and scope statement.

* The decision-support DISCLAIMER and the SCOPE-HONESTY text come from the single source in tbcore
  (P4). They are imported, never retyped; without them no PDF is produced (fail closed).
* Only the salted case_id identifies the case. Free text that could carry identifiers (input.error,
  quality/fail reasons) is shown only after redaction (paths, file names, DICOM tags, DICOM person names).
* REJECTED_GATE: reasons + "Send for manual review", no analysis. UNREADABLE: the (redacted) error,
  no analysis. A null detector block (e.g. detector contingency) omits the lesion section.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.pdfgen import canvas as rl_canvas

from p3_xai.overlay import CAPTION as OVERLAY_CAPTION

PAGE_W, PAGE_H = A4
M = 36                                                   # margin (pt)
CAT_COLOURS = {"TB": colors.HexColor("#b00020"), "REFER": colors.HexColor("#c77700"),
               "NOT_TB": colors.HexColor("#2e7d32")}
CAT_LABEL = {"TB": "TB", "REFER": "REFER", "NOT_TB": "NOT TB"}
IMAGE_PX = 640                                          # images are downsampled to keep the file small
MAX_LESIONS_LISTED = 12

_REDACT = [
    (re.compile(r"[A-Za-z]:[\\/][^\s,;]*"), "[path]"),                   # Windows paths
    (re.compile(r"(?<![\w.])/(?:[\w.-]+/)+[\w.-]*"), "[path]"),          # Unix paths
    (re.compile(r"\b[\w.-]+\.(?:dcm|dicom|png|jpe?g|tiff?|bmp|nii(?:\.gz)?)\b", re.I), "[file]"),
    (re.compile(r"\(\s*[0-9A-Fa-f]{4}\s*,\s*[0-9A-Fa-f]{4}\s*\)"), "[tag]"),
    (re.compile(r"\b[A-Za-z]+\^[A-Za-z^]+\b"), "[name]"),
    (re.compile(r"\bPatient\w*\s*[=:]?\s*\S*", re.I), "[redacted]"),
    (re.compile(r"\d{6,}"), "[number]"),
]


def redact(text) -> str:
    s = " ".join(str(text or "").split())[:200]
    for pat, rep in _REDACT:
        s = pat.sub(rep, s)
    return s


def tbcore_texts() -> tuple[str, str]:
    """(DISCLAIMER, SCOPE_HONESTY) from tbcore, the single source (P4-01). Fails closed."""
    try:
        from tbcore import disclaimer as d
    except ImportError as e:
        raise RuntimeError("tbcore.disclaimer (P4) is not available: refusing to produce a PDF without the "
                           "single-source disclaimer and scope statement") from e
    try:
        return str(d.DISCLAIMER), str(d.SCOPE_HONESTY)
    except AttributeError as e:
        raise RuntimeError("tbcore.disclaimer must provide DISCLAIMER and SCOPE_HONESTY (names to confirm "
                           "with P4)") from e


def _short(v, n: int) -> str:
    v = str(v or "?")
    return (v.split(":", 1)[1] if v.startswith("sha256:") else v)[:n]


def _img_reader(src) -> ImageReader | None:
    if src is None:
        return None
    im = Image.open(src) if isinstance(src, (str, Path)) else Image.fromarray(np.asarray(src).astype(np.uint8))
    im = im.convert("RGB")
    im.thumbnail((IMAGE_PX, IMAGE_PX * 2), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return ImageReader(buf)


def _para(c, text, x, y_top, width, style) -> float:
    """Draw a wrapped paragraph with its top at y_top; return the y below it."""
    p = Paragraph(text, style)
    _, h = p.wrap(width, PAGE_H)
    p.drawOn(c, x, y_top - h)
    return y_top - h


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


STYLE = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=12.5)
SMALL = ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11)
BOLD = ParagraphStyle("bold", parent=STYLE, fontName="Helvetica-Bold")


def _image_box(c, reader, x, y, w, h, title, caption=None, missing="not available"):
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x, y + h + 4, title)
    c.setStrokeColor(colors.HexColor("#999999"))
    c.rect(x, y, w, h)
    if reader is None:
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawCentredString(x + w / 2, y + h / 2, missing)
        c.setFillColor(colors.black)
    else:
        iw, ih = reader.getSize()
        s = min(w / iw, h / ih)
        c.drawImage(reader, x + (w - iw * s) / 2, y + (h - ih * s) / 2, iw * s, ih * s)
    if caption:
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.HexColor("#780000"))
        c.drawCentredString(x + w / 2, y - 12, caption)
        c.setFillColor(colors.black)


def _results(c, cr: dict, x: float, y: float, w: float) -> float:
    cls, tri, seg = (cr.get(k) or {} for k in ("classifier", "triage", "segmentation"))
    cat = tri.get("category_final")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Results")
    y -= 18
    # category tag
    c.setFont("Helvetica", 9.5)
    c.drawString(x, y, "Final category")
    tag = CAT_LABEL.get(cat, str(cat))
    c.setFillColor(CAT_COLOURS.get(cat, colors.grey))
    c.roundRect(x + 200, y - 4, 70, 16, 3, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x + 235, y, tag)
    c.setFillColor(colors.black)
    y -= 18
    link = tri.get("link_fired")
    rows = [
        ("Calibrated TB probability (p_tb_cal)", f"{cls.get('p_tb_cal'):.3f}" if cls.get("p_tb_cal") is not None else "-"),
        ("Thresholds", f"t_lower {tri.get('t_lower')}  /  t_upper {tri.get('t_upper')}"),
        ("Category before link rule", CAT_LABEL.get(tri.get("category_pre_link"), "-")),
        ("Detector link rule", "fired (NOT TB moved to REFER)" if link else "applied, did not fire" if link is False else "-"),
        ("Operating point", {"locked_default": "locked default", "user_override": "user override"}.get(
            tri.get("operating_point"), "-")),
    ]
    if seg.get("fallback_full"):
        rows.append(("Lung segmentation", "failed sanity check; whole image analysed"))
    for k, v in rows:
        c.setFont("Helvetica", 9.5)
        c.drawString(x, y, k)
        c.drawString(x + 200, y, _esc(v))
        y -= 14
    return y


def _lesions(c, det: dict, x: float, y: float, w: float) -> float:
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Lesion detector")
    y -= 16
    boxes = det.get("boxes") or []
    scores = sorted((float(b.get("score", 0)) for b in boxes), reverse=True)
    text = (f"{det.get('n_lesions', len(boxes))} lesion box(es) above the confidence threshold "
            f"{det.get('conf_threshold')}")
    if scores:
        shown = ", ".join(f"{s:.2f}" for s in scores[:MAX_LESIONS_LISTED])
        more = f" (+{len(scores) - MAX_LESIONS_LISTED} more)" if len(scores) > MAX_LESIONS_LISTED else ""
        text += f"; scores: {shown}{more}"
    text += ". A detected lesion is a reason for review; it does not establish TB."
    return _para(c, _esc(text), x, y + 10, w, STYLE) - 8


def build(cr: dict, out: Path, *, image=None, overlay=None, now: datetime | None = None) -> Path:
    disclaimer, scope = tbcore_texts()
    status = cr.get("status")
    c = rl_canvas.Canvas(str(out), pagesize=A4, pageCompression=1)
    c.setTitle(f"TB triage decision-support report {cr['case_id']}")
    c.setAuthor("TB screening project (decision support)")
    c.setSubject("")
    c.setCreator("p3_ops.make_pdf")

    # header
    c.setFillColor(colors.HexColor("#1f2933"))
    c.rect(0, PAGE_H - 70, PAGE_W, 70, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(M, PAGE_H - 30, "TB triage - decision-support report")
    c.setFont("Helvetica", 10)
    c.drawRightString(PAGE_W - M, PAGE_H - 30, f"Case {cr['case_id']}")
    v = cr.get("versions") or {}
    gen = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%MZ")
    c.setFont("Helvetica", 8.5)
    c.drawString(M, PAGE_H - 50, f"Generated {gen}   |   model {_esc(v.get('model_version'))}   split "
                                 f"{_esc(v.get('split_version'))}   decisions {_short(v.get('decisions_hash'), 12)}   "
                                 f"git {_short(v.get('git_sha'), 7)}")
    c.setFillColor(colors.black)
    y = PAGE_H - 70 - 24
    inner = PAGE_W - 2 * M

    if status in ("REJECTED_GATE", "UNREADABLE"):
        c.setFillColor(colors.HexColor("#fff4e5"))
        c.rect(M, y - 120, inner, 120, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 13)
        title = "Image rejected by the quality / out-of-distribution gate" if status == "REJECTED_GATE" \
            else "Image could not be read"
        c.drawString(M + 12, y - 22, title)
        if status == "REJECTED_GATE":
            g = cr.get("gate") or {}
            reasons = [redact(r) for r in (g.get("quality_reasons") or [])]
            if g.get("ood_score") is not None and g.get("ood_threshold") is not None and \
                    g["ood_score"] > g["ood_threshold"]:
                reasons.append(f"out-of-distribution score {g['ood_score']:.3f} above threshold {g['ood_threshold']:.3f}")
            body = "Reasons: " + (_esc("; ".join(reasons)) if reasons else "not specified")
        else:
            body = "Error: " + _esc(redact((cr.get("input") or {}).get("error")) or "not specified")
        yy = _para(c, body, M + 12, y - 34, inner - 24, STYLE)
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.HexColor("#b00020"))
        c.drawString(M + 12, yy - 22, "Send for manual review." if status == "REJECTED_GATE"
                     else "No analysis performed. Send for manual review.")
        c.setFillColor(colors.black)
        y -= 150
    else:
        box = (inner - 20) / 2
        img_y = y - 14 - box
        _image_box(c, _img_reader(image), M, img_y, box, box, "Image", missing="image not provided")
        _image_box(c, _img_reader(overlay), M + box + 20, img_y, box, box, "Grad-CAM overlay",
                   caption=OVERLAY_CAPTION, missing="overlay not available")
        y = img_y - 34
        y = _results(c, cr, M, y, inner) - 10
        if cr.get("detector"):
            y = _lesions(c, cr["detector"], M, y, inner)

    # disclaimer + scope (single source: tbcore)
    c.setStrokeColor(colors.HexColor("#780000"))
    c.setLineWidth(1.2)
    yy = _para(c, "<b>Disclaimer.</b> " + _esc(disclaimer), M + 8, y - 8, inner - 16, SMALL)
    yy = _para(c, "<b>Scope.</b> " + _esc(scope), M + 8, yy - 6, inner - 16, SMALL)
    c.rect(M, yy - 6, inner, y - (yy - 6))
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(M, 22, "Heatmaps and boxes are attention evidence, not a finding. Identifies the case only by its "
                        "salted case ID.")
    c.showPage()
    c.save()
    return out
