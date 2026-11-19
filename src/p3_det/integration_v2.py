"""Optional P3 bridge for Atul's regional service; no classifier retraining or score changes.

The service has screening flags, not calibrated TB/NOT_TB categories. Therefore
detector referral is recorded separately and never represented as a calibrated label.
"""
from __future__ import annotations

import copy
import io
import time

import numpy as np
from PIL import Image, ImageDraw

from . import api


def region_agreement(analysis, evidence):
    """Exploratory geometric overlap with P2's displayed seed-mean argmax cell.

    Positive-area intersection in full512 coordinates; touching edges do not
    count. This is neither ground-truth accuracy nor P2's published patch metric.
    """
    note = ("Exploratory predicted-box overlap with the displayed seed-mean region; "
            "not lesion accuracy or exact attribution of the ensemble score")
    if evidence.get("status") != "available":
        return {"status": "unavailable", "any_overlap": None, "note": note}
    region = analysis.get("argmax_region") or {}
    row, col = region.get("row"), region.get("col")
    if (isinstance(row, bool) or isinstance(col, bool) or
            not isinstance(row, int) or not isinstance(col, int) or
            not (0 <= row < 12 and 0 <= col < 12)):
        return {"status": "unavailable", "any_overlap": None,
                "reason": "invalid_or_missing_region", "note": note}
    cell = np.array([col, row, col + 1, row + 1], float) * (512.0 / 12)
    rows = []
    for box in evidence["boxes"]:
        b = np.asarray(box["xyxy"], float)
        wh = np.maximum(0, np.minimum(cell[2:], b[2:]) - np.maximum(cell[:2], b[:2]))
        area = float(np.prod(wh))
        union = float(np.prod(cell[2:] - cell[:2]) + np.prod(b[2:] - b[:2]) - area)
        rows.append({"intersection_area": area, "iou": area / union, "overlaps": area > 0})
    return {"status": "available", "method": "positive_area_intersection_full512",
            "cell_xyxy": cell.tolist(), "n_boxes": len(rows),
            "n_overlapping": sum(r["overlaps"] for r in rows),
            "any_overlap": any(r["overlaps"] for r in rows), "per_box": rows, "note": note}


def analyse(image, *, crop_box=None, detector=api):
    """512 uint8 image -> JSON-safe lesion count and boxes. Unavailable != zero lesions."""
    image = np.asarray(image)
    crop_box = [0, 0, 512, 512] if crop_box is None else crop_box
    result = detector.detect(image, crop_box)
    info = detector.detector_info()
    if info.get("mock"):
        raise RuntimeError("Mock detections cannot be attached to a real v2 analysis")
    if info.get("locked") is not True:
        raise RuntimeError("Unlocked detector cannot be attached to the v2 service")
    threshold = float(info["conf_threshold"])
    boxes = np.asarray(result.boxes_full512)
    scores = np.asarray(result.scores)
    if boxes.shape != (len(scores), 4) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid detector box/score output")
    if np.any(boxes < 0) or np.any(boxes > 512) or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Detector boxes outside full512 coordinates")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Detector scores outside [0,1]")
    kept = [{"xyxy": [float(v) for v in b], "score": float(s)}
            for b, s in zip(boxes, scores) if s >= threshold]
    if len(kept) != result.n_lesions_above_thr:
        raise ValueError("Detector count disagrees with the locked confidence threshold")
    return {"status": "available", "model_id": info["model_id"], "conf_threshold": threshold,
            "n_lesions": len(kept), "boxes": kept, "coords": "full512",
            "checkpoint_sha256": info.get("checkpoint_sha256"), "mock": False}


def referral(category, n_lesions):
    """The original monotone link rule. A detector never downgrades a category."""
    if category not in ("TB", "REFER", "NOT_TB"):
        raise ValueError("Unknown triage category")
    if isinstance(n_lesions, bool) or not isinstance(n_lesions, int) or n_lesions < 0:
        raise ValueError("Lesion count must be a non-negative integer")
    return "REFER" if category == "NOT_TB" and n_lesions > 0 else category


def unavailable(result, reason, *, disabled=False):
    """An unavailable/disabled detector must not imply a negative lesion result."""
    out = copy.deepcopy(result)
    if out.get("status") != "ANALYSED":
        return out
    evidence = {"status": "disabled" if disabled else "unavailable", "reason": reason,
                "n_lesions": None, "boxes": [], "coords": "full512"}
    out["analysis"]["detector"] = evidence
    out["analysis"]["detector_region_agreement"] = region_agreement(out["analysis"], evidence)
    out["analysis"]["detector_referral"] = {"available": False, "additional_review": False,
        "note": "Independent lesion review unavailable; classifier operating points unchanged"}
    return out


def enrich(result, image, *, detector=api, crop_box=None):
    """Add independent detector evidence; preserve P2 scores, flags, fusion and region maps."""
    out = copy.deepcopy(result)
    if out.get("status") != "ANALYSED":
        return out
    t0 = time.perf_counter()
    try:
        evidence = analyse(image, crop_box=crop_box, detector=detector)
    except Exception as exc:
        evidence = {"status": "unavailable", "reason": type(exc).__name__,
                    "n_lesions": None, "boxes": [], "coords": "full512"}
    out["analysis"]["detector"] = evidence
    out["analysis"]["detector_region_agreement"] = region_agreement(out["analysis"], evidence)
    available = evidence["status"] == "available"
    positive = out["analysis"]["operating_points"]["screen_positive_at_spec_0.70"]
    out["analysis"]["detector_referral"] = {
        "available": available,
        "additional_review": bool(available and evidence["n_lesions"] > 0 and not positive),
        "note": "Independent lesion review flag; classifier operating points unchanged",
    }
    out["detector_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if out.get("latency_ms") is not None:
        out["latency_ms"] += out["detector_latency_ms"]
    return out


def draw_boxes(png, analysis):
    """Add cyan detector boxes to P2's region overlay, preserving its decision-cell layer."""
    out = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(out)
    evidence = analysis.get("detector") or {}
    for box in evidence.get("boxes", []):
        draw.rectangle(box["xyxy"], outline=(0, 220, 255), width=2)
        x, y = box["xyxy"][:2]
        draw.text((max(0, min(470, x)), max(0, y - 12)), f"{box['score']:.2f}", fill=(0, 220, 255))
    # Add the caption outside the image: never cover lesions at the bottom edge.
    captioned = Image.new("RGB", (out.width, out.height + 48), "black")
    captioned.paste(out, (0, 0))
    draw = ImageDraw.Draw(captioned)
    label = (f"P3 lesion boxes: {evidence['n_lesions']} (cyan)" if evidence.get("status") == "available"
             else "P3 detector unavailable")
    draw.text((5, out.height + 4), label, fill="white")
    draw.text((5, out.height + 18), "Attention evidence, not a finding", fill="white")
    draw.text((5, out.height + 32), "Green: displayed region. Cyan: predicted lesions.", fill="white")
    buf = io.BytesIO()
    captioned.save(buf, format="PNG")
    return buf.getvalue()


def install(service):
    """Opt in on the loaded scripts/serve_v2.py module before register_v2(app).

    Both single and batch routes call analyse_bytes_v2, and all overlays call
    overlay_png, so this one hook covers direct and cached results.
    """
    if getattr(service, "_p3_installed", False):
        return service
    original_analyse, original_overlay = service.analyse_bytes_v2, service.overlay_png
    original_rank = getattr(service, "rank_batch", None)

    def analyse_bytes(raw, filename="upload", clinical=None):
        started = time.perf_counter()
        result = original_analyse(raw, filename=filename, clinical=clinical)
        if result.get("status") != "ANALYSED":
            return result
        if getattr(service, "p3_detector_enabled", True) is False:
            return unavailable(result, "disabled_by_configuration", disabled=True)
        import cv2
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        # A provider must reproduce BOTH D08 image preparation and D09 geometry.
        # Never silently substitute P2's lungs-masked input for detector training input.
        provider = getattr(service, "p3_input_provider", None)
        if provider is None:
            return unavailable(result, "input_provider_missing")
        try:
            prepared, crop = provider(image)
        except Exception as exc:
            return unavailable(result, type(exc).__name__)
        result = enrich(result, prepared, crop_box=crop)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result

    def overlay(raw, analysis):
        return draw_boxes(original_overlay(raw, analysis), analysis)

    service.analyse_bytes_v2, service.overlay_png = analyse_bytes, overlay
    if original_rank is not None:
        def rank(results):
            ranked = original_rank(results)
            analysed = sorted((r for r in results if r.get("status") == "ANALYSED"),
                              key=lambda r: -r["analysis"]["image_logit"])
            for row, source in zip(ranked["cases"], analysed):
                row["detector"] = source["analysis"].get("detector")
                row["detector_referral"] = source["analysis"].get("detector_referral")
                row["detector_region_agreement"] = source["analysis"].get("detector_region_agreement")
            ranked["n_additional_detector_review"] = sum(
                bool((r["analysis"].get("detector_referral") or {}).get("additional_review")) for r in analysed)
            return ranked
        service.rank_batch = rank
    if hasattr(service, "register_v2"):
        from p3_ops.v2_reports import install as install_reports
        install_reports(service)
    service._p3_installed = True
    return service


def make_input_provider(bundle):
    """Reuse P1's C04/C05 rules and P2's loaded segmenter; no new crop algorithm."""
    from p2_cls import pipeline_v2 as p2
    from .data.config import resolve_preprocessing
    margin, clahe, provisional = resolve_preprocessing()
    if provisional:
        raise RuntimeError("Detector service requires locked D08/D09 preprocessing")
    p2._scripts_on_path()
    import p1_build_c04_c05 as p1

    def prepare(image):
        segmenter = p2._segmenter(bundle)
        mask = p2._lung_mask(segmenter, image, bundle["_segmenter_device"])
        info = p1.roi_from_mask(mask)
        crop = p1.crop_box(info["box"], margin) if info["sanity_pass"] else (0, 0, 512, 512)
        return (p1.clahe(image) if clahe else image), list(crop)

    return prepare
