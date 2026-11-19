"""P3 v2 report integration: preserved scores, cache isolation and PDF semantics."""
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

from p3_det import integration_v2 as bridge
from p3_det.result import DetResult
from p3_ops import v2_reports as reports


def png():
    buf = io.BytesIO()
    Image.new("L", (512, 512), 120).save(buf, format="PNG")
    return buf.getvalue()


def case():
    return {"image": "PRIVATE_123456.png", "status": "ANALYSED", "provisional": True, "latency_ms": 12,
            "analysis": {"image_logit": -0.1234, "region_logits": [[0] * 12] * 12,
                         "argmax_region": {"row": 0, "col": 0, "logit": 0},
                         "operating_points": {"screen_positive_at_spec_0.70": False, "positive_at_spec_0.90": False},
                         "lung_mask": {"lung_fraction": 0.4}, "fusion": None}}


def test_agreement_positive_area_and_empty_vs_unavailable():
    analysis = case()["analysis"]
    evidence = {"status": "available", "boxes": [{"xyxy": [10, 20, 50, 60], "score": 0.9}]}
    agreement = bridge.region_agreement(analysis, evidence)
    assert agreement["any_overlap"] and agreement["n_overlapping"] == 1
    assert 0 < agreement["per_box"][0]["iou"] < 1
    edge = 512 / 12
    evidence["boxes"][0]["xyxy"] = [edge, 0, 100, 10]
    assert not bridge.region_agreement(analysis, evidence)["any_overlap"]
    assert bridge.region_agreement(analysis, {"status": "available", "boxes": []})["any_overlap"] is False
    assert bridge.region_agreement(analysis, {"status": "unavailable"})["any_overlap"] is None
    assert bridge.region_agreement({}, evidence)["status"] == "unavailable"


def test_cache_is_bounded_and_different_clinical_results_cannot_overwrite():
    cache = reports.ReportCache(capacity=2)
    a = case()
    k1 = cache.remember(png(), a, {"model_id": "test"})
    a["analysis"]["image_logit"] = 2
    k2 = cache.remember(png(), a, {})
    assert k1 != k2 and cache.get(k1)[1]["analysis"]["image_logit"] == -0.1234
    assert "image" not in cache.get(k1)[1]
    cached = cache.get(k2)[1]
    cached["analysis"]["image_logit"] = 3
    assert cache.get(k2)[1]["analysis"]["image_logit"] == 2
    cache.remember(b"bad", {"status": "REJECTED", "analysis": None}, {})
    with pytest.raises(KeyError):
        cache.get(k1)


def test_pdf_uses_actual_tbcore_text_and_logits_without_private_filename():
    from p3_ops.pdf_report import tbcore_texts
    result = case()
    content = reports.build_pdf(result, case_id="a" * 32, metadata={"model_id": "regional_v2"}, original=png())
    text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(content)).pages)
    assert "Screening logit: -0.1234" in text
    assert "not calibrated probabilities" in text
    assert "PRIVATE" not in text
    assert all(" ".join(t.split()) in " ".join(text.split()) for t in tbcore_texts())
    assert "detector unavailable" in text.lower()


def test_rejected_report_has_no_score_and_refuses_non_null_analysis():
    result = {"status": "REJECTED", "analysis": None, "reasons": ["PRIVATE_123456.png"]}
    content = reports.build_pdf(result, case_id="b" * 32, metadata={})
    text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(content)).pages)
    assert "Screening logit:" not in text and "PRIVATE" not in text
    assert "Send for manual review" in text
    result["analysis"] = case()["analysis"]
    with pytest.raises(ValueError, match="must not contain"):
        reports.build_pdf(result, case_id="test", metadata={})


def test_overlay_keeps_bottom_edge_and_caption():
    result = case()
    result["analysis"]["detector"] = {"status": "available", "n_lesions": 1,
        "boxes": [{"xyxy": [10, 490, 50, 510], "score": 0.8}]}
    im = Image.open(io.BytesIO(bridge.draw_boxes(png(), result["analysis"])))
    assert im.size == (512, 560)
    assert im.getpixel((10, 505)) == (0, 220, 255)


def test_routes_reuse_cached_inference_and_keep_batch_report_keys(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    calls = []
    detector = SimpleNamespace(
        detect=lambda *a: DetResult(np.array([[10, 20, 50, 60]], np.float32), np.array([0.9]), 1),
        detector_info=lambda: {"model_id": "test", "conf_threshold": 0.5, "mock": False, "locked": True})
    monkeypatch.setattr(bridge.api, "detect", detector.detect)
    monkeypatch.setattr(bridge.api, "detector_info", detector.detector_info)

    def analyse(raw, filename="upload", clinical=None):
        calls.append(clinical)
        return case()

    service = SimpleNamespace(analyse_bytes_v2=analyse, overlay_png=lambda *a: png(),
        rank_batch=lambda results: {"cases": [{"status": "ANALYSED"} for r in results]},
        register_v2=lambda app: None, p3_input_provider=lambda image: (image, [0, 0, 512, 512]),
        _bundle=lambda: {"meta": {"model": {"model_id": "regional_v2"}}},
        UI_HTML="<body><script>function render(res){}</script></body>")
    bridge.install(service)
    app = FastAPI()
    service.register_v2(app)
    client = TestClient(app)
    out = service.analyse_bytes_v2(png(), clinical={"sex": 1})
    key = out["report_key"]
    ranked = service.rank_batch([out])
    assert ranked["cases"][0]["report_key"] == key
    assert ranked["cases"][0]["detector_region_agreement"]["any_overlap"]
    response = client.get(f"/v2/report/{key}")
    assert response.status_code == 200 and response.content.startswith(b"%PDF")
    assert response.headers["cache-control"] == "no-store"
    assert calls == [{"sex": 1}]  # download does not rerun or lose fusion selection
    assert client.get("/v2/report/unknown").status_code == 404
    response = client.post("/v2/report", files={"file": ("PRIVATE.png", png(), "image/png")},
                           data={"clinical": '{"sex": 0}'})
    assert response.status_code == 200 and calls[-1] == {"sex": 0}
    assert client.post("/v2/report", files={"file": ("x.png", png())}, data={"clinical": "[]"}).status_code == 400
    assert "Download PDF" in service.UI_HTML


def test_disabled_detector_never_calls_model_or_turns_unavailable_into_zero():
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled detector was called")
    service = SimpleNamespace(analyse_bytes_v2=lambda *a, **kw: case(), overlay_png=lambda *a: png(),
        p3_input_provider=forbidden, p3_detector_enabled=False)
    bridge.install(service)
    result = service.analyse_bytes_v2(png())
    assert result["analysis"]["detector"]["status"] == "disabled"
    assert result["analysis"]["detector"]["n_lesions"] is None
    assert not result["analysis"]["detector_referral"]["available"]
    assert not result["analysis"]["detector_referral"]["additional_review"]
    assert result["analysis"]["image_logit"] == case()["analysis"]["image_logit"]


def test_rejected_upload_pdf_survives_missing_classifier_bundle():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    service = SimpleNamespace(analyse_bytes_v2=lambda *a, **kw: {"status": "REJECTED", "analysis": None},
        rank_batch=lambda results: {"cases": []}, register_v2=lambda app: None,
        overlay_png=lambda *a: pytest.fail("Rejected report must not draw an overlay"),
        _bundle=lambda: (_ for _ in ()).throw(FileNotFoundError("private/path")))
    reports.install(service)
    app = FastAPI()
    service.register_v2(app)
    response = TestClient(app).post("/v2/report", files={"file": ("PRIVATE.png", b"corrupt")})
    assert response.status_code == 200
    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(response.content)).pages)
    assert "Send for manual review" in text and "PRIVATE" not in text and "private/path" not in text
