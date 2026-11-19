from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p3_det.result import DetResult
from p3_ops import latency
from p4_pipeline.orchestrator import TriagePipelineOrchestrator


def test_pipeline_uses_full_image_scores_and_real_heatmap(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_RUNS_DIR", str(tmp_path / "runs"))
    path = tmp_path / "case.png"
    path.write_bytes(b"fixture")
    image = np.full((512, 512), 100, np.uint8)
    heatmap = np.full((512, 512), 0.5, np.float16)
    seen = {}

    def detect(img, crop):
        assert img.shape == (512, 512)
        assert crop == [10, 20, 400, 450]
        return DetResult(np.array([[20, 30, 50, 70]], np.float32), np.array([0.8], np.float32), 1)

    def overlay(img, heat, boxes):
        assert heat is heatmap
        assert len(boxes) == 1 and boxes[0]["score"] > 0.7
        seen["overlay"] = True
        return str(tmp_path / "overlay.png")

    r1 = SimpleNamespace(standardise=lambda p: SimpleNamespace(img512=image, status="ok"),
                         lung_roi=lambda img: SimpleNamespace(crop_box=[10, 20, 400, 450]))
    det = SimpleNamespace(detect=detect, detector_info=lambda: {"model_id": "real", "conf_threshold": 0.5})
    xai = SimpleNamespace(explain=lambda img, crop: heatmap, render_overlay=overlay)
    orch = TriagePipelineOrchestrator(r1_module=r1, r3_det_module=det, r3_xai_module=xai)
    result = orch.process_file(path)
    assert result.detector.n_lesions == 1
    assert seen["overlay"]
    assert Path(result.explainability.heatmap_path).is_file()
    assert np.array_equal(np.load(result.explainability.heatmap_path), heatmap)


def test_failed_detector_does_not_leave_fabricated_boxes(tmp_path):
    path = tmp_path / "case.png"
    path.write_bytes(b"fixture")

    def broken(*args):
        raise RuntimeError("Missing checkpoint")

    result = TriagePipelineOrchestrator(r3_det_module=SimpleNamespace(detect=broken)).process_file(path)
    assert result.detector is None
    assert result.triage.link_fired is False
    assert result.explainability is None


def test_latency_accepts_nested_pydantic_case_result(tmp_path):
    path = tmp_path / "case.png"
    path.write_bytes(b"fixture")
    result = TriagePipelineOrchestrator().process_file(path)
    rows = latency.measure([path], lambda paths: [result], n_warmup=0)
    assert len(rows) == 1
    assert rows[0]["status"] == "ANALYSED"
    assert isinstance(rows[0]["total"], (float, int))


def test_actual_c14_can_be_exported_and_logged_with_real_tbcore(tmp_path, monkeypatch):
    import json
    from pypdf import PdfReader
    from p3_ops import api, _tbcore_adapter
    monkeypatch.setenv("TB_RUN_ID", "integration")
    monkeypatch.setenv("TB_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    path = tmp_path / "case.png"
    path.write_bytes(b"synthetic fixture")
    result = TriagePipelineOrchestrator().process_file(path)
    output = api.make_pdf_with_original(result, out_path=tmp_path / "case.pdf")
    text = PdfReader(output).pages[0].extract_text()
    disclaimer, _ = _tbcore_adapter.disclaimer_texts()
    assert " ".join(disclaimer.split()) in " ".join(text.split())
    api.log_case(result, "locked_default")
    record = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert record["run_id"] == "integration"
    assert record["case_id"] == result.case_id
    assert _tbcore_adapter.validate("C15", tmp_path / "predictions.jsonl")
