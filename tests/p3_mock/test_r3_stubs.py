"""P3-03: R3 entry points keep the frozen signatures; stubs/mocks return the right types and shapes.

Since P3-08, p3_det.api.detect is the real detector; its P3-03 stub is served when TB_MOCK=1, which
is what these consumer-facing tests use. explain/render_overlay/make_pdf/log_case are still stubs.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("jsonschema")

from PIL import Image  # noqa: E402

import standin_contracts as sc  # noqa: E402
from p3_det import api as det_api  # noqa: E402
from p3_ops import api as ops_api  # noqa: E402
from p3_xai import api as xai_api  # noqa: E402

CROP = [10, 0, 507, 493]

# README 4.3, R3: frozen positional parameters
SIGNATURES = {
    (det_api, "detect"): ["img_for_analysis", "crop_box"],
    (xai_api, "explain"): ["img_for_analysis", "crop_box"],
    (xai_api, "render_overlay"): ["img512", "heatmap", "boxes"],
    (ops_api, "make_pdf"): ["case_result"],
    (ops_api, "log_case"): ["case_result", "operating_point"],
}


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_LOG_PATH", str(tmp_path / "logs" / "predictions.jsonl"))
    monkeypatch.setenv("TB_PDF_DIR", str(tmp_path / "pdf"))
    monkeypatch.setenv("TB_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.delenv("TB_RUN_ID", raising=False)
    monkeypatch.setenv("TB_MOCK", "1")


@pytest.mark.parametrize("mod_fn, params", SIGNATURES.items(), ids=lambda x: getattr(x, "__name__", ""))
def test_signatures_frozen(mod_fn, params):
    mod, name = mod_fn
    sig = inspect.signature(getattr(mod, name)).parameters.values()
    frozen = [p.name for p in sig if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert frozen == params                                   # frozen R3 positional parameters unchanged
    extras = [p for p in sig if p.name not in params]
    assert all(p.kind is p.KEYWORD_ONLY and p.default is not p.empty for p in extras)   # only optional kw-only
    if mod is det_api:
        assert det_api.IS_STUB is False                      # real since P3-08; mock via TB_MOCK=1
    elif mod is ops_api:
        assert ops_api.IS_STUB is False and ops_api.LOGGER_IS_STUB is False       # P3-15 / P3-14
    elif mod is xai_api:
        assert xai_api.IS_STUB is False and xai_api.RENDER_IS_STUB is False  # P3-09 / P3-11
    else:
        assert mod.IS_STUB is True and "STUB" in mod.__doc__


def test_detresult_fields():
    assert [f.name for f in det_api.DetResult.__dataclass_fields__.values()] == [
        "boxes_full512", "scores", "n_lesions_above_thr"]


def test_detect_types_shapes_and_bounds(img512):
    hits = 0
    for k in range(25):
        r = det_api.detect((img512 + k).astype(np.uint8), CROP)
        assert isinstance(r, det_api.DetResult)
        assert r.boxes_full512.dtype == np.float32 and r.boxes_full512.ndim == 2
        assert r.boxes_full512.shape[1] == 4 and r.scores.shape == (len(r.boxes_full512),)
        assert r.scores.dtype == np.float32 and (r.scores >= 0.01).all() and (r.scores <= 1).all()
        assert list(r.scores) == sorted(r.scores, reverse=True)
        b = r.boxes_full512
        assert (b[:, 0] >= CROP[0]).all() and (b[:, 2] <= CROP[2]).all()
        assert (b[:, 1] >= CROP[1]).all() and (b[:, 3] <= CROP[3]).all()
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
        assert isinstance(r.n_lesions_above_thr, int)
        assert r.n_lesions_above_thr == int((r.scores >= det_api.STUB_CONF_THRESHOLD).sum())
        hits += len(b) > 0
    assert hits > 0


def test_detect_deterministic_and_accepts_rgb(img512):
    a, b = det_api.detect(img512, CROP), det_api.detect(img512.copy(), CROP)
    assert np.array_equal(a.boxes_full512, b.boxes_full512) and np.array_equal(a.scores, b.scores)
    det_api.detect(np.stack([img512] * 3, axis=-1), CROP)


@pytest.mark.parametrize("bad_img, bad_crop", [
    (np.zeros((256, 256), np.uint8), CROP), (None, [0, 0, 600, 10]), (None, [50, 0, 40, 10])])
def test_detect_rejects_bad_inputs(img512, bad_img, bad_crop):
    with pytest.raises(ValueError):
        det_api.detect(img512 if bad_img is None else bad_img, bad_crop)


def test_explain_heatmap_contract(img512, tmp_path):
    h = xai_api.explain(img512, CROP)
    path = tmp_path / "h.npy"
    np.save(path, h)
    sc.validate_c11_heatmap(path, crop_box=CROP)      # float16 [512,512], [0,1], zero outside crop
    assert h.max() == 1.0
    assert np.array_equal(h, xai_api.explain(img512, CROP))


def test_render_overlay_png_with_caption(img512):
    h = xai_api.explain(img512, CROP)
    det = det_api.detect(img512, CROP)
    for boxes in (det, [{"xyxy": [301, 88, 372, 160], "score": 0.77}], [[301, 88, 372, 160, 0.77]], []):
        path = xai_api.render_overlay(img512, h, boxes)
        assert isinstance(path, str) and path.endswith(".png")
        with Image.open(path) as im:
            assert im.format == "PNG" and im.width >= 512 and im.height >= 512
            assert im.text["Caption"] == "Attention evidence, not a finding"


def test_make_pdf(case_result, monkeypatch):
    """Consumer-level check only; the full P3-15 tests are in tests/p3_ops/test_pdf_report.py."""
    import tbcore_standin
    tbcore_standin.install(monkeypatch)
    path = ops_api.make_pdf(case_result)
    assert isinstance(path, str) and path.endswith(f"{case_result['case_id']}.pdf")
    data = open(path, "rb").read()
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")


def test_make_pdf_refuses_unsafe_case_id(case_result):
    case_result["case_id"] = "../../etc/passwd"
    with pytest.raises(ValueError):
        ops_api.make_pdf(case_result)


def test_case_result_examples_are_valid_c14(case_result):
    sc.validate_c14(case_result)


def test_log_case_appends_valid_c15(case_result, tmp_path):
    ops_api.log_case(case_result, "locked_default")
    ops_api.log_case(case_result, "user_override")
    lines = sc.validate_c15_file(tmp_path / "logs" / "predictions.jsonl")
    assert len(lines) == 2 and [l["operating_point"] for l in lines] == ["locked_default", "user_override"]
    l = lines[0]
    assert l["case_id"] == case_result["case_id"] and l["status"] == case_result["status"]
    assert l["git_sha"] == case_result["versions"]["git_sha"]
    if case_result["status"] == "ANALYSED":
        assert l["run_id"] == "r0012" and l["latency_ms"] == 1205
        assert l["category_final"] == case_result["triage"]["category_final"]
    else:
        assert l["category_final"] is None and l["p_tb_cal"] is None


def test_log_case_run_id_from_env(monkeypatch, tmp_path):
    from p3_case_results import ANALYSED
    monkeypatch.setenv("TB_RUN_ID", "r9999")
    ops_api.log_case(ANALYSED, "locked_default")
    assert json.loads((tmp_path / "logs" / "predictions.jsonl").read_text())["run_id"] == "r9999"


def test_log_case_rejects_bad_operating_point():
    from p3_case_results import ANALYSED
    with pytest.raises(ValueError):
        ops_api.log_case(ANALYSED, "whatever")


def test_log_never_contains_free_text(tmp_path):
    from p3_case_results import ANALYSED
    leaky = json.loads(json.dumps(ANALYSED))
    leaky["input"]["error"] = "PatientName=DOE^JANE"
    leaky["gate"]["quality_reasons"] = ["PatientID 12345"]
    ops_api.log_case(leaky, "locked_default")
    text = (tmp_path / "logs" / "predictions.jsonl").read_text()
    assert "DOE" not in text and "12345" not in text
