"""P3-15: per-case PDF (R3 make_pdf) for the 7 CaseResult scenario types.

Inputs are the test-only stand-ins for P4's 7 fixtures (tests/p3_mock/p3_case_results.py) and a
test-only tbcore.disclaimer stand-in (placeholder texts) until P4's tbcore exists.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "tests" / "p3_mock", REPO / "tests" / "p3_ops"):
    sys.path.insert(0, str(p))

import tbcore_standin  # noqa: E402
from p3_case_results import ANALYSED, P4_SCENARIOS  # noqa: E402
from test_logger import IDENTIFIER_PATTERNS, leaky  # noqa: E402

from p3_ops import api, pdf_report  # noqa: E402

ANALYSED_KINDS = ("tb", "refer", "not_tb", "link_fired", "segmentation_fallback")


def norm(s: str) -> str:
    return " ".join(s.split())


@pytest.fixture
def env(tmp_path, monkeypatch):
    tbcore_standin.install(monkeypatch)
    monkeypatch.setenv("TB_CASE_ROOT", str(tmp_path / "case"))
    monkeypatch.delenv("TB_PDF_DIR", raising=False)
    monkeypatch.delenv("TB_RUNS_DIR", raising=False)
    monkeypatch.delenv("TB_RUN_ID", raising=False)
    # the overlay the CaseResults point to (runs/r0012/c_7f3a9e21_overlay.png), and the image
    from p3_xai import api as xai_api
    yy, xx = np.mgrid[0:512, 0:512]
    img = (90 + 60 * np.sin(xx / 40) * np.cos(yy / 55)).clip(0, 255).astype(np.uint8)
    heat = np.exp(-((xx - 330) ** 2 + (yy - 125) ** 2) / (2 * 35 ** 2)).astype(np.float16)
    for cr in P4_SCENARIOS.values():                       # each case's own overlay, at its overlay_path
        if cr.get("explainability"):
            xai_api.render_overlay(img, heat, (cr.get("detector") or {}).get("boxes"), case_id=cr["case_id"],
                                   out_path=tmp_path / "case" / cr["explainability"]["overlay_path"])
    return {"tmp": tmp_path, "img": img}


def render(cr, env, **kw) -> tuple[Path, str]:
    path = Path(api.make_pdf(cr, image=env["img"], **kw))
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1
    return path, norm(reader.pages[0].extract_text())


# --- 7 fixture types -----------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", list(P4_SCENARIOS))
def test_pdf_renders_for_each_fixture_type(env, kind):
    cr = P4_SCENARIOS[kind]
    path, text = render(cr, env)
    assert path == env["tmp"] / "case" / "runs" / ("r0012" if cr.get("explainability") else "unknown") / \
        f"{cr['case_id']}_report.pdf"
    page = pypdf.PdfReader(str(path)).pages[0]
    assert (round(float(page.mediabox.width)), round(float(page.mediabox.height))) == (595, 842)   # A4
    assert f"Case {cr['case_id']}" in text
    assert path.stat().st_size < 2 * 1024 * 1024                                            # email-able


@pytest.mark.parametrize("kind", list(P4_SCENARIOS))
def test_disclaimer_and_scope_extractable_in_every_pdf(env, kind):
    _, text = render(P4_SCENARIOS[kind], env)
    assert norm(tbcore_standin.DISCLAIMER) in text
    assert norm(tbcore_standin.SCOPE_HONESTY) in text


@pytest.mark.parametrize("kind", ANALYSED_KINDS)
def test_analysed_pdf_has_results_overlay_caption_and_lesions(env, kind):
    cr = P4_SCENARIOS[kind]
    _, text = render(cr, env)
    assert "Results" in text and "Attention evidence, not a finding" in text
    assert f"{cr['classifier']['p_tb_cal']:.3f}" in text
    assert {"TB": "TB", "REFER": "REFER", "NOT_TB": "NOT TB"}[cr["triage"]["category_final"]] in text
    assert "t_lower 0.18" in text and "t_upper 0.83" in text and "locked default" in text
    assert "Lesion detector" in text
    for b in cr["detector"]["boxes"]:
        assert f"{b['score']:.2f}" in text
    assert ("fired (NOT TB moved to REFER)" in text) == bool(cr["triage"]["link_fired"])
    assert ("whole image analysed" in text) == (kind == "segmentation_fallback")


def test_rejected_gate_pdf(env):
    _, text = render(P4_SCENARIOS["rejected_gate"], env)
    assert "rejected" in text.lower() and "Send for manual review." in text
    assert "out-of-distribution score 0.510 above threshold 0.241" in text
    assert "Results" not in text and "Lesion detector" not in text and "Grad-CAM overlay" not in text


def test_unreadable_pdf(env):
    _, text = render(P4_SCENARIOS["unreadable"], env)
    assert "could not be read" in text and "Error: cannot decode image" in text
    assert "Results" not in text and "Lesion detector" not in text


def test_detector_contingency_omits_lesion_section(env):
    cr = copy.deepcopy(P4_SCENARIOS["refer"])
    cr["detector"] = None
    _, text = render(cr, env)
    assert "Lesion detector" not in text and "Results" in text and "Attention evidence, not a finding" in text


def test_overlay_of_another_case_is_never_embedded(env, tmp_path):
    cr = copy.deepcopy(P4_SCENARIOS["tb"])
    cr["explainability"]["overlay_path"] = ANALYSED["explainability"]["overlay_path"]   # exists, other case
    path = Path(api.make_pdf(cr, image=env["img"], out_path=tmp_path / "x.pdf"))
    assert "overlay not available" in norm(pypdf.PdfReader(str(path)).pages[0].extract_text())


def test_user_override_and_missing_overlay_and_image(env, tmp_path):
    cr = copy.deepcopy(ANALYSED)
    cr["triage"]["operating_point"] = "user_override"
    cr["explainability"]["overlay_path"] = "runs/r0012/does_not_exist.png"
    path = Path(api.make_pdf(cr, out_path=tmp_path / "x.pdf"))
    text = norm(pypdf.PdfReader(str(path)).pages[0].extract_text())
    assert "user override" in text and "overlay not available" in text and "image not provided" in text


# --- no identifiers -----------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", list(P4_SCENARIOS))
def test_no_identifiers_in_text_or_metadata(env, kind):
    path, text = render(leaky(P4_SCENARIOS[kind]), env)
    meta = " ".join(str(v) for v in (pypdf.PdfReader(str(path)).metadata or {}).values())
    for name, pat in IDENTIFIER_PATTERNS.items():
        assert not re.search(pat, text), (name, re.findall(pat, text))
        assert not re.search(pat, meta), (name, meta)


def test_redaction():
    s = pdf_report.redact(r"PatientName=DOE^JANE (0010,0010) C:\Users\jane\scan_1985.dcm /home/js/x.png id 1234567")
    assert s == "[redacted] [tag] [path] [path] id [number]"     # "PatientName=DOE^JANE" redacted whole
    assert pdf_report.redact("cannot decode image") == "cannot decode image"


# --- single source of the texts -----------------------------------------------------------------------

def test_refuses_without_tbcore(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tbcore", None)          # import tbcore -> ImportError
    with pytest.raises(RuntimeError, match="tbcore"):
        api.make_pdf(ANALYSED, out_path=tmp_path / "x.pdf")
    assert not (tmp_path / "x.pdf").exists()


def test_texts_are_not_retyped_in_p3_code():
    src = (REPO / "src" / "p3_ops" / "pdf_report.py").read_text() + (REPO / "src" / "p3_ops" / "api.py").read_text()
    assert "not a diagnosis" not in src and "Validated only on public" not in src
