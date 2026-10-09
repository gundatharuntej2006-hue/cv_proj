"""R3 `make_pdf` (P3-15, p3_ops.pdf_report) and `log_case` (P3-14, p3_ops.logger).

Frozen signatures (README 4.3):
    make_pdf(case_result) -> pdf_path
    log_case(case_result, operating_point)

``case_result`` is a C14 CaseResult dict. Only whitelisted fields reach the PDF or the log; the case is
identified only by its salted case_id.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from . import logger as _logger
from . import pdf_report
from .logger import LogError, build_record as c15_record, log_path  # noqa: F401  (re-exported)

IS_STUB = False            # make_pdf (P3-15) and log_case (P3-14) are real
LOGGER_IS_STUB = False
REPO = Path(__file__).resolve().parents[2]
OPERATING_POINTS = ("locked_default", "user_override")
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _case_id(case_result: dict) -> str:
    cid = case_result.get("case_id", "")
    if not isinstance(cid, str) or not _CASE_ID_RE.match(cid):
        raise ValueError("case_id must be a short salted hash ([A-Za-z0-9_-]{1,64})")
    return cid


# --- log_case (C15) --------------------------------------------------------------------------------

def log_case(case_result: dict, operating_point: str) -> None:
    """Append exactly one C15 line to logs/predictions.jsonl (p3_ops.logger). P1's threshold slider
    calls this with operating_point="user_override"."""
    _logger.log_case(case_result, operating_point)


# --- make_pdf --------------------------------------------------------------------------------------

def _case_root() -> Path:
    return Path(os.environ.get("TB_CASE_ROOT", REPO))


def _overlay_file(case_result: dict) -> Path | None:
    p = ((case_result.get("explainability") or {}).get("overlay_path"))
    if not p:
        return None
    q = Path(p)
    q = q if q.is_absolute() else _case_root() / q
    if not q.name.startswith(f"{case_result.get('case_id')}_"):
        return None                       # never embed an overlay that belongs to another case
    return q if q.is_file() else None


def _default_pdf_path(case_result: dict, case_id: str) -> Path:
    if os.environ.get("TB_PDF_DIR"):
        return Path(os.environ["TB_PDF_DIR"]) / f"{case_id}.pdf"
    runs = Path(os.environ.get("TB_RUNS_DIR", _case_root() / "runs"))
    return runs / _logger._run_id(case_result) / f"{case_id}_report.pdf"


def make_pdf(case_result: dict, *, image=None, out_path: str | os.PathLike | None = None) -> str:
    """One-page PDF report for a CaseResult; returns its path.

    Frozen R3 signature + keyword-only extensions (docs/p3_contract_notes.md item 15): ``image`` (the
    512x512 image shown next to the overlay; C14 carries no image path) and ``out_path``. The overlay
    is read from ``explainability.overlay_path``. The disclaimer and scope statement come from tbcore.
    """
    cid = _case_id(case_result)
    path = Path(out_path) if out_path is not None else _default_pdf_path(case_result, cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pdf.tmp")
    pdf_report.build(case_result, tmp, image=image, overlay=_overlay_file(case_result))
    os.replace(tmp, path)
    return str(path)
