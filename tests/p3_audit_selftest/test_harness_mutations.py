"""Mutation tests for the audit harness (FIX 0): a check that cannot be made to fail protects nothing.

For D2, D3 and G4: plant a real violation in a temporary copy, the check must FAIL; the clean copy must PASS.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "p3_verify"))
sys.path.insert(0, str(REPO / "tests" / "p3_ops"))
import checks  # noqa: E402

pytest.importorskip("yaml")


def _copy_src(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    return root


# --- D3: hard-coded confidence threshold ------------------------------------------------------------

def test_D3_detects_planted_hardcoded_threshold():
    clean = (REPO / "src/p3_det/api.py").read_text(encoding="utf-8")
    assert checks.hardcoded_settings(clean) == []                                  # real file passes
    mutated = clean.replace("IS_STUB = False", "IS_STUB = False\nCONF_THRESHOLD = 0.42")
    assert checks.hardcoded_settings(mutated) == ["CONF_THRESHOLD = 0.42"]         # planted setting caught
    assert checks.hardcoded_settings(clean + '\n"""text with >= 0.01 and 512 px"""') == []   # docstrings ignored


# --- D2: STUB marker in a real module -----------------------------------------------------------------

def test_D2_detects_planted_stub_marker_in_real_module(tmp_path):
    root = _copy_src(tmp_path)
    target = root / "src/p3_ops/logger.py"
    assert not [h for h in checks.code_markers(root) if "p3_ops/logger.py" in h]   # clean copy: no marker
    target.write_text(target.read_text(encoding="utf-8") + "\n\ndef log_case_stub():  # STUB returns nothing\n"
                      "    return None\n", encoding="utf-8")
    hits = [h for h in checks.code_markers(root) if "p3_ops/logger.py" in h]
    assert hits and "STUB" in hits[0]


def test_D2_flag_lines_and_mock_module_are_not_counted(tmp_path):
    root = _copy_src(tmp_path)
    (root / "src/p3_ops/x.py").write_text("IS_STUB = False\nLOGGER_IS_STUB = False\n", encoding="utf-8")
    (root / "src/p3_det/_mock.py").write_text("# MOCK only, served with TB_MOCK=1; STUB wording ok here\n",
                                              encoding="utf-8")
    assert not [h for h in checks.code_markers(root) if "x.py" in h or "_mock.py" in h]


def test_D2b_detects_null_config(tmp_path):
    root = _copy_src(tmp_path)
    cfg = root / "src/p3_det/configs/dfine_512.yaml"
    text = cfg.read_text(encoding="utf-8")
    import re
    filled = re.sub(r"epochs: .*", "epochs: 48", re.sub(r"batch_size: .*", "batch_size: 8", text))
    cfg.write_text(filled, encoding="utf-8")
    assert not [h for h in checks.config_pending(root) if h.startswith("dfine_512.yaml")]
    cfg.write_text(re.sub(r"epochs: .*", "epochs: null", filled), encoding="utf-8")
    assert "dfine_512.yaml: schedule.epochs = null" in checks.config_pending(root)


# --- G4: DICOM-style tag in a log line -------------------------------------------------------------

def test_G4_detects_planted_dicom_tag_in_log():
    from test_logger import IDENTIFIER_PATTERNS
    clean = {"logs/predictions.jsonl": '{"ts":"2026-11-30T10:00:00.000+00:00","case_id":"c_7f3a9e21"}'}
    assert checks.identifier_hits(clean, IDENTIFIER_PATTERNS, set()) == {}
    dirty = {"logs/predictions.jsonl": clean["logs/predictions.jsonl"] + ' {"note":"(0010,0010) DOE^JANE"}'}
    hits = checks.identifier_hits(dirty, IDENTIFIER_PATTERNS, set())
    assert any("DICOM tag number" in k for k in hits) and any("DICOM person name" in k for k in hits)


def test_G4_names_only_exemption_is_narrow():
    from test_logger import IDENTIFIER_PATTERNS
    names = {"artifact names": "c_7f3a9e21_overlay.png"}
    assert checks.identifier_hits(names, IDENTIFIER_PATTERNS, {"artifact names"}) == {}
    leaked = {"artifact names": "c_7f3a9e21_overlay.png /home/jsmith/scan.dcm"}       # user path still caught
    assert any("Unix user path" in k for k in checks.identifier_hits(leaked, IDENTIFIER_PATTERNS, {"artifact names"}))
    assert checks.identifier_hits({"pdf text": "scan_0001.png"}, IDENTIFIER_PATTERNS, set())   # PDFs: png caught
