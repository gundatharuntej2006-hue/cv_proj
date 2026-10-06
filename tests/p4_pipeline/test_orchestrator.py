"""Tests for p4_pipeline orchestrator, batch processing, and R4 API."""

from pathlib import Path
import numpy as np
import pytest

from tbcore.enums import CaseStatus
from tbcore.schemas import CaseResult
from p4_pipeline.api import run_batch, run_case, triage
from p4_pipeline.orchestrator import TriagePipelineOrchestrator


def test_orchestrator_valid_case(tmp_path):
    img_file = tmp_path / "valid_cxr.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    orch = TriagePipelineOrchestrator()
    res = orch.process_file(img_file)

    assert isinstance(res, CaseResult)
    assert res.status in (CaseStatus.ANALYSED, CaseStatus.UNREADABLE)
    assert res.versions.model_version is not None
    assert res.disclaimer is not None


def test_orchestrator_unreadable_file(tmp_path):
    bad_file = tmp_path / "corrupt.png"
    bad_file.touch()  # 0-byte file

    orch = TriagePipelineOrchestrator()
    res = orch.process_file(bad_file)

    assert res.status == CaseStatus.UNREADABLE
    assert res.input.error is not None
    assert res.gate is None
    assert res.classifier is None


def test_r4_api_entry_points(tmp_path):
    f1 = tmp_path / "img1.png"
    f1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    f2 = tmp_path / "img2.png"
    f2.touch()

    # run_batch with mixed files isolates errors
    batch_res = run_batch([f1, f2])
    assert len(batch_res) == 2
    assert isinstance(batch_res[0], CaseResult)
    assert isinstance(batch_res[1], CaseResult)

    # triage entry point
    cat = triage(0.90, n_lesions=1, t_lower=0.18, t_upper=0.83)
    assert cat == "TB"
