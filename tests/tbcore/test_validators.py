"""Tests for tbcore contract validators and schemas."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tbcore.guard import SealedSplitAccessError, check_split_access
from tbcore.paths import get_repo_root
from tbcore.schemas import CaseResult, DecisionRecord
from tbcore.validators import (
    ContractValidationError,
    validate_c01_splits,
    validate_c12_triage,
    validate_c13_operating_curve,
    validate_c14_case_result,
    validate_c16_decision,
    validate_c18_tables,
)


def test_validate_c14_fixtures():
    root = get_repo_root()
    fixtures_dir = root / "mock" / "fixtures" / "case_results"
    assert fixtures_dir.exists(), "Fixtures directory must exist"

    fixtures = list(fixtures_dir.glob("*.json"))
    assert len(fixtures) >= 7, f"Expected at least 7 fixtures, found {len(fixtures)}"

    for f in fixtures:
        res = validate_c14_case_result(f)
        assert isinstance(res, CaseResult)
        assert res.schema_version == "1.0"
        assert res.disclaimer is not None
        assert "not a diagnosis" in res.disclaimer.lower()


def test_validate_c16_all_decisions():
    root = get_repo_root()
    dec_dir = root / "decisions"
    dec_files = list(dec_dir.glob("*.json"))
    assert len(dec_files) >= 23, f"Expected 23 decisions, found {len(dec_files)}"

    for df in dec_files:
        rec = validate_c16_decision(df)
        assert isinstance(rec, DecisionRecord)
        assert rec.locked is True
        assert rec.decision_id.startswith("D")


def test_validate_c12_mock_triage():
    root = get_repo_root()
    c12_path = root / "mock" / "artifacts" / "p4" / "triage" / "triage_development_v1.parquet"
    if c12_path.exists():
        df = validate_c12_triage(c12_path)
        assert isinstance(df, pd.DataFrame)
        assert "p_tb_cal" in df.columns
        assert "category_final" in df.columns


def test_validate_c13_mock_operating_curve():
    root = get_repo_root()
    c13_path = root / "mock" / "artifacts" / "p4" / "triage" / "operating_curve_calibration_v1.parquet"
    if c13_path.exists():
        df = validate_c13_operating_curve(c13_path)
        assert isinstance(df, pd.DataFrame)
        assert df["is_locked_default"].any()


def test_guard_access_control(tmp_path):
    # train/dev/calibration always accessible
    assert check_split_access("train") is True
    assert check_split_access("development") is True
    assert check_split_access("calibration") is True

    # dedup purpose is exempt
    assert check_split_access("internal_test", purpose="dedup") is True

    # sealed split without token raises error
    with pytest.raises(SealedSplitAccessError):
        check_split_access("internal_test", token_dir=tmp_path)

    # create a mock token
    tok_file = tmp_path / "token_internal_test.json"
    with open(tok_file, "w") as fp:
        json.dump({"token_id": "tok_123", "split": "internal_test", "consumed": False}, fp)

    assert check_split_access("internal_test", token_dir=tmp_path) is True

    # consume token
    check_split_access("internal_test", token_dir=tmp_path, consume_token=True)

    # next access without consume_token flag should fail because it was consumed
    with pytest.raises(SealedSplitAccessError):
        check_split_access("internal_test", token_dir=tmp_path, consume_token=False)
