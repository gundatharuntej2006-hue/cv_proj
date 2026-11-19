"""Tests for p4_eval: No-test-tuning gate, table generators, challenge mapping, and comparisons."""

from pathlib import Path
import json
import pandas as pd
import pytest

from p4_eval.challenge import format_challenge_submission, generate_table_12_8_challenge
from p4_eval.comparisons import generate_table_14_comparisons
from p4_eval.external import run_all_external_evaluations
from p4_eval.gate import check_all_decisions, issue_token, GateValidationError
from p4_eval.recalibration import run_external_recalibration
from p4_eval.tables import export_evaluation_tables
from tbcore.validators import validate_c18_tables


@pytest.fixture
def locked_test_decisions(tmp_path, monkeypatch):
    """Gate mechanics use synthetic test copies, never lock the research decisions."""
    import p4_eval.gate as gate
    target = tmp_path / "synthetic_decisions"
    target.mkdir()
    for path in gate.get_decisions_dir().glob("*.json"):
        data = json.loads(path.read_text())
        data.update(locked=True, value=data.get("value") or {"synthetic_test_fixture": True})
        (target / path.name).write_text(json.dumps(data))
    monkeypatch.setattr(gate, "get_decisions_dir", lambda: target)
    monkeypatch.setattr(gate, "resolve_path", lambda rel: tmp_path / rel)
    return target


def test_gate_check_all_decisions(locked_test_decisions):
    decs = check_all_decisions()
    assert len(decs) == 23
    for d_id, rec in decs.items():
        assert rec.locked is True


def test_gate_issue_token(tmp_path, locked_test_decisions):
    token_path = issue_token("internal_test", out_dir=tmp_path)
    assert token_path.exists()


def test_gate_refuses_unlocked_decision(locked_test_decisions):
    path = locked_test_decisions / "p3_det_input_size.json"
    data = json.loads(path.read_text())
    data["locked"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(GateValidationError, match="D16.*not locked"):
        check_all_decisions()


def test_recalibration_rejects_subsets_without_enough_tb_cases(tmp_path):
    df = pd.DataFrame({"tb_label": [1] * 20 + [0] * 80,
                       "p_tb_raw": [0.8] * 20 + [0.2] * 80})
    with pytest.raises(ValueError, match="only 5 labelled TB cases"):
        run_external_recalibration(df, subset_sizes=[25], out_dir=tmp_path)


def test_table_generation_c18(tmp_path):
    export_evaluation_tables(out_dir=tmp_path)

    t12_1 = tmp_path / "t12_1_classification.csv"
    assert t12_1.exists()
    df12_1 = validate_c18_tables(t12_1)
    assert "roc_auc" in df12_1["metric"].values

    t12_3 = tmp_path / "t12_3_triage.csv"
    assert t12_3.exists()
    df12_3 = validate_c18_tables(t12_3)
    assert df12_3["n_tb"].gt(0).all()


def test_comparisons_and_external_tables(tmp_path):
    t14 = generate_table_14_comparisons(out_dir=tmp_path)
    assert len(t14) >= 9
    validate_c18_tables(tmp_path / "t14_comparisons.csv")

    t13 = run_all_external_evaluations(out_dir=tmp_path)
    assert len(t13) >= 4
    validate_c18_tables(tmp_path / "t13_external.csv")
