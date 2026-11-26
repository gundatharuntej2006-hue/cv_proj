"""Tests for p4_eval: No-test-tuning gate, table generators, challenge mapping, and comparisons."""

import json
from pathlib import Path
import pandas as pd
import pytest

from p4_eval.challenge import format_challenge_submission, generate_table_12_8_challenge
from p4_eval.comparisons import generate_table_14_comparisons
from p4_eval.external import run_all_external_evaluations
from p4_eval.gate import GateValidationError, check_all_decisions, issue_token
from p4_eval.recalibration import run_external_recalibration
from p4_eval.tables import export_evaluation_tables
from tbcore.validators import validate_c18_tables


def test_gate_check_all_decisions():
    with pytest.raises(GateValidationError, match="placeholder git_sha"):
        check_all_decisions()


def test_gate_issue_token(tmp_path, monkeypatch):
    dec_dir = tmp_path / "decisions"
    dec_dir.mkdir()
    (tmp_path / "evidence.txt").write_text("review evidence", encoding="utf-8")
    for i in range(1, 24):
        record = {
            "decision_id": f"D{i:02d}", "owner": "p4", "title": "reviewed",
            "criterion_preregistered": "fixed", "preregistered_at": "2026-10-05",
            "candidates": [], "value": {}, "evidence": ["evidence.txt"],
            "data_used": [], "reviewed_by": "p1", "guide_ack": False,
            "locked": True, "locked_at": "2026-10-05",
            "git_sha": "a" * 40,
        }
        (dec_dir / f"d{i:02d}.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr("p4_eval.gate.get_decisions_dir", lambda: dec_dir)
    monkeypatch.setattr("p4_eval.gate.get_repo_root", lambda: tmp_path)
    monkeypatch.setattr("p4_eval.gate.resolve_path", lambda _: tmp_path / "audit.md")
    token_path = issue_token("internal_test", out_dir=tmp_path / "tokens")
    assert token_path.exists()
    assert len(check_all_decisions()) == 23
    with pytest.raises(GateValidationError, match="already exists"):
        issue_token("internal_test", out_dir=tmp_path / "tokens")

    record_path = dec_dir / "d01.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["git_sha"] = "c0ffee1"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(GateValidationError, match="placeholder git_sha"):
        issue_token("external_test", out_dir=tmp_path / "tokens")
    assert not (tmp_path / "tokens" / "token_external_test.json").exists()


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
    with pytest.raises(RuntimeError, match="no verified comparison artifacts"):
        generate_table_14_comparisons(out_dir=tmp_path)
    assert not (tmp_path / "t14_comparisons.csv").exists()

    t13 = run_all_external_evaluations(out_dir=tmp_path)
    assert len(t13) >= 4
    validate_c18_tables(tmp_path / "t13_external.csv")
