import json

import pytest

from p3_det import decisions as d


def test_result_recording_refuses_conflicting_older_results(tmp_path, monkeypatch):
    path = tmp_path / "decision.json"
    doc = d.new_decision("D16", "test", "current rule", [512, 640], ["development"])
    path.write_text(json.dumps(doc))
    before = path.read_bytes()
    commits = [d.Commit("old", "old", "legacy locked scaffold", "older rule", True, True),
               d.Commit("new", "new", "current preregistration", "current rule", False, False)]
    monkeypatch.setattr(d, "history", lambda p: commits)
    with pytest.raises(d.Refused, match="history conflicts"):
        d.load_for_result(path, "current rule")
    assert path.read_bytes() == before


def test_final_training_and_preflight_require_a_detector_protocol(tmp_path, monkeypatch, capsys):
    from p3_det.train import run as training
    from p3_ops import preflight

    decision_dir = tmp_path / "decisions"
    decision_dir.mkdir()
    decision = {"locked": True, "value": {"clahe": False}, "git_sha": "c0ffee1",
                "reviewed_by": "p4", "locked_at": "2026-11-19", "evidence": ["result.csv"]}
    path = decision_dir / "p2_clahe.json"
    path.write_text(json.dumps(decision))
    assert "placeholder git_sha" in d.reviewed_lock_problem(path)
    path.write_text(json.dumps({**decision, "git_sha": "a" * 40}))
    assert d.reviewed_lock_problem(path) is None
    (decision_dir / "p2_crop_margin.json").write_text(
        json.dumps({**decision, "git_sha": "b" * 40, "value": {"margin": 0.10}}))

    from p3_det import setup

    def no_checkout(_):
        raise RuntimeError("fixture has no D-FINE checkout")

    monkeypatch.setattr(setup, "setup", no_checkout)
    checks = {c["check"]: c for c in preflight.inspect(tmp_path, check_cuda=False, stage="training")}
    assert not checks["detector preprocessing protocol"]["ready"]
    assert "P2 D08/D09 apply to classification" in checks["detector preprocessing protocol"]["detail"]

    monkeypatch.setattr(training, "REPO", tmp_path)
    monkeypatch.delenv("TB_MOCK", raising=False)
    with pytest.raises(SystemExit) as exc:
        training.main(["--config", "dfine_512.yaml"])
    assert exc.value.code == 2
    assert "agreed detector preprocessing protocol" in capsys.readouterr().err
