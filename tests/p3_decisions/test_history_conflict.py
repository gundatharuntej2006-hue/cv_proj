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
