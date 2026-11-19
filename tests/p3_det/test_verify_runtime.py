import pytest
from p3_det.verify_runtime import compare_entry, verify


def test_runtime_comparison_catches_count_score_and_geometry_drift():
    expected = {"crop_box": [0, 0, 512, 512], "n_lesions_above_thr": 1,
                "boxes": [[10, 20, 30, 40, 0.8], [50, 60, 70, 80, 0.2]]}
    actual = {**expected, "boxes": list(reversed(expected["boxes"]))}
    assert compare_entry(expected, actual)
    assert not compare_entry(expected, {**actual, "n_lesions_above_thr": 0})
    assert not compare_entry(expected, {**actual, "boxes": [[10, 20, 30, 40, 0.7]]})
    assert not compare_entry(expected, {**actual, "crop_box": [10, 20, 30, 40]})
    empty = {**expected, "boxes": [], "n_lesions_above_thr": 0}
    assert compare_entry(empty, empty)


def test_acceptance_refuses_mock_before_loading_data_or_model(monkeypatch):
    monkeypatch.setenv("TB_MOCK", "1")
    with pytest.raises(PermissionError, match="mock"):
        verify(device="cpu")
