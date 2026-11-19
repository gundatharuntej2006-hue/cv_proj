import json

import pytest

from p3_ops import _tbcore_adapter as ad
from p3_ops.sealed_run import run


def token(tmp_path, **changes):
    data = {"token_id": "token_test", "split": "internal_test", "git_sha": "abc", "consumed": False}
    data.update(changes)
    path = tmp_path / "token_internal_test.json"
    path.write_text(json.dumps(data))
    return path


def test_token_consumed_before_scoring_and_cannot_be_reused(tmp_path):
    path = token(tmp_path)
    with ad.sealed_scope("tbx11k", "internal_test", tmp_path, "abc"):
        assert json.loads(path.read_text())["consumed"]
        ad.require_unlocked("internal_test", "tbx11k", "p3_det")
    with pytest.raises(PermissionError):
        with ad.sealed_scope("tbx11k", "internal_test", tmp_path, "abc"):
            pytest.fail("Token reused")


def test_scope_closes_after_failure_and_claim_survives(tmp_path):
    token(tmp_path)
    with pytest.raises(RuntimeError):
        with ad.sealed_scope("tbx11k", "internal_test", tmp_path, "abc"):
            raise RuntimeError("Scoring failed")
    assert ad._sealed_scope.get() is None
    assert (tmp_path / "p3_claim_internal_test.json").is_file()


@pytest.mark.parametrize("changes", [{"consumed": True}, {"git_sha": "different"}, {"split": "challenge_test"}])
def test_invalid_token_refused_without_claim(tmp_path, changes):
    token(tmp_path, **changes)
    with pytest.raises(PermissionError):
        with ad.sealed_scope("tbx11k", "internal_test", tmp_path, "abc"):
            pytest.fail("Invalid token accepted")
    assert not (tmp_path / "p3_claim_internal_test.json").exists()


def test_concurrent_claim_refused(tmp_path):
    token(tmp_path)
    (tmp_path / "p3_claim_internal_test.json").write_text("reserved")
    with pytest.raises(FileExistsError):
        with ad.sealed_scope("tbx11k", "internal_test", tmp_path, "abc"):
            pytest.fail("Duplicate claim")


def test_runner_refuses_mock_before_accessing_data(monkeypatch, tmp_path):
    monkeypatch.setenv("TB_MOCK", "1")
    with pytest.raises(PermissionError, match="mock"):
        run("internal_test", model_id="test", token_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("split", ["internal_test", "challenge_test"])
@pytest.mark.parametrize("with_comparator", [False, True])
def test_runner_exports_audited_artifacts_without_inventing_challenge_ground_truth(tmp_path, monkeypatch, split, with_comparator):
    import hashlib
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from p3_ops import sealed_run as runner
    from p3_det.data.config import DataPaths
    from p3_det.result import DetResult
    from p3_xai import export_c11
    from p3_xai import gradcam

    monkeypatch.delenv("TB_MOCK", raising=False)
    monkeypatch.delenv("TB_DET_ALLOW_UNLOCKED", raising=False)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda args, **kw: "abc" if "rev-parse" in args else "")
    monkeypatch.setattr(ad, "validate_gate", lambda: None)
    monkeypatch.setattr(export_c11, "check_registry", lambda models: "D14")
    monkeypatch.setattr(runner, "resolve_preprocessing", lambda: (0.1, False, False))
    monkeypatch.setattr(runner.detector, "detector_info", lambda: {
        "model_id": "test_detector", "conf_threshold": 0.5, "input_size": 512,
        "locked": True, "mock": False})
    monkeypatch.setattr(runner.detector, "detect", lambda *args: DetResult(
        np.array([[10, 20, 40, 50]], np.float32), np.array([0.8], np.float32), 1))
    monkeypatch.setattr(runner.xai, "explainer", lambda: SimpleNamespace(
        module=SimpleNamespace(model_id="test_classifier"),
        spec=SimpleNamespace(as_record=lambda: {"fixture": "synthetic"})))
    monkeypatch.setattr(runner.xai, "explain", lambda *args: np.ones((512, 512), np.float16))
    full_calls = []

    class FullExplainer:
        def __init__(self, module):
            self.module = module
            self.spec = SimpleNamespace(as_record=lambda: {"fixture": "synthetic_full"})

        def __call__(self, image, crop):
            full_calls.append(crop)
            return np.ones((512, 512), np.float16)

    monkeypatch.setattr(export_c11, "p2_provider", lambda mid, role: lambda: SimpleNamespace(model_id=mid))
    monkeypatch.setattr(gradcam, "GradCamExplainer", FullExplainer)
    image_path = tmp_path / "image.png"
    Image.new("L", (512, 512)).save(image_path)
    monkeypatch.setattr(runner, "split_images", lambda *args: ([("tb0001", image_path, [0, 0, 512, 512])], 0))
    paths = DataPaths(tmp_path)
    paths.splits.parent.mkdir(parents=True)
    paths.splits.write_text("image_id,split,category\ntb0001," + split +
                            (",active_tb\n" if split == "internal_test" else ",unknown\n"))
    if split == "internal_test":
        full = tmp_path / "data/boxes/v1/coco_internal_test_full.json"
        full.parent.mkdir(parents=True)
        full.write_text(json.dumps({"images": [{"id": 1, "file_name": "tb0001"}],
            "annotations": [{"image_id": 1, "bbox": [10, 20, 30, 30], "tb_subtype": "active"}]}))
    out = tmp_path / "outputs"
    if split == "challenge_test":
        prior = out / "sealed/internal_test.json"
        prior.parent.mkdir(parents=True)
        prior.write_text(json.dumps({"status": "complete"}))
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / f"token_{split}.json").write_text(json.dumps(
        {"token_id": "test", "split": split, "git_sha": "abc", "consumed": False}))
    audit = json.loads(runner.run(split, model_id="test_classifier", paths=paths,
                                  full_model_id="test_full" if with_comparator else None,
                                  out_root=out, token_dir=gate).read_text())
    assert audit["status"] == "complete"
    assert audit["explanation_spec"] == {"fixture": "synthetic"}
    assert audit["localisation_evaluable"] == (split == "internal_test")
    assert len(audit["artifact_sha256"]) == (5 if with_comparator else 3)
    assert full_calls == ([[0, 0, 512, 512]] if with_comparator else [])
    assert len(audit["explanation_specs"]) == (2 if with_comparator else 1)
    for path, digest in audit["artifact_sha256"].items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == digest
    import pyarrow.parquet as pq
    table = pq.read_table(audit["localisation"])
    if split == "internal_test":
        assert table.column("tb_subtype").to_pylist() == ["active"]
    else:
        assert table.num_rows == 0
        assert not (tmp_path / "data/boxes/v1/coco_challenge_test_full.json").exists()
