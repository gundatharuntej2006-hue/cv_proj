import json

import pytest

from p3_det import compare_preprocessing as study
from p3_det.decisions import PREPROCESSING_ABLATION, Refused


def _run(root, name, *, margin, clahe, hit):
    path = root / name
    path.mkdir()
    prep = {"margin": margin, "clahe": clahe, "provisional": True}
    meta = {
        "purpose": PREPROCESSING_ABLATION, "run_dir": name, "input_size": 512,
        "schedule_hash": "same", "seed": 0, "dfine_commit": "test", "repo_git_sha": "test",
        "init": {"note": "test"}, "amp": False, "preprocessing": prep,
        "development_artifact_precision": "native",
        "train_data": {"n": 2, "positives": 1, "negatives": 1, "split": "train"},
        "config": {"schedule": {"epochs": 1}},
        "input_sha256": {key: "a" * 64 for key in ("c01_split", *study.SHARED_INPUTS)},
    }
    gt = {"split": "development", "coords": "full512",
          "images": {"tb": {"boxes": [[0, 0, 10, 10]], "subtype": "active"},
                     "healthy": {"boxes": [], "subtype": None}}}
    det = {"coords": "full512", "input_size": 512, "model_id": name, "epoch": 0, "min_score": 0.01,
           "images": [{"image_id": "tb", "boxes": [[0, 0, 10, 10, 0.99]] if hit else []},
                      {"image_id": "healthy", "boxes": []}]}
    for file, doc in (("run_meta.json", meta), ("dev_ground_truth.json", gt), ("dev_detections_last.json", det)):
        (path / file).write_text(json.dumps(doc))
    (path / "train_log.csv").write_text("epoch\n0\n")
    return path


@pytest.fixture
def runs(tmp_path):
    baseline = _run(tmp_path, "baseline", margin=None, clahe=False, hit=False)
    full_on = _run(tmp_path, "full_on", margin=None, clahe=True, hit=True)
    crop_off = _run(tmp_path, "crop_off", margin=0.10, clahe=False, hit=True)
    crop_on = _run(tmp_path, "crop_on", margin=0.10, clahe=True, hit=True)
    return baseline, [crop_on, crop_off, full_on]  # CLI order must not control comparisons


def test_reports_ordered_comparisons_without_writing_any_decision(tmp_path, runs):
    baseline, candidates = runs
    out = tmp_path / "report.json"
    args = ["--baseline", str(baseline), "--resamples", "8", "--seed", "42", "--out", str(out)]
    for candidate in candidates:
        args += ["--candidate", str(candidate)]
    assert study.main(args) == 0
    report = json.loads(out.read_text())
    assert report["status"] == "provisional"
    assert report["c01_verified_at_training"]
    assert report["runs"][0]["ap50"]["value"] == 0
    assert report["runs"][0]["train_counts"] == {"n": 2, "positives": 1, "negatives": 1}
    assert report["bootstrap"]["seed"] == 42
    assert [r["run"] for r in report["runs"]] == ["baseline", "full_on", "crop_off", "crop_on"]
    assert [r["ap50"]["value"] for r in report["runs"]] == [0, 1, 1, 1]
    assert [c["current"] for c in report["comparisons"]] == ["baseline", "full_on", "full_on"]
    assert [c["promoted"] for c in report["comparisons"]] == [True, False, False]
    assert report["choice_for_review"] == "full_on"
    delta = report["comparisons"][0]["paired_ap50_difference"]
    assert delta["value"] == delta["ci_low"] == delta["ci_high"] == 1
    assert delta["n_boxes"] == 1 and delta["n_images"] == 2
    assert {path.name for path in tmp_path.iterdir()} == {"baseline", "full_on", "crop_off", "crop_on", "report.json"}


def test_promotion_requires_both_the_gain_and_a_positive_interval(runs, monkeypatch):
    from p3_det.metrics.detection import MetricResult

    baseline, candidates = runs
    values = iter([(0.0099, 0.002), (0.01, 0.0), (0.03 - 0.02, 0.001)])

    def paired(*args, **kwargs):
        gain, lower = next(values)
        return MetricResult("ap50_difference", gain, lower, 0.02, 2, 1, 1, 1, "bootstrap_percentile")

    monkeypatch.setattr(study, "ap50_difference", paired)
    report = study.compare(baseline, candidates, resamples=8, seed=42)
    assert [c["current"] for c in report["comparisons"]] == ["baseline"] * 3
    assert [c["promoted"] for c in report["comparisons"]] == [False, False, True]
    assert report["choice_for_review"] == "crop_on"


def test_requires_all_four_protocol_candidates(runs):
    baseline, candidates = runs
    with pytest.raises(Refused, match="all three"):
        study.compare(baseline, candidates[:-1], resamples=8, seed=42)


@pytest.mark.parametrize(("change", "message"), [
    ("split_digest", "shared input digests"),
    ("cohort", "ground truth"),
    ("missing_detection", "cover each development image"),
    ("initialization", "in init"),
    ("precision", "artifact precision"),
])
def test_refuses_incomparable_runs(runs, change, message):
    baseline, candidates = runs
    candidate = candidates[0]
    if change in ("split_digest", "initialization", "precision"):
        path = candidate / "run_meta.json"
        doc = json.loads(path.read_text())
        if change == "split_digest":
            doc["input_sha256"]["c01_split"] = "b" * 64
        elif change == "initialization":
            doc["init"] = {"note": "different initialization"}
        else:
            doc["development_artifact_precision"] = "rounded_3dp"
    elif change == "cohort":
        path = candidate / "dev_ground_truth.json"
        doc = json.loads(path.read_text())
        doc["images"]["tb"]["boxes"] = [[1, 1, 10, 10]]
    else:
        path = candidate / "dev_detections_last.json"
        doc = json.loads(path.read_text())
        doc["images"].pop()
    path.write_text(json.dumps(doc))
    with pytest.raises(Refused, match=message):
        study.compare(baseline, candidates, resamples=8, seed=42)
