"""Regression: AP must count lesions that the lung crop cannot see."""
import json
import warnings

import numpy as np
import pytest

from p3_det.data import DataPaths, DetDataConfig, TBDetectionDataset
from p3_det.data.fake_p1 import make_fake_p1_tree


def test_evaluation_retains_a_tb_case_with_all_boxes_outside_crop(tmp_path):
    root = make_fake_p1_tree(tmp_path / "p1", n_per_split=12)
    paths = DataPaths(root)
    cropped_path = paths.coco("development", 0.10)
    cropped = json.loads(cropped_path.read_text())
    positive = cropped["annotations"][0]["image_id"]
    name = next(im["file_name"] for im in cropped["images"] if im["id"] == positive)
    full = json.loads(cropped_path.read_text())
    full["annotations"] = [a for a in full["annotations"] if a["image_id"] != positive]
    full["annotations"].append({"id": 999999, "image_id": positive, "bbox": [0, 0, 2, 2],
                                "orig_bbox_512": [0, 0, 2, 2], "tb_subtype": "active"})
    (root / "data/boxes/v1/coco_development_full.json").write_text(json.dumps(full))
    cropped["annotations"] = [a for a in cropped["annotations"] if a["image_id"] != positive]
    cropped_path.write_text(json.dumps(cropped))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluation = TBDetectionDataset(DetDataConfig("development", eval_mode=True), paths,
                                        tmp_path / "no_decisions")
        training_style = TBDetectionDataset(DetDataConfig("development"), paths, tmp_path / "no_decisions")
    sample = next(s for s in evaluation.samples if s.image_id == name)
    assert np.array_equal(sample.boxes_full512, [[0, 0, 2, 2]])
    assert not any(s.image_id == name for s in training_style.samples)
    assert sample.subtypes == ["active"]


def test_locked_evaluation_refuses_incomplete_cropped_ground_truth(tmp_path):
    root = make_fake_p1_tree(tmp_path / "p1", n_per_split=6)
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    for name, value in (("p2_clahe", {"clahe": False}), ("p2_crop_margin", {"margin": 0.10})):
        (decisions / f"{name}.json").write_text(json.dumps({"locked": True, "value": value}))
    with pytest.raises(FileNotFoundError, match="all original boxes"):
        TBDetectionDataset(DetDataConfig("development", eval_mode=True), DataPaths(root), decisions)
