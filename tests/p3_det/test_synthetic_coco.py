"""P3-01: synthetic COCO generator used for detector feasibility runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from p3_det.synthetic_coco import make_synthetic_coco  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    return out, json.loads(make_synthetic_coco(out, n_images=20, size=128, seed=3).read_text())


def test_single_tb_lesion_category_like_c06(coco):
    _, d = coco
    assert d["categories"] == [{"id": 1, "name": "tb_lesion"}]
    assert {a["category_id"] for a in d["annotations"]} == {1}
    assert all(a["iscrowd"] == 0 for a in d["annotations"])


def test_has_positives_and_negatives(coco):
    _, d = coco
    positives = {a["image_id"] for a in d["annotations"]}
    assert 0 < len(positives) < len(d["images"])


def test_boxes_in_bounds_and_area_consistent(coco):
    _, d = coco
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        assert w > 0 and h > 0 and x >= 0 and y >= 0
        assert x + w <= 128 and y + h <= 128
        assert a["area"] == pytest.approx(w * h)


def test_images_written_and_ids_synthetic(coco):
    out, d = coco
    ids = [a["id"] for a in d["annotations"]]
    assert len(ids) == len(set(ids))
    for im in d["images"]:
        assert im["file_name"].startswith("syn_")
        with Image.open(out / "train" / im["file_name"]) as img:
            assert img.size == (128, 128) and img.mode == "RGB"


def test_deterministic_for_seed(tmp_path):
    a = make_synthetic_coco(tmp_path / "a", 5, size=64, seed=7).read_text()
    b = make_synthetic_coco(tmp_path / "b", 5, size=64, seed=7).read_text()
    c = make_synthetic_coco(tmp_path / "c", 5, size=64, seed=8).read_text()
    assert a == b and a != c
