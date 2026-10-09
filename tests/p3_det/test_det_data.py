"""P3-04: detector dataset adapter on FAKE P1 data (P1 mocks are not in the repo yet)."""
from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
pytest.importorskip("PIL")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from p3_det.data import (  # noqa: E402
    AugmentConfig, DataPaths, DetDataConfig, SealedSplitError, TBDetectionDataset, augment, check_split,
    det_to_full512, full512_to_crop, full512_to_det, make_dataloader)
from p3_det.data.dataset import crop_and_resize  # noqa: E402
from p3_det.data.fake_p1 import make_fake_p1_tree  # noqa: E402


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return make_fake_p1_tree(tmp_path_factory.mktemp("p1"), n_per_split=15, seed=0)


@pytest.fixture(scope="module")
def no_decisions(tmp_path_factory):
    return tmp_path_factory.mktemp("decisions")   # D08/D09 not locked -> provisional


def ds(root, no_decisions, split="train", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBDetectionDataset(DetDataConfig(split=split, **kw), DataPaths(root), no_decisions)


# --- index: splits, positives, negatives ---------------------------------------------------------

def test_train_index_positives_negatives(root, no_decisions):
    d = ds(root, no_decisions)
    info = d.describe()
    assert info["provisional"] and info["margin"] == 0.10 and info["clahe"] is False and info["augment"]
    assert info["positives"] > 0 and info["negatives"] > 0
    assert info["skipped_tb_without_boxes"] > 0               # uncertain TB is not a negative
    import csv
    with open(root / "data/splits/tbx11k_splits_v1.csv", newline="") as f:
        split_of = {r["image_id"]: r["split"] for r in csv.DictReader(f)}
    assert {split_of[s.image_id] for s in d.samples} == {"train"}
    assert all((len(s.boxes_full512) > 0) == (s.tb_label == 1) for s in d.samples)


def test_fake_data_covers_edge_cases(root, no_decisions):
    d = ds(root, no_decisions)
    assert any(s.crop_box == (0, 0, 512, 512) for s in d.samples)          # fallback_full crop
    clipped = [s for s in d.samples if len(s.boxes_full512)
               and (s.boxes_full512[:, 0] < s.crop_box[0]).any()]          # lesion straddles crop edge
    assert clipped


def test_development_split_no_augmentation(root, no_decisions):
    d = ds(root, no_decisions, split="development")
    assert d.describe()["augment"] is False
    img1, b1, _ = d.load(0)
    img2, b2, _ = d.load(0)
    assert np.array_equal(img1, img2) and np.array_equal(b1, b2)


def test_provisional_warning_and_locked_decisions(root, tmp_path):
    with pytest.warns(UserWarning, match="PROVISIONAL"):
        TBDetectionDataset(DetDataConfig("train"), DataPaths(root), tmp_path)
    for name, value in (("p2_crop_margin.json", {"margin": 0.05}), ("p2_clahe.json", {"clahe": True})):
        (tmp_path / name).write_text(json.dumps({"value": value, "locked": True}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # no provisional warning now
        d = TBDetectionDataset(DetDataConfig("train"), DataPaths(root), tmp_path)
    assert (d.margin, d.clahe, d.provisional) == (0.05, True, False)
    assert all("img512_clahe" in str(s.img_path) for s in d.samples)
    with pytest.raises(ValueError, match="D09"):
        TBDetectionDataset(DetDataConfig("train", margin=0.10), DataPaths(root), tmp_path)


def test_c06_inconsistency_is_caught(root, no_decisions, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(root, bad)
    p = bad / "data/boxes/v1/coco_train_m0.10.json"
    doc = json.loads(p.read_text())
    doc["annotations"][0]["bbox"][0] += 5
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="disagrees"):
        ds(bad, no_decisions)


# --- sealed splits -------------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["internal_test", "challenge_test", "external_test", "excluded_duplicate"])
def test_sealed_split_refused_before_any_file_is_read(split, tmp_path):
    missing = DataPaths(tmp_path / "does_not_exist")           # opening anything would raise OSError
    with pytest.raises(SealedSplitError):
        TBDetectionDataset(DetDataConfig(split), missing, tmp_path)
    with pytest.raises(SealedSplitError):
        make_dataloader(DetDataConfig(split), batch_size=2, num_workers=0, paths=missing)


def _fake_tbcore(monkeypatch, guard):
    pkg = types.ModuleType("tbcore")
    pkg.guard = guard
    monkeypatch.setitem(sys.modules, "tbcore", pkg)
    monkeypatch.setitem(sys.modules, "tbcore.guard", guard)


def test_sealed_split_with_guard(monkeypatch, root, no_decisions):
    # guard without the expected API -> still refused (fails closed)
    _fake_tbcore(monkeypatch, types.ModuleType("tbcore.guard"))
    with pytest.raises(SealedSplitError):
        check_split("internal_test")
    # guard that refuses (no token)
    g = types.ModuleType("tbcore.guard")

    def no_token(**kw):
        raise PermissionError("no unlock token")
    g.require_unlocked = no_token
    _fake_tbcore(monkeypatch, g)
    with pytest.raises(PermissionError):
        ds(root, no_decisions, split="internal_test")
    # guard that allows (token present): loads
    g.require_unlocked = lambda **kw: None
    assert len(ds(root, no_decisions, split="internal_test")) > 0
    # excluded duplicates are never loadable, guard or not
    with pytest.raises(SealedSplitError):
        check_split("excluded_duplicate")


# --- geometry: round trip ------------------------------------------------------------------------

@pytest.mark.parametrize("S", [512, 640])
def test_box_round_trip_exact(S):
    r = np.random.default_rng(0)
    for _ in range(500):
        x0, y0 = r.integers(0, 80, 2)
        crop = (int(x0), int(y0), int(r.integers(400, 513)), int(r.integers(400, 513)))
        w, h = r.uniform(3, 100, 2)
        bx, by = r.uniform(crop[0], crop[2] - w), r.uniform(crop[1], crop[3] - h)
        box = np.array([[bx, by, bx + w, by + h]])
        det, keep = full512_to_det(box, crop, S)
        assert keep.all()
        back = det_to_full512(det, crop, S)
        assert np.abs(back - box).max() < 1e-6


@pytest.mark.parametrize("S", [512, 640])
def test_resize_round_trip_through_pixels_under_1px(S):
    """Draw a box into the image, crop+resize it like the dataset, find it, map it back: < 1 px."""
    r = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        crop = (int(r.integers(0, 60)), int(r.integers(0, 60)), int(r.integers(440, 513)), int(r.integers(440, 513)))
        x0, y0 = int(r.integers(crop[0] + 5, crop[2] - 60)), int(r.integers(crop[1] + 5, crop[3] - 60))
        x1, y1 = x0 + int(r.integers(10, 55)), y0 + int(r.integers(10, 55))
        img = np.zeros((512, 512), np.uint8)
        img[y0:y1, x0:x1] = 255
        det = crop_and_resize(img, crop, S)
        ys, xs = np.nonzero(det >= 128)
        found = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], float)
        back = det_to_full512(found, crop, S)
        worst = max(worst, float(np.abs(back - [[x0, y0, x1, y1]]).max()))
    assert worst < 1.0, worst


def test_crop_clipping_and_drop_outside():
    crop = (100, 100, 300, 300)
    boxes = [[90, 120, 150, 160],     # straddles left edge -> clipped
             [10, 10, 50, 50],        # fully outside -> dropped
             [200, 200, 250, 260]]    # inside
    c, keep = full512_to_crop(boxes, crop)
    assert keep.tolist() == [True, False, True]
    assert c.tolist() == [[0, 20, 50, 60], [100, 100, 150, 160]]


# --- augmentation: property tests ----------------------------------------------------------------

def _assert_valid(boxes, S, min_size):
    assert (boxes >= 0).all() and (boxes[:, [0, 2]] <= S).all() and (boxes[:, [1, 3]] <= S).all()
    assert (boxes[:, 2] - boxes[:, 0] >= min_size).all() and (boxes[:, 3] - boxes[:, 1] >= min_size).all()


@pytest.mark.parametrize("S", [512, 640])
def test_augment_property_random_boxes(S):
    r = np.random.default_rng(2)
    cfg = AugmentConfig(hflip_p=0.5)                      # exercise the flip path too
    img = np.full((S, S), 100, np.uint8)
    for _ in range(400):
        k = int(r.integers(0, 5))
        w, h = r.uniform(2, 150, (2, k))
        x0, y0 = r.uniform(0, S - w), r.uniform(0, S - h)  # includes boxes touching the border
        boxes = np.stack([x0, y0, x0 + w, y0 + h], 1) if k else np.zeros((0, 4))
        out_img, out, params = augment(img, boxes, r, cfg)
        assert out_img.shape == (S, S) and out_img.dtype == np.uint8
        assert len(out) == k                               # never loses a box
        _assert_valid(out, S, cfg.min_size)


@pytest.mark.parametrize("S", [512, 640])
def test_dataset_samples_always_valid(root, no_decisions, S):
    d = ds(root, no_decisions, input_size=S)
    n_checked = 0
    for epoch in range(8):
        d.set_epoch(epoch)
        for i in range(len(d)):
            img, boxes, info = d.load(i)
            _, plain, _ = d.load(i, augment_=False)
            assert img.shape == (S, S)
            assert len(boxes) == len(plain)                 # augmentation keeps every box
            assert (len(boxes) > 0) == (d.samples[i].tb_label == 1)
            _assert_valid(boxes, S, d.cfg.min_box_size)
            n_checked += 1
    assert n_checked == 8 * len(d)


def test_hflip_off_by_default():
    assert AugmentConfig().hflip_p == 0.0


def test_augmentation_reproducible_per_epoch(root, no_decisions):
    d = ds(root, no_decisions)
    a = d.load(3)
    assert np.array_equal(a[0], d.load(3)[0]) and np.array_equal(a[1], d.load(3)[1])
    d.set_epoch(1)
    assert not np.array_equal(a[0], d.load(3)[0])


# --- torch interface -----------------------------------------------------------------------------

@pytest.mark.parametrize("S", [512, 640])
def test_getitem_and_dataloader(root, no_decisions, S):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dl = make_dataloader(DetDataConfig("train", input_size=S), batch_size=4, num_workers=0,
                             paths=DataPaths(root), decisions_dir=no_decisions)
    images, targets = next(iter(dl))
    assert images.shape == (4, 3, S, S) and images.dtype == torch.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    for t in targets:
        assert t["boxes"].dtype == torch.float32 and t["boxes"].shape[1] == 4
        assert (t["labels"] == 0).all() and len(t["labels"]) == len(t["boxes"])
        assert t["input_size"] == S and t["crop_box"].shape == (4,)
