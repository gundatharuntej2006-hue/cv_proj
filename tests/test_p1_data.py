# P1 Comprehensive Unit Tests
import numpy as np
from p1_data.dataset import TBDatasetLoader
from p1_data.preprocessing import CXRPreprocessor
from p1_viewhead.orientation_checker import OrientationChecker
from tbcore.enums import ViewOrientation, CXRClass


def test_tb_dataset_loader():
    loader = TBDatasetLoader(split="train")
    assert len(loader) == 0
    loader.add_sample("SAMPLE_01", "non_existent.png", CXRClass.TB, [0.1, 0.1, 0.5, 0.5])
    assert len(loader) == 1
    sample = loader[0]
    assert sample["image_id"] == "SAMPLE_01"
    assert sample["label"] == CXRClass.TB
    assert sample["image"].shape == (512, 512)


def test_cxr_preprocessor_clahe():
    prep = CXRPreprocessor()
    img = np.random.uniform(0.1, 0.9, (256, 256)).astype(np.float32)
    enhanced = prep.apply_clahe(img)
    assert enhanced.shape == (256, 256)
    assert 0.0 <= enhanced.min() and enhanced.max() <= 1.0


def test_cxr_preprocessor_segmentation_crop():
    prep = CXRPreprocessor()
    img = np.random.uniform(0.1, 0.9, (512, 512)).astype(np.float32)
    crop, res = prep.segment_and_crop(img, "IMG_CROP")
    assert crop.shape == (512, 512)
    assert res.image_id == "IMG_CROP"
    assert len(res.crop_box_normalized) == 4
    assert res.lung_area_ratio > 0.0


def test_orientation_checker_pa():
    checker = OrientationChecker()
    # Bilateral symmetric image simulating PA
    img = np.zeros((512, 512), dtype=np.float32)
    img[:, :256] = 0.5
    img[:, 256:] = 0.5
    res = checker.predict(img, "PA_TEST")
    assert res.predicted_orientation == ViewOrientation.PA
    assert res.is_valid_pa is True


def test_orientation_checker_lateral():
    checker = OrientationChecker()
    # Tall skinny aspect ratio simulating lateral
    img = np.zeros((600, 300), dtype=np.float32)
    res = checker.predict(img, "LAT_TEST")
    assert res.predicted_orientation == ViewOrientation.LATERAL
    assert res.is_valid_pa is False
