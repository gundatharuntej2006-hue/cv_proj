import numpy as np
from p1_viewhead.orientation_checker import OrientationChecker
from p1_data.preprocessing import CXRPreprocessor
from tbcore.enums import ViewOrientation


def test_p1_viewhead():
    checker = OrientationChecker()
    img = np.zeros((100, 100), dtype=np.float32)
    res = checker.predict(img, "IMG_P1")
    assert res.predicted_orientation == ViewOrientation.PA
    assert res.is_valid_pa is True


def test_p1_preprocessing():
    prep = CXRPreprocessor()
    img = np.ones((512, 512), dtype=np.float32)
    crop, seg = prep.segment_and_crop(img, "IMG_P1")
    assert crop.shape == (512, 512)
    assert seg.crop_box_normalized == (0.1, 0.1, 0.9, 0.9)
