import numpy as np
from tbcore.enums import CXRClass, TriageCategory, QualityStatus, ViewOrientation
from tbcore.schemas import ClassProbabilities, ClassificationResult
from tbcore.io import save_image_grayscale, load_image_normalized
from tbcore.utils import seed_everything, timed_execution


def test_enums():
    assert CXRClass.TB.value == "TB"
    assert TriageCategory.REFER.value == "REFER"
    assert QualityStatus.PASS.value == "PASS"
    assert ViewOrientation.PA.value == "PA"


def test_probabilities_validation():
    probs = ClassProbabilities(healthy=0.2, sick_non_tb=0.3, tb=0.5)
    cls_res = ClassificationResult(
        image_id="T1",
        probabilities=probs,
        predicted_class=CXRClass.TB,
        tb_probability_raw=0.5
    )
    assert cls_res.predicted_class == CXRClass.TB


def test_io_and_timer(tmp_path):
    seed_everything(123)
    arr = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)
    path = tmp_path / "test.png"
    save_image_grayscale(arr, path)
    loaded = load_image_normalized(path, target_size=(128, 128))
    assert loaded.shape == (128, 128)

    _, elapsed = timed_execution(lambda: sum(range(1000)))
    assert elapsed >= 0.0
