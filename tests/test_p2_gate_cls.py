import numpy as np
from p2_gate.quality_gate import QualityGate
from p2_cls.classifier import TBClassifier
from tbcore.enums import QualityStatus, CXRClass


def test_p2_quality_gate():
    gate = QualityGate(contrast_threshold=0.01)
    img = np.random.uniform(0.1, 0.9, (100, 100)).astype(np.float32)
    res = gate.evaluate(img, "IMG_P2")
    assert res.status == QualityStatus.PASS


def test_p2_classifier():
    clf = TBClassifier()
    img = np.zeros((224, 224), dtype=np.float32)
    res = clf.predict(img, "IMG_P2")
    assert res.predicted_class in [CXRClass.TB, CXRClass.HEALTHY, CXRClass.SICK_NON_TB]
    assert 0.0 <= res.tb_probability_raw <= 1.0
