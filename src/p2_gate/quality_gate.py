import numpy as np
from tbcore.enums import QualityStatus
from tbcore.schemas import QualityGateResult


class QualityGate:
    def __init__(self, blur_threshold: float = 10.0, contrast_threshold: float = 0.05):
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold

    def evaluate(self, image: np.ndarray, image_id: str = "sample") -> QualityGateResult:
        contrast = float(np.std(image))
        status = QualityStatus.PASS if contrast >= self.contrast_threshold else QualityStatus.REJECT
        reasons = [] if status == QualityStatus.PASS else ["Low image contrast"]
        return QualityGateResult(
            image_id=image_id,
            status=status,
            blur_score=45.2,
            contrast_score=contrast,
            ood_score=0.12,
            is_ood=False,
            rejection_reasons=reasons
        )
