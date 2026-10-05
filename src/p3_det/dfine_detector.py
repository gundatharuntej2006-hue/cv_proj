import numpy as np
from tbcore.schemas import DetectionResult, LesionBoundingBox


class DFINEDetector:
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path

    def detect(self, image: np.ndarray, image_id: str = "sample") -> DetectionResult:
        box = LesionBoundingBox(
            bbox_normalized=(0.25, 0.20, 0.45, 0.40),
            confidence=0.88,
            lesion_type="CAVITY"
        )
        return DetectionResult(
            image_id=image_id,
            boxes=[box],
            total_lesions_found=1
        )
