import numpy as np
import os
from p3_ops._tbcore_adapter import legacy_schema

DetectionResult = legacy_schema("DetectionResult")
LesionBoundingBox = legacy_schema("LesionBoundingBox")


class DFINEDetector:
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path

    def detect(self, image: np.ndarray, image_id: str = "sample", *, crop_box=None) -> DetectionResult:
        from . import api
        if crop_box is None and os.environ.get("TB_MOCK") != "1":
            raise ValueError("Legacy detector requires the locked D09 crop_box; use p3_det.api.detect")
        if self.checkpoint_path:
            raise ValueError("Select checkpoints through D17 and TB_DET_CHECKPOINT, with SHA verification")
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.size and img.max() <= 1 else img, 0, 255).astype(np.uint8)
        result = api.detect(img, crop_box or [0, 0, 512, 512])
        threshold = api.detector_info()["conf_threshold"]
        boxes = [LesionBoundingBox(bbox_normalized=tuple(float(v) / 512 for v in b), confidence=float(s))
                 for b, s in zip(result.boxes_full512, result.scores) if s >= threshold]
        return DetectionResult(
            image_id=image_id,
            boxes=boxes,
            total_lesions_found=len(boxes)
        )
