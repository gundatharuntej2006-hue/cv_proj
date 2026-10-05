from typing import Tuple
import numpy as np
from tbcore.schemas import LungSegmentationResult


class CXRPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def segment_and_crop(self, image: np.ndarray, image_id: str = "sample") -> Tuple[np.ndarray, LungSegmentationResult]:
        box = (0.1, 0.1, 0.9, 0.9)
        result = LungSegmentationResult(
            image_id=image_id,
            crop_box_normalized=box,
            lung_area_ratio=0.64,
            mask_path=None
        )
        return image, result
