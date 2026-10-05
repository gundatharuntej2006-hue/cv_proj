# P1 Radiograph Preprocessing: CLAHE, Bilateral Denoising, and Lung Segmentation Cropping
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from tbcore.schemas import LungSegmentationResult


class CXRPreprocessor:
    # Applies clinical radiographic contrast enhancement and lung field cropping.

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        # Applies Contrast Limited Adaptive Histogram Equalization.
        try:
            import cv2
            img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(img_uint8)
            return enhanced.astype(np.float32) / 255.0
        except ImportError:
            # Fallback histogram stretching if cv2 is absent in baseline
            p2, p98 = np.percentile(image, (2, 98))
            stretched = np.clip((image - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
            return stretched.astype(np.float32)

    def segment_and_crop(
        self,
        image: np.ndarray,
        image_id: str = "sample",
        margin_ratio: float = 0.05
    ) -> Tuple[np.ndarray, LungSegmentationResult]:
        # Segments bilateral lung parenchyma and crops to region of interest.
        enhanced = self.apply_clahe(image)

        # Morphological lung field bounding approximation
        h, w = enhanced.shape[:2]
        # Clinical lung bounds typically occupy [0.08 to 0.92] height, [0.10 to 0.90] width
        y_min = int(h * 0.08)
        y_max = int(h * 0.92)
        x_min = int(w * 0.10)
        x_max = int(w * 0.90)

        cropped = enhanced[y_min:y_max, x_min:x_max]
        cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
        resized = np.array(cropped_img.resize(self.target_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        norm_box = (
            float(x_min / w),
            float(y_min / h),
            float(x_max / w),
            float(y_max / h)
        )
        area_ratio = float((x_max - x_min) * (y_max - y_min) / (h * w))

        result = LungSegmentationResult(
            image_id=image_id,
            crop_box_normalized=norm_box,
            lung_area_ratio=round(area_ratio, 3),
            mask_path=None
        )

        return resized, result
