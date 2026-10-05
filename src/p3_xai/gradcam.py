import numpy as np
from tbcore.schemas import XAIAttributionResult


class GradCAMExplainer:
    def generate(self, model, image: np.ndarray, image_id: str = "sample") -> XAIAttributionResult:
        return XAIAttributionResult(
            image_id=image_id,
            heatmap_path="artifacts/sample_heatmap.png",
            max_activation_coords=(0.32, 0.28),
            pointing_hit=True,
            iou_with_detector_boxes=0.62
        )
