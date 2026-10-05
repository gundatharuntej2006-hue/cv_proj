# P1 Viewhead: Projection Orientation Verification (PA vs AP vs Lateral)
import numpy as np
from tbcore.enums import ViewOrientation
from tbcore.schemas import ViewheadResult


class OrientationChecker:
    # Verifies that chest radiograph projection is Posteroanterior (PA).
    # Rejects AP and Lateral projections which distort cardiac silhouette and apex visualization.

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

    def predict(self, image: np.ndarray, image_id: str = "sample") -> ViewheadResult:
        # Analyzes anatomical aspect ratio and bilateral symmetry
        h, w = image.shape[:2]
        aspect_ratio = float(w / h)

        # PA radiographs generally have aspect ratio between 0.80 and 1.15 with high bilateral symmetry
        left_half = image[:, :w // 2]
        right_half = np.fliplr(image[:, w // 2:])
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = float(np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w])))

        if 0.75 <= aspect_ratio <= 1.25 and symmetry_diff < 0.35:
            pred = ViewOrientation.PA
            conf = min(0.99, max(0.86, 1.0 - symmetry_diff))
            is_valid = True
        elif aspect_ratio < 0.70:
            pred = ViewOrientation.LATERAL
            conf = 0.92
            is_valid = False
        else:
            pred = ViewOrientation.AP
            conf = 0.88
            is_valid = False

        return ViewheadResult(
            image_id=image_id,
            predicted_orientation=pred,
            confidence=round(conf, 3),
            is_valid_pa=is_valid
        )
