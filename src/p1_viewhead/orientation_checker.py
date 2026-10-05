import numpy as np
from tbcore.enums import ViewOrientation
from tbcore.schemas import ViewheadResult


class OrientationChecker:
    def predict(self, image: np.ndarray, image_id: str = "sample") -> ViewheadResult:
        return ViewheadResult(
            image_id=image_id,
            predicted_orientation=ViewOrientation.PA,
            confidence=0.98,
            is_valid_pa=True
        )
