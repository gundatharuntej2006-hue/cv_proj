import numpy as np
from tbcore.enums import CXRClass
from tbcore.schemas import ClassificationResult, ClassProbabilities


class TBClassifier:
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path

    def predict(self, lung_crop: np.ndarray, image_id: str = "sample") -> ClassificationResult:
        probs = ClassProbabilities(healthy=0.15, sick_non_tb=0.10, tb=0.75)
        return ClassificationResult(
            image_id=image_id,
            probabilities=probs,
            predicted_class=CXRClass.TB,
            tb_probability_raw=probs.tb
        )
