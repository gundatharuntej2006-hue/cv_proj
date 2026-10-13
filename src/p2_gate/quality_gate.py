# -*- coding: utf-8 -*-
"""R2 quality gate (pipeline-facing entry point).

Keeps the signature P4's pipeline imports (`QualityGate().evaluate(image, image_id)
-> QualityGateResult`) while delegating the actual rules to `p2_gate.quality`, whose
thresholds are derived from a synthetic degradation sweep (P2-05, decision D07) rather
than guessed. Reasons are mapped to the C09 vocabulary.
"""
import numpy as np
from tbcore.enums import QualityStatus
from tbcore.schemas import QualityGateResult

from p2_gate import quality as _q
from p2_gate.ood_detector import OODDetector


class QualityGate:
    def __init__(self, blur_threshold: float = 10.0, contrast_threshold: float = 0.05):
        # kept for API compatibility; the real rules live in p2_gate.quality (D07)
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self._ood = OODDetector()

    def evaluate(self, image: np.ndarray, image_id: str = "sample") -> QualityGateResult:
        report = _q.assess(image)
        ood_score = float(self._ood.score(image))
        status = QualityStatus.PASS if report.passed else QualityStatus.REJECT
        return QualityGateResult(
            image_id=image_id,
            status=status,
            blur_score=float(report.stats.get("std", 0.0)),
            contrast_score=float(report.stats.get("std", 0.0)),
            ood_score=ood_score,
            is_ood=False,                      # real OOD gate is p2_gate.ood (needs a C07 bank)
            rejection_reasons=list(report.reasons),
        )
