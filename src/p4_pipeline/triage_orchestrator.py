import time
from datetime import datetime, timezone
import numpy as np
from tbcore.schemas import InputCXR, PipelineResult
from tbcore.enums import QualityStatus
from p2_gate.quality_gate import QualityGate
from p1_viewhead.orientation_checker import OrientationChecker
from p1_data.preprocessing import CXRPreprocessor
from p2_cls.classifier import TBClassifier
from p3_det.dfine_detector import DFINEDetector
from p3_xai.gradcam import GradCAMExplainer
from p4_stats.calibration import TemperatureScaler
from p4_pipeline.conformal_triage import ConformalTriageEngine


class TriagePipelineOrchestrator:
    def __init__(self):
        self.gate = QualityGate()
        self.viewhead = OrientationChecker()
        self.preproc = CXRPreprocessor()
        self.classifier = TBClassifier()
        self.detector = DFINEDetector()
        self.xai = GradCAMExplainer()
        self.scaler = TemperatureScaler()
        self.triage_engine = ConformalTriageEngine()

    def process(self, image: np.ndarray, meta: InputCXR) -> PipelineResult:
        t0 = time.perf_counter()

        q_res = self.gate.evaluate(image, meta.image_id)
        if q_res.status == QualityStatus.REJECT:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                image_id=meta.image_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=elapsed,
                quality_gate=q_res
            )

        v_res = self.viewhead.predict(image, meta.image_id)
        lung_crop, seg_res = self.preproc.segment_and_crop(image, meta.image_id)
        cls_res = self.classifier.predict(lung_crop, meta.image_id)
        det_res = self.detector.detect(lung_crop, meta.image_id)
        xai_res = self.xai.generate(self.classifier, lung_crop, meta.image_id)
        calib_prob = self.scaler.calibrate(cls_res.tb_probability_raw)
        triage_res = self.triage_engine.triage(calib_prob, meta.image_id)

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            image_id=meta.image_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=elapsed,
            quality_gate=q_res,
            viewhead=v_res,
            segmentation=seg_res,
            classification=cls_res,
            detection=det_res,
            xai=xai_res,
            triage=triage_res
        )
