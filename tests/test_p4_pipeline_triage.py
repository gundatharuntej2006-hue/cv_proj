from pathlib import Path
import numpy as np
from p4_stats.calibration import TemperatureScaler
from p4_pipeline.conformal_triage import ConformalTriageEngine
from p4_pipeline.triage_orchestrator import TriagePipelineOrchestrator
from tbcore.schemas import InputCXR
from tbcore.enums import TriageCategory


def test_p4_calibration():
    scaler = TemperatureScaler(temperature=1.2)
    calib = scaler.calibrate(0.8)
    assert 0.0 <= calib <= 1.0


def test_p4_conformal_triage():
    engine = ConformalTriageEngine(tau_low=0.2, tau_high=0.7)
    res_tb = engine.triage(0.85)
    assert res_tb.triage_category == TriageCategory.TB
    res_not_tb = engine.triage(0.10)
    assert res_not_tb.triage_category == TriageCategory.NOT_TB
    res_refer = engine.triage(0.45)
    assert res_refer.triage_category == TriageCategory.REFER


def test_p4_orchestrator(mock_cxr_image_path):
    orch = TriagePipelineOrchestrator()
    img = np.random.uniform(0.1, 0.9, (512, 512)).astype(np.float32)
    meta = InputCXR(
        image_id="ORCH_001",
        file_path=str(mock_cxr_image_path),
        width=512,
        height=512,
        channels=1,
        format="PNG"
    )
    result = orch.process(img, meta)
    assert result.image_id == "ORCH_001"
    assert result.latency_ms >= 0.0
    assert result.triage is not None
