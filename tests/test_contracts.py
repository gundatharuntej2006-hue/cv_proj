import json
from pathlib import Path
from tbcore.schemas import PipelineResult, QualityGateResult, ClassificationResult, ConformalTriageResult


def test_sample_packet_contract():
    with open("mock/sample_pipeline_packet.json", "r") as f:
        data = json.load(f)

    packet = PipelineResult(**data)
    assert packet.image_id == "CXR_SYNTH_001"
    assert packet.quality_gate.status.value == "PASS"
    assert packet.classification.predicted_class.value == "TB"
    assert packet.triage.triage_category.value == "TB"


def test_json_schemas_exist():
    schema_dir = Path("contracts")
    expected = [
        "input_cxr.schema.json",
        "quality_gate.schema.json",
        "viewhead.schema.json",
        "segmentation.schema.json",
        "classification.schema.json",
        "detection.schema.json",
        "xai_attribution.schema.json",
        "conformal_triage.schema.json",
        "pipeline_result.schema.json",
    ]
    for s in expected:
        f = schema_dir / s
        assert f.exists(), f"Missing schema: {s}"
        with open(f) as fp:
            data = json.load(fp)
            assert "$schema" in data
