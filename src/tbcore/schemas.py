"""Pydantic V2 data validation models enforcing inter-module contracts (C14, C16, etc.)."""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from tbcore.enums import CaseStatus, CXRClass, OperatingPoint, QualityStatus, TriageCategory, ViewOrientation
from tbcore.disclaimer import DISCLAIMER_TEXT


# ---------------------------------------------------------------------------
# C14 CaseResult Models
# ---------------------------------------------------------------------------

class InputBlock(BaseModel):
    format: str = "png"
    phi_tags_removed: int = 0
    error: Optional[str] = None


class GateBlock(BaseModel):
    quality_pass: bool = True
    quality_reasons: List[str] = Field(default_factory=list)
    ood_score: float = 0.0
    ood_threshold: float = 0.241
    gate_pass: bool = True
    view: Optional[str] = None


class SegmentationBlock(BaseModel):
    sanity_pass: bool = True
    fail_reasons: List[str] = Field(default_factory=list)
    crop_box: List[int] = Field(default_factory=lambda: [0, 0, 512, 512])
    fallback_full: bool = False


class ThreeClassProbs(BaseModel):
    healthy: float = 0.0
    sick_non_tb: float = 0.0
    tb: float = 0.0


class ClassifierBlock(BaseModel):
    model_id: str
    head: str = "binary"
    p_tb_raw: float
    p_tb_cal: float
    temperature: float = 1.0
    three_class: Optional[ThreeClassProbs] = None


class DetectedBox(BaseModel):
    xyxy: List[int]
    score: float


class DetectorBlock(BaseModel):
    model_id: str
    conf_threshold: float = 0.42
    n_lesions: int = 0
    boxes: List[DetectedBox] = Field(default_factory=list)


class TriageBlock(BaseModel):
    t_lower: float = 0.18
    t_upper: float = 0.83
    category_pre_link: str = "REFER"
    link_fired: bool = False
    category_final: str = "REFER"
    screen_positive: bool = True
    operating_point: str = "locked_default"


class ExplainabilityBlock(BaseModel):
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None


class VersionsBlock(BaseModel):
    model_version: str = "reg-v1-3b9c"
    split_version: str = "v1"
    decisions_hash: str = "sha256:5e1d"
    git_sha: str = "c0ffee1"


class TimingBlock(BaseModel):
    standardise: int = 0
    gate: int = 0
    segment: int = 0
    classify: int = 0
    detect: int = 0
    explain: int = 0
    total: int = 0


class CaseResult(BaseModel):
    """C14 CaseResult contract schema."""
    schema_version: str = "1.0"
    case_id: str
    status: CaseStatus = CaseStatus.ANALYSED
    input: InputBlock
    gate: Optional[GateBlock] = None
    segmentation: Optional[SegmentationBlock] = None
    classifier: Optional[ClassifierBlock] = None
    detector: Optional[DetectorBlock] = None
    triage: Optional[TriageBlock] = None
    explainability: Optional[ExplainabilityBlock] = None
    versions: VersionsBlock
    timing_ms: TimingBlock = Field(default_factory=TimingBlock)
    disclaimer: str = DISCLAIMER_TEXT


# ---------------------------------------------------------------------------
# C16 Decision JSON Model
# ---------------------------------------------------------------------------

class DecisionRecord(BaseModel):
    """C16 Decision JSON schema."""
    decision_id: str
    owner: str
    title: str
    criterion_preregistered: str
    preregistered_at: str
    candidates: List[Any]
    value: Dict[str, Any]
    evidence: List[str] = Field(default_factory=list)
    data_used: List[str]
    reviewed_by: str
    guide_ack: bool = False
    locked: bool = False
    locked_at: Optional[str] = None
    git_sha: Optional[str] = None


# ---------------------------------------------------------------------------
# Legacy and Auxiliary Schemas for Backwards Compatibility
# ---------------------------------------------------------------------------

class InputCXR(BaseModel):
    image_id: str
    file_path: str
    width: int = Field(default=512, gt=0)
    height: int = Field(default=512, gt=0)
    channels: int = Field(default=1, ge=1, le=3)
    format: str = Field(default="PNG")
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class QualityGateResult(BaseModel):
    image_id: str
    status: QualityStatus
    blur_score: float = 0.0
    contrast_score: float = 0.0
    ood_score: float = 0.0
    is_ood: bool = False
    rejection_reasons: List[str] = Field(default_factory=list)


class ViewheadResult(BaseModel):
    image_id: str
    predicted_orientation: ViewOrientation = ViewOrientation.PA
    confidence: float = Field(default=0.99, ge=0.0, le=1.0)
    is_valid_pa: bool = True


class LungSegmentationResult(BaseModel):
    image_id: str
    crop_box_normalized: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    lung_area_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    mask_path: Optional[str] = None


class ClassProbabilities(BaseModel):
    healthy: float = Field(default=0.0, ge=0.0, le=1.0)
    sick_non_tb: float = Field(default=0.0, ge=0.0, le=1.0)
    tb: float = Field(default=0.0, ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    image_id: str
    probabilities: ClassProbabilities
    predicted_class: CXRClass
    tb_probability_raw: float = Field(ge=0.0, le=1.0)


class LesionBoundingBox(BaseModel):
    bbox_normalized: Tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    lesion_type: str = "SUSPICIOUS_TB"


class DetectionResult(BaseModel):
    image_id: str
    boxes: List[LesionBoundingBox] = Field(default_factory=list)
    total_lesions_found: int = 0


class XAIAttributionResult(BaseModel):
    image_id: str
    heatmap_path: str
    max_activation_coords: Tuple[float, float] = (0.0, 0.0)
    pointing_hit: Optional[bool] = None
    iou_with_detector_boxes: Optional[float] = None


class ConformalTriageResult(BaseModel):
    image_id: str
    triage_category: TriageCategory
    calibrated_tb_prob: float = Field(ge=0.0, le=1.0)
    tau_low: float = Field(default=0.18, ge=0.0, le=1.0)
    tau_high: float = Field(default=0.83, ge=0.0, le=1.0)
    alpha_target: float = Field(default=0.10, ge=0.0, le=1.0)
    risk_bound_satisfied: bool = True
    clinical_recommendation: str = ""


class PipelineResult(BaseModel):
    image_id: str
    timestamp: str
    latency_ms: float = Field(ge=0.0)
    quality_gate: QualityGateResult
    viewhead: Optional[ViewheadResult] = None
    segmentation: Optional[LungSegmentationResult] = None
    classification: Optional[ClassificationResult] = None
    detection: Optional[DetectionResult] = None
    xai: Optional[XAIAttributionResult] = None
    triage: Optional[ConformalTriageResult] = None
