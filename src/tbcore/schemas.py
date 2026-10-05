# Pydantic V2 data validation models enforcing inter-module contracts.

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from tbcore.enums import CXRClass, QualityStatus, TriageCategory, ViewOrientation


class InputCXR(BaseModel):
    # Incoming CXR metadata container.
    image_id: str
    file_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(default=1, ge=1, le=3)
    format: str = Field(default="PNG")
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class QualityGateResult(BaseModel):
    # P2 Quality gate and Out-Of-Distribution evaluation output.
    image_id: str
    status: QualityStatus
    blur_score: float
    contrast_score: float
    ood_score: float
    is_ood: bool = False
    rejection_reasons: List[str] = Field(default_factory=list)


class ViewheadResult(BaseModel):
    # P1 Radiograph projection view verification output.
    image_id: str
    predicted_orientation: ViewOrientation
    confidence: float = Field(ge=0.0, le=1.0)
    is_valid_pa: bool


class LungSegmentationResult(BaseModel):
    # P1 Bilateral lung field localization bounding box output.
    image_id: str
    crop_box_normalized: Tuple[float, float, float, float]
    lung_area_ratio: float = Field(ge=0.0, le=1.0)
    mask_path: Optional[str] = None


class ClassProbabilities(BaseModel):
    # 3-Class calibrated or uncalibrated softmax distribution.
    healthy: float = Field(ge=0.0, le=1.0)
    sick_non_tb: float = Field(ge=0.0, le=1.0)
    tb: float = Field(ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    # P2 RAD-DINO classifier output.
    image_id: str
    probabilities: ClassProbabilities
    predicted_class: CXRClass
    tb_probability_raw: float = Field(ge=0.0, le=1.0)


class LesionBoundingBox(BaseModel):
    # Individual TB lesion localization.
    bbox_normalized: Tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    lesion_type: str = "SUSPICIOUS_TB"


class DetectionResult(BaseModel):
    # P3 D-FINE lesion bounding-box detector output.
    image_id: str
    boxes: List[LesionBoundingBox] = Field(default_factory=list)
    total_lesions_found: int = 0


class XAIAttributionResult(BaseModel):
    # P3 Grad-CAM visual explanation output.
    image_id: str
    heatmap_path: str
    max_activation_coords: Tuple[float, float]
    pointing_hit: Optional[bool] = None
    iou_with_detector_boxes: Optional[float] = None


class ConformalTriageResult(BaseModel):
    # P4 Conformal Risk Control triage sorting result.
    image_id: str
    triage_category: TriageCategory
    calibrated_tb_prob: float = Field(ge=0.0, le=1.0)
    tau_low: float = Field(default=0.15, ge=0.0, le=1.0)
    tau_high: float = Field(default=0.65, ge=0.0, le=1.0)
    alpha_target: float = Field(default=0.10, ge=0.0, le=1.0)
    risk_bound_satisfied: bool = True
    clinical_recommendation: str = ""


class PipelineResult(BaseModel):
    # Complete aggregated result packet for a single radiograph.
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
