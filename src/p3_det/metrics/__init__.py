"""P3-05: detection metrics (methodology 12.4), for P3 and P4.

    from p3_det.metrics import ImageGT, ap50, image_level_lesion_sensitivity, froc, subtype_stratified

All boxes in full 512 coordinates; ground truth = all original boxes (D21). See detection.py.
"""
from .detection import (
    ImageGT,
    MetricResult,
    ap50,
    ap50_difference,
    detections_from_c10,
    froc,
    froc_curve,
    ground_truth_from_coco,
    image_level_lesion_sensitivity,
    iou_matrix,
    match_image,
    subtype_stratified,
)
from .stats import bootstrap_ci, bootstrap_settings, wilson_ci

__all__ = [
    "ImageGT", "MetricResult", "ap50", "ap50_difference", "bootstrap_ci", "bootstrap_settings", "detections_from_c10", "froc",
    "froc_curve", "ground_truth_from_coco", "image_level_lesion_sensitivity", "iou_matrix", "match_image",
    "subtype_stratified", "wilson_ci",
]
