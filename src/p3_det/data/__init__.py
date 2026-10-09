"""P3-04: detector dataset adapter (C01 + C04 + C05 + C06 -> D-FINE-ready samples)."""
from .augment import AugmentConfig, augment
from .config import DataPaths, SealedSplitError, check_split, resolve_preprocessing
from .dataset import DetDataConfig, TBDetectionDataset, collate, make_dataloader
from .geometry import det_to_full512, full512_to_crop, full512_to_det

__all__ = [
    "AugmentConfig", "DataPaths", "DetDataConfig", "SealedSplitError", "TBDetectionDataset", "augment",
    "check_split", "collate", "det_to_full512", "full512_to_crop", "full512_to_det", "make_dataloader",
    "resolve_preprocessing",
]
