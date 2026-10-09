"""P3-04: data paths, locked preprocessing (D08 CLAHE, D09 margin) and the sealed-split guard.

Until P2 locks D08/D09 the adapter runs on PROVISIONAL settings (margin 0.10, CLAHE off, H4).
Once the decision files are locked, their values are used and a conflicting explicit argument
is an error.

Paths follow README 4.2. ``TB_MOCK=1`` reads the same relative paths under ``mock/`` (the
tbcore.paths switch, re-implemented here until tbcore exists).
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROVISIONAL_MARGIN = 0.10
PROVISIONAL_CLAHE = False
MARGINS = (0.0, 0.05, 0.10, 0.15)
INPUT_SIZES = (512, 640)

OPEN_SPLITS = ("train", "development", "calibration")
SEALED_SPLITS = ("internal_test", "challenge_test", "external_test")
NEVER_SPLITS = ("excluded_duplicate",)


class SealedSplitError(PermissionError):
    """Raised when a sealed split is requested without the gate's unlock token."""


def check_split(split: str, dataset: str = "tbx11k") -> None:
    """Allow open splits; sealed splits only if tbcore.guard says so. Fails closed."""
    if split in OPEN_SPLITS:
        return
    if split in SEALED_SPLITS:
        try:
            from tbcore import guard  # P4 (P4-01, P4-09)
        except ImportError:
            raise SealedSplitError(
                f"split '{split}' is sealed and tbcore.guard is not available; refusing") from None
        # tbcore.guard raises if no single-use token exists for this split (README 4.1).
        # The function name is a placeholder until P4 publishes the guard API; if it differs,
        # this still refuses (fails closed) instead of loading sealed data.
        require = getattr(guard, "require_unlocked", None)
        if require is None:
            raise SealedSplitError(f"tbcore.guard has no require_unlocked(); refusing sealed split '{split}'")
        require(split=split, dataset=dataset, purpose="p3_det")
        return
    raise SealedSplitError(f"split '{split}' may not be loaded by the detector adapter")


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def default(cls) -> "DataPaths":
        base = Path(os.environ.get("TB_DATA_ROOT", REPO))
        return cls(base / "mock" if os.environ.get("TB_MOCK") == "1" else base)

    @property
    def splits(self) -> Path:
        return self.root / "data/splits/tbx11k_splits_v1.csv"

    @property
    def manifest(self) -> Path:
        return self.root / "data/processed/v1/manifest.parquet"

    @property
    def lung_roi(self) -> Path:
        return self.root / "data/processed/v1/lung_roi.parquet"

    def coco(self, split: str, margin: float) -> Path:
        return self.root / f"data/boxes/v1/coco_{split}_m{margin:.2f}.json"


def _locked_value(path: Path, key: str):
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d["value"][key] if d.get("locked") else None


def resolve_preprocessing(margin: float | None = None, clahe: bool | None = None,
                          decisions_dir: Path | None = None) -> tuple[float, bool, bool]:
    """Return (margin, clahe, provisional). Locked D09/D08 win; explicit conflicts are errors."""
    ddir = decisions_dir or REPO / "decisions"
    locked_margin = _locked_value(ddir / "p2_crop_margin.json", "margin")
    locked_clahe = _locked_value(ddir / "p2_clahe.json", "clahe")
    if locked_margin is not None and margin is not None and float(margin) != float(locked_margin):
        raise ValueError(f"margin {margin} conflicts with locked D09 margin {locked_margin}")
    if locked_clahe is not None and clahe is not None and bool(clahe) != bool(locked_clahe):
        raise ValueError(f"clahe={clahe} conflicts with locked D08 clahe={locked_clahe}")
    m = float(locked_margin if locked_margin is not None else (PROVISIONAL_MARGIN if margin is None else margin))
    c = bool(locked_clahe if locked_clahe is not None else (PROVISIONAL_CLAHE if clahe is None else clahe))
    provisional = locked_margin is None or locked_clahe is None
    if m not in MARGINS:
        raise ValueError(f"margin must be one of {MARGINS}, got {m}")
    if provisional:
        warnings.warn(f"PROVISIONAL preprocessing (D08/D09 not locked): margin={m}, clahe={c}", stacklevel=2)
    return m, c, provisional
