"""R3 `detect` (P3, P3-08): the trained D-FINE TB-lesion detector.

Frozen signature (README 4.3): detect(img_for_analysis, crop_box) -> DetResult.

    img_for_analysis : uint8 512x512 image as produced by R1 (CLAHE per D08 already applied);
                       grayscale HxW, or HxWxC (converted to grayscale like the training data).
    crop_box         : [x0, y0, x1, y1] lung crop in 512 coordinates (end-exclusive), from R1/C05.

Everything model-specific is READ from the locked decision files, never hard-coded:
    D16 decisions/p3_det_input_size.json   -> input size
    D17 decisions/p3_det_checkpoint.json   -> run (architecture via run_meta.json) + checkpoint SHA-256
    D18 decisions/p3_det_conf_threshold.json -> confidence threshold for n_lesions_above_thr
The checkpoint file's SHA-256 must equal D17's; its EMA weights are used. The model is loaded once
per process and cached.

Environment:
    TB_MOCK=1                 deterministic fake detector (the P3-03 stub) for consumers' tests
    TB_DET_CHECKPOINT         path to ckpt_best.pt (default artifacts/p3/det/checkpoints/<D17 run>/ckpt_best.pt)
    TB_DET_DEVICE             cuda / cpu (default: cuda if available)
    TB_DECISIONS_DIR, TB_DET_RUNS_DIR   alternative locations (tests)
    TB_DET_ALLOW_UNLOCKED=1   accept unlocked D16-D18 (development only; never for reported results)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ._stub import STUB_CONF_THRESHOLD, STUB_MODEL_ID, detect_stub
from ._stub import stub_rng as _stub_rng  # noqa: F401  (used by the p3_xai stub)
from .result import DetResult, check_inputs
from .result import check_inputs as _check_inputs  # noqa: F401  (used by the p3_xai stub)

IS_STUB = False
MIN_SCORE = 0.01
REPO = Path(__file__).resolve().parents[2]

__all__ = ["DetResult", "detect", "detector_info", "IS_STUB"]


def _mock() -> bool:
    return os.environ.get("TB_MOCK") == "1"


@dataclass(frozen=True)
class Settings:
    input_size: int
    conf_threshold: float
    run_dir: str
    epoch: int
    checkpoint: Path
    checkpoint_sha256: str
    run_meta: dict
    locked: bool

    @property
    def model_id(self) -> str:
        return f"{self.run_meta['run_name']}_e{self.epoch:03d}_{self.checkpoint_sha256[:8]}"


def _read(path: Path, allow_unlocked: bool) -> dict:
    if not path.exists():
        raise RuntimeError(f"{path} missing: the detector is not selected yet (D16-D18)")
    d = json.loads(path.read_text())
    if not d.get("value"):
        raise RuntimeError(f"{path.name} has no result yet")
    if not d.get("locked") and not allow_unlocked:
        raise RuntimeError(f"{path.name} ({d.get('decision_id')}) is not locked; refusing to serve an unlocked model")
    return d


def load_settings() -> Settings:
    ddir = Path(os.environ.get("TB_DECISIONS_DIR", REPO / "decisions"))
    runs = Path(os.environ.get("TB_DET_RUNS_DIR", REPO / "artifacts" / "p3" / "det" / "runs"))
    allow = os.environ.get("TB_DET_ALLOW_UNLOCKED") == "1"
    d16 = _read(ddir / "p3_det_input_size.json", allow)
    d17 = _read(ddir / "p3_det_checkpoint.json", allow)
    d18 = _read(ddir / "p3_det_conf_threshold.json", allow)
    size = int(d16["value"]["input_size"])
    v17 = d17["value"]
    if int(v17["input_size"]) != size:
        raise RuntimeError(f"D17 checkpoint is {v17['input_size']} px but D16 chose {size}")
    meta = json.loads((runs / v17["run_dir"] / "run_meta.json").read_text())
    if int(meta["input_size"]) != size:
        raise RuntimeError("run_meta.json input size disagrees with D16")
    ckpt = Path(os.environ.get("TB_DET_CHECKPOINT",
                               REPO / "artifacts" / "p3" / "det" / "checkpoints" / v17["run_dir"] / "ckpt_best.pt"))
    return Settings(size, float(d18["value"]["conf_threshold"]), v17["run_dir"], int(v17["epoch"]), ckpt,
                    v17["checkpoint"]["sha256"], meta, all(d.get("locked") for d in (d16, d17, d18)))


class _Detector:
    def __init__(self, settings: Settings):
        import torch

        from .train import dfine_bridge as bridge
        from .train.config import RunConfig

        self.s = settings
        if not settings.checkpoint.exists():
            raise RuntimeError(f"checkpoint {settings.checkpoint} not found (download ckpt_best.pt of "
                               f"{settings.run_dir} from Drive, or set TB_DET_CHECKPOINT)")
        h = hashlib.sha256()
        with open(settings.checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != settings.checkpoint_sha256:
            raise RuntimeError(f"{settings.checkpoint} SHA-256 does not match D17; refusing to load")
        dev = os.environ.get("TB_DET_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.amp = self.device.type == "cuda"
        cfg = RunConfig(settings.run_meta["config"], Path("run_meta.json"))
        self.model, self.postprocessor = bridge.build_inference(cfg, self.device)
        ckpt = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
        ema = (ckpt.get("extra") or {}).get("ema")
        self.model.load_state_dict(ema["module"] if ema else ckpt["model"])
        self.model.eval()
        self._infer = bridge.infer
        self._lock = threading.Lock()

    def __call__(self, img: np.ndarray, crop_box: list[int]) -> DetResult:
        from .data.dataset import crop_and_resize
        if img.dtype != np.uint8:
            raise ValueError(f"img_for_analysis must be uint8 (R1 output), got {img.dtype}")
        gray = img if img.ndim == 2 else np.asarray(Image.fromarray(np.ascontiguousarray(img[..., :3])).convert("L"))
        det_img = crop_and_resize(gray, crop_box, self.s.input_size)
        with self._lock:
            (rows,) = self._infer(self.model, self.postprocessor, [det_img], [crop_box], self.s.input_size,
                                  self.device, self.amp, MIN_SCORE)
        scores = rows[:, 4].astype(np.float32)
        return DetResult(boxes_full512=rows[:, :4].astype(np.float32).reshape(-1, 4), scores=scores,
                         n_lesions_above_thr=int((rows[:, 4] >= self.s.conf_threshold).sum()))


_DETECTOR: _Detector | None = None
_DETECTOR_LOCK = threading.Lock()


def _detector() -> _Detector:
    global _DETECTOR
    if _DETECTOR is None:
        with _DETECTOR_LOCK:
            if _DETECTOR is None:
                _DETECTOR = _Detector(load_settings())
    return _DETECTOR


def reset_cache() -> None:
    """Forget the loaded model (tests; or after the decision files changed)."""
    global _DETECTOR
    _DETECTOR = None


def detector_info() -> dict:
    """model_id, input_size, conf_threshold for CaseResult's detector block (C14)."""
    if _mock():
        return {"model_id": STUB_MODEL_ID, "input_size": 512, "conf_threshold": STUB_CONF_THRESHOLD, "mock": True}
    s = _detector().s
    return {"model_id": s.model_id, "input_size": s.input_size, "conf_threshold": s.conf_threshold,
            "checkpoint_sha256": s.checkpoint_sha256, "locked": s.locked, "mock": False}


def detect(img_for_analysis, crop_box) -> DetResult:
    """Detect TB lesions. Boxes in full 512 coordinates inside crop_box; every box with score >= 0.01."""
    img, box = check_inputs(img_for_analysis, crop_box)
    if _mock():
        return detect_stub(img, box)
    return _detector()(img, box)
