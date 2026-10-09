"""P3-06: run configuration (src/p3_det/configs/*.yaml) and its schedule fingerprint."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..data.augment import AugmentConfig
from ..data.config import INPUT_SIZES

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "src" / "p3_det" / "configs"
DFINE_DIR = REPO / "third_party" / "D-FINE"
SIZE_KEYS = ("run_name", "input_size")         # the only keys allowed to differ between 512 and 640


@dataclass(frozen=True)
class RunConfig:
    raw: dict
    path: Path

    def __getattr__(self, name):
        try:
            return self.raw[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def schedule(self) -> dict:
        return self.raw["schedule"]

    @property
    def seed(self) -> int:
        return int(self.raw["schedule"]["seed"])

    @property
    def augment(self) -> AugmentConfig:
        a = dict(self.raw["data"]["augment"])
        a["scale"] = tuple(a["scale"])
        return AugmentConfig(**a)

    @property
    def schedule_hash(self) -> str:
        """sha256 of everything except run_name / input_size: equal hashes = identical schedule."""
        return schedule_hash(self.raw)


def schedule_hash(raw: dict) -> str:
    rest = {k: v for k, v in copy.deepcopy(raw).items() if k not in SIZE_KEYS}
    return "sha256:" + hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    raw = yaml.safe_load(p.read_text())
    for section in ("model", "schedule", "data", "eval"):
        if section not in raw:
            raise ValueError(f"{p.name}: missing section '{section}'")
    if raw["input_size"] not in INPUT_SIZES:
        raise ValueError(f"{p.name}: input_size must be one of {INPUT_SIZES}")
    if raw["model"]["init"] not in ("coco", "none"):
        raise ValueError(f"{p.name}: model.init must be 'coco' or 'none'")
    s = raw["schedule"]
    if not 0 < s["multiscale_stop_epoch"] <= s["epochs"]:
        raise ValueError(f"{p.name}: multiscale_stop_epoch must be in (0, epochs]")
    return RunConfig(raw, p)
