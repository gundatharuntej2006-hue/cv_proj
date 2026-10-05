# Central configuration loader.

from dataclasses import dataclass
from typing import Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    batch_size: int = 16
    image_size: int = 512
    device: str = "cpu"
    alpha_triage: float = 0.10
    tau_low: float = 0.15
    tau_high: float = 0.65


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {"model": ModelConfig().__dict__}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
