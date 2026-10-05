# P1 Data Ingestion: TBX11K, Shenzhen, and Montgomery CXR Datasets
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from tbcore.enums import CXRClass
from tbcore.io import load_image_normalized


class TBDatasetLoader:
    # Ingests and formats benchmarks: TBX11K (11.2k cases), Shenzhen, and Montgomery.

    def __init__(self, root_dir: str = "data", split: str = "train", target_size: Tuple[int, int] = (512, 512)):
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = target_size
        self.samples: List[Dict[str, Any]] = []

    def load_manifest(self, dataset_name: str) -> List[Dict[str, Any]]:
        manifest_path = self.root_dir / dataset_name / f"{self.split}_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
        return self.samples

    def add_sample(self, image_id: str, file_path: str, label: CXRClass, bbox: Optional[List[float]] = None):
        self.samples.append({
            "image_id": image_id,
            "file_path": file_path,
            "label": label.value,
            "bbox": bbox
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_path = item["file_path"]
        if Path(image_path).exists():
            image = load_image_normalized(image_path, self.target_size)
        else:
            image = np.zeros(self.target_size, dtype=np.float32)

        return {
            "image_id": item["image_id"],
            "image": image,
            "label": CXRClass(item["label"]),
            "bbox": item.get("bbox")
        }
