from typing import List, Dict, Any
from pathlib import Path


class TBDatasetLoader:
    def __init__(self, root_dir: str = "data"):
        self.root_dir = Path(root_dir)

    def load_manifest(self, dataset_name: str) -> List[Dict[str, Any]]:
        manifest_path = self.root_dir / dataset_name / "manifest.json"
        if not manifest_path.exists():
            return []
        return []
