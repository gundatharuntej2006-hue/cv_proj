"""Compatibility for the original demo API. Real inference requires a real classifier."""
import hashlib
import os

import numpy as np


class GradCAMExplainer:
    def generate(self, classifier, image, image_id="sample"):
        from . import api
        from .gradcam import GradCamExplainer
        from p3_ops._tbcore_adapter import legacy_schema
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.size and img.max() <= 1 else img, 0, 255).astype(np.uint8)
        if os.environ.get("TB_MOCK") == "1":
            heatmap = api.explain(img, [0, 0, 512, 512])
        else:
            import torch
            if not isinstance(classifier, torch.nn.Module):
                raise RuntimeError("A real differentiable classifier is required; no placeholder heatmap")
            heatmap = GradCamExplainer(classifier)(img, [0, 0, 512, 512])
        cid = hashlib.sha256(str(image_id).encode()).hexdigest()[:12]
        path = api.runs_root() / "legacy" / f"c_{cid}_cam.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, heatmap)
        y, x = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
        return legacy_schema("XAIAttributionResult")(image_id=image_id, heatmap_path=str(path),
                                                      max_activation_coords=(float(x) / 512, float(y) / 512))
