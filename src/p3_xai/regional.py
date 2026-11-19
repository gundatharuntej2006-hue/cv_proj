"""Differentiable adapter for P2 v2/v3 regional heads, separate from the CLS-based D19.

No claim of validation: approve a revised D19 and run score-reproduction and
randomisation checks on real development images before reporting these maps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RegionalClassifier(nn.Module):
    tb_index = 0

    def __init__(self, backbone, processor, weights, bias):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        weights = torch.as_tensor(np.asarray(weights).copy(), dtype=torch.float32)
        bias = torch.as_tensor(np.asarray(bias).copy(), dtype=torch.float32)
        if weights.ndim != 3 or weights.shape[1] != 1 or bias.shape != weights.shape[:2]:
            raise ValueError("Expected seed weights [S,1,C] and bias [S,1]")
        self.head = nn.Linear(weights.shape[2], weights.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(weights[:, 0])
            self.head.bias.copy_(bias[:, 0])
        self.input_size = int(backbone.config.image_size)
        self.class_names = ["tb"]

    def preprocess(self, image):
        image = np.asarray(image)
        rgb = np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image
        return self.processor(images=[rgb], return_tensors="pt")["pixel_values"]

    def forward(self, x):
        # No inference_mode: Grad-CAM needs gradients through the real score.
        dtype = next(self.backbone.parameters()).dtype
        tokens = self.backbone(pixel_values=x.to(dtype=dtype)).last_hidden_state
        g = self.backbone.config.image_size // self.backbone.config.patch_size
        patches = tokens[:, -g * g:, :].float()
        grid = patches.reshape(len(x), g, g, -1).permute(0, 3, 1, 2)
        regions = F.adaptive_avg_pool2d(grid, 12).permute(0, 2, 3, 1).reshape(len(x), 144, -1)
        regions = regions / regions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # P2's embed() and repool(12,12) each round to float16.
        regions = regions.to(torch.float16).float()
        regions = regions / regions.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        regions = regions.to(torch.float16).float()
        logits = self.head(regions).permute(0, 2, 1)
        # Ensemble score is mean(seed maxima), NOT max(mean region logits).
        return logits.max(dim=-1).values.mean(dim=-1, keepdim=True)


def from_bundle(bundle):
    from p2_cls import pipeline_v2 as p2
    embedder = p2._embedder(bundle)
    model = getattr(embedder, "model", None)
    processor = getattr(embedder, "proc", None)
    if model is None or processor is None:
        raise RuntimeError("P2 embedder must expose its model and proc; no substitute model is loaded")
    device = next(model.parameters()).device
    # Use the served encoder and processor, preserving revision and precision.
    return RegionalClassifier(model, processor, bundle["W"], bundle["b"]).to(device).eval()


def explain(image512, bundle):
    """Use P2's actual lung mask and full-frame geometry; never crop a v2 input."""
    from p2_cls import pipeline_v2 as p2
    from .gradcam import GradCamExplainer
    masked, _ = p2.lungs_only(image512, bundle)
    return GradCamExplainer(from_bundle(bundle))(masked, [0, 0, 512, 512])
