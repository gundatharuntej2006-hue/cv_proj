# -*- coding: utf-8 -*-
"""R2, classification half: the frozen entry points Person 2 owes the rest of the team.

    classify(img_for_analysis) -> ClsResult
    classifier_module()        -> torch.nn.Module

STUB UNTIL W7. The signatures here are final and frozen at `contracts-v1`; the weights
are not. Today this is the real RAD-DINO backbone with a randomly initialised head, so
Person 3 can build Grad-CAM (P3-09) and Person 4 can build `run_case` (P4-10) in Week 1
instead of waiting for D11 in Week 6. Every result carries `is_stub=True`, and P4-09's
gate must refuse any reported number produced while that flag is set.

What is already real, and will not change when the head is trained:
  - the tensor shapes and the preprocessing,
  - which module Grad-CAM hooks (`target_layer`) and how ViT tokens fold back into a
    patch grid (`reshape_transform`),
  - the fact that gradients reach the activations at all, which on a frozen backbone
    they do not by default. See `classifier_module.__doc__`.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

MODEL_ID = "microsoft/rad-dino"
REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"   # pinned in D15
STUB_MODEL_ID = "stub_radino_randomhead_v0"
STUB_SEED = 20260929

_CACHE = {}


@dataclass
class ClsResult:
    """R2 return type. `three_class` is (healthy, sick_non_tb, tb) or None for a binary head."""
    model_id: str
    head: str                                  # "binary" | "three_class"
    logits: List[float]
    p_tb_raw: float
    three_class: Optional[List[float]] = None
    is_stub: bool = field(default=True)


def _load(pretrained=True):
    """Lazy, cached. Never at import time: Person 4's CI imports this module."""
    key = ("backbone", pretrained)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    if pretrained:
        model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION)
    else:
        # same architecture, random weights: lets CI run the shape and gradient tests
        # without pulling 330 MB of checkpoint
        model = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_ID, revision=REVISION))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)                # frozen backbone, per the methodology
    torch.manual_seed(STUB_SEED)               # a deterministic random head
    head = torch.nn.Linear(model.config.hidden_size, 2)
    _CACHE[key] = (proc, model, head)
    return _CACHE[key]


def _as_rgb(img):
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def preprocess(img_for_analysis, pretrained=True):
    """uint8 image (H, W) or (H, W, 3) -> pixel_values [1, 3, S, S]. Use this rather than
    rolling your own resize: the crop-to-518 path must match between P2, P3 and P4 or the
    heatmap will not line up with the boxes."""
    proc, _, _ = _load(pretrained)
    return proc(images=_as_rgb(img_for_analysis), return_tensors="pt")["pixel_values"]


def classify(img_for_analysis, pretrained=True):
    """R2. Returns a ClsResult; the probability is meaningless until the head is trained."""
    import torch
    _, model, head = _load(pretrained)
    x = preprocess(img_for_analysis, pretrained)
    with torch.inference_mode():
        out = model(pixel_values=x)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        logits = head(feat.float())[0]
        p = torch.softmax(logits, dim=-1)[1]
    return ClsResult(model_id=STUB_MODEL_ID, head="binary",
                     logits=[float(v) for v in logits], p_tb_raw=float(p),
                     three_class=None, is_stub=True)


def classifier_module(pretrained=True):
    """R2. The module Grad-CAM hooks (P3-09).

    Two things about it are not optional and cost a day each to rediscover:

    1. **The backbone is frozen, so its activations carry no gradient by default.** With
       every parameter at requires_grad=False and an ordinary input tensor, the output of
       the last block is a leaf with requires_grad=False: `retain_grad()` raises, and if
       you swallow that exception the heatmap is silently all zeros. This module fixes it
       itself, in a forward hook on the target layer, so callers do not have to remember.
       Measured in P2-01: the forward plus backward path peaks at 0.84 GB, well inside the
       5.5 GB budget.
    2. **ViT tokens are not a picture.** The last hidden state is [B, 1 + G*G, C] for a
       518-pixel input at patch 14, so G = 37 and there are 1370 tokens. Grad-CAM needs
       [B, C, G, G]. Use `reshape_transform`; it works out the number of prefix tokens
       from the config rather than assuming the CLS token is the only one.
    """
    import torch
    _, model, head = _load(pretrained)

    class StubClassifier(torch.nn.Module):
        is_stub = True
        model_id = STUB_MODEL_ID

        def __init__(self):
            super().__init__()
            self.backbone = model
            self.head = head
            cfg = model.config
            self.image_size = int(getattr(cfg, "image_size", 518))
            self.patch_size = int(getattr(cfg, "patch_size", 14))
            self.grid = self.image_size // self.patch_size
            self.hidden_size = int(cfg.hidden_size)
            self.target_layer = model.encoder.layer[-1]
            self._hook = self.target_layer.register_forward_hook(self._keep_grad)

        @staticmethod
        def _keep_grad(_module, _inputs, output):
            act = output[0] if isinstance(output, tuple) else output
            if not act.requires_grad:          # frozen backbone: make it a differentiable leaf
                act.requires_grad_(True)
            return output

        def reshape_transform(self, tensor):
            """[B, T, C] -> [B, C, G, G], dropping however many prefix tokens there are."""
            b, t, c = tensor.shape
            n_prefix = t - self.grid * self.grid
            if n_prefix < 0:
                raise ValueError("unexpected token count %d for grid %d" % (t, self.grid))
            return tensor[:, n_prefix:, :].reshape(b, self.grid, self.grid, c).permute(0, 3, 1, 2)

        def forward(self, pixel_values):
            out = self.backbone(pixel_values=pixel_values)
            feat = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
            return self.head(feat.float())      # [B, 2]; index 1 is the TB logit

        def preprocess(self, img_for_analysis):
            return preprocess(img_for_analysis, pretrained)

    return StubClassifier()
