# -*- coding: utf-8 -*-
"""R2, classification half: the frozen entry points Person 2 owes the rest of the team.

    classify(img_for_analysis) -> ClsResult
    classifier_module()        -> torch.nn.Module

NO LONGER A STUB. If a trained head is present at `artifacts/p2/bundle/`, both entry points
load it and return `is_stub=False`, so Person 4's P4-09 gate will accept the numbers and
Person 3's Grad-CAM runs against the head that is actually being reported. With no bundle on
disk — a fresh clone, or CI — the Week-1 stub behaviour is unchanged: a deterministic random
head and `is_stub=True`. Nothing about the signatures moved; they are still frozen at
`contracts-v1`.

Two things it is easy to get wrong when wiring a real head in here, both of which produce
plausible-looking but incorrect probabilities rather than an error:

  1. **The head was trained on L2-normalised features.** `embed.Embedder.embed` normalises
     before writing C07, so the head's weights live in that space. Feeding it a raw
     `pooler_output` yields confident nonsense. `_features` normalises; do not remove it.
  2. **Grad-CAM must differentiate the same function that is reported.** The module's
     forward therefore normalises too, so the gradients explain the real probability and not
     a close cousin of it.

What was already real in Week 1 and still is:
  - the tensor shapes and the preprocessing,
  - which module Grad-CAM hooks (`target_layer`) and how ViT tokens fold back into a
    patch grid (`reshape_transform`),
  - the fact that gradients reach the activations at all, which on a frozen backbone
    they do not by default. See `classifier_module.__doc__`.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

MODEL_ID = "microsoft/rad-dino"
REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"   # pinned in D15
STUB_MODEL_ID = "stub_radino_randomhead_v0"
STUB_SEED = 20260929

BUNDLE_DIR = Path(__file__).resolve().parents[2] / "artifacts/p2/bundle"

_CACHE = {}


def trained_head(bundle_dir=None):
    """(W, b, model_id, variant) for the trained head, or None if no bundle is on disk.

    TB_P2_FORCE_STUB=1 pins the stub even when a bundle exists, so a test can exercise the
    stub path on a machine that has one.
    """
    if os.environ.get("TB_P2_FORCE_STUB") == "1":
        return None
    d = Path(bundle_dir or BUNDLE_DIR)
    key = ("head", str(d))
    if key in _CACHE:
        return _CACHE[key]
    meta_p, head_p = d / "bundle.json", d / "head.npz"
    if not (meta_p.exists() and head_p.exists()):
        return None
    import json
    meta = json.load(open(meta_p, encoding="utf-8"))
    with np.load(head_p) as z:
        W = z["W"].astype(np.float32)
        b = z["b"].astype(np.float32)
        n = int(z["n_classes"])
    _CACHE[key] = (W, b, meta["model"]["model_id"], meta.get("variant"), n)
    return _CACHE[key]


def is_stub():
    """True when no trained head is available, so callers can check before running a batch."""
    return trained_head() is None


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
        # without pulling 330 MB of checkpoint. Seeded, because otherwise every cache
        # clear produces a different backbone and anything built against it — P4's
        # fixtures, a cross-check between two entry points — silently stops matching.
        torch.manual_seed(STUB_SEED)
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


def _features(model, x):
    """Backbone features in the space the head was trained in: pooled, float32, L2-normalised.

    The normalisation is not cosmetic — see the module docstring. It matches
    `embed.Embedder.embed`, which is what produced the C07 embeddings the head was fitted on.
    """
    out = model(pixel_values=x)
    feat = out.pooler_output if getattr(out, "pooler_output", None) is not None \
        else out.last_hidden_state[:, 0]
    feat = feat.float()
    return feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-12)


def classify(img_for_analysis, pretrained=True, bundle_dir=None):
    """R2. TB probability for one standardised image.

    Real when a trained head is on disk (`is_stub=False`); the Week-1 deterministic stub
    otherwise, in which case the probability is meaningless and `is_stub` says so.
    """
    import torch
    _, model, stub_head = _load(pretrained)
    tr = trained_head(bundle_dir)
    x = preprocess(img_for_analysis, pretrained)
    with torch.inference_mode():
        feat = _features(model, x)
        if tr is None:
            logits = stub_head(feat)[0]
        else:
            W, b, _mid, _var, _n = tr
            logits = feat[0] @ torch.from_numpy(W).T + torch.from_numpy(b)
        p = torch.softmax(logits, dim=-1)
    n_cls = len(p)
    tb_idx = 1 if n_cls == 2 else 2
    return ClsResult(
        model_id=STUB_MODEL_ID if tr is None else tr[2],
        head="binary" if n_cls == 2 else "three_class",
        logits=[float(v) for v in logits], p_tb_raw=float(p[tb_idx]),
        three_class=None if n_cls == 2 else [float(v) for v in p],
        is_stub=tr is None)


def classifier_module(pretrained=True, bundle_dir=None):
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
    _, model, stub_head = _load(pretrained)
    tr = trained_head(bundle_dir)
    if tr is None:
        head, model_id, stub_flag = stub_head, STUB_MODEL_ID, True
    else:
        W, b, model_id, _var, n = tr
        head = torch.nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(W))
            head.bias.copy_(torch.from_numpy(b))
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        stub_flag = False

    class R2Classifier(torch.nn.Module):
        # is_stub and model_id are set below: a class body cannot read the enclosing
        # function's locals (LOAD_NAME skips the closure), so assigning them here is a
        # NameError, not a shadow.
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
            # normalised exactly as in `classify`, so Grad-CAM explains the reported number
            return self.head(_features(self.backbone, pixel_values))   # [B, C]

        @property
        def tb_index(self):
            """Which output column is TB — 1 for a binary head, 2 for three-class. Grad-CAM
            must target this, not column 1 blindly."""
            return 1 if self.head.out_features == 2 else 2

        def preprocess(self, img_for_analysis):
            return preprocess(img_for_analysis, pretrained)

    R2Classifier.is_stub = stub_flag
    R2Classifier.model_id = model_id
    return R2Classifier()
