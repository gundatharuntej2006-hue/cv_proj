"""P3-09: Grad-CAM for the RAD-DINO (ViT) classifier, per D19 (decisions/p3_gradcam_config.json).

Everything model-specific is found by INSPECTING the module from P2's R2 classifier_module():

* backbone config  - the submodule with a ``config`` that has ``patch_size`` (Hugging Face DINOv2 /
                     RAD-DINO), or a timm ViT (``num_prefix_tokens``, ``patch_embed.patch_size``).
* prefix tokens    - 1 CLS + ``config.num_register_tokens`` (0 when absent), or timm ``num_prefix_tokens``.
* target layer     - ``norm1`` (attention-input LayerNorm) of the LAST block of the longest block list
                     (HF ``backbone.encoder.layer``, timm ``blocks``). Not the block output: with a
                     CLS-pooled head, patch tokens there get zero gradient (see D19 and the tests).
* input spec       - ``module.preprocess(crop_uint8) -> tensor[1,3,H,W]`` if P2 provides it; otherwise
                     ``module.input_size`` (H, W), ``module.image_mean``, ``module.image_std``.
* TB output        - ``module.tb_index`` if set; else 0 for a single logit, ``class_names.index("tb")``
                     or C08's order (healthy, sick_non_tb, tb) -> 2 for three outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn

THREE_CLASS_ORDER = ("healthy", "sick_non_tb", "tb")   # C08 / ClsResult.three_class


@dataclass(frozen=True)
class ClassifierSpec:
    input_hw: tuple[int, int]
    patch_size: int
    n_prefix: int                    # CLS + register tokens dropped by the reshape
    grid_hw: tuple[int, int]
    tb_index: int
    n_outputs: int
    target_layer_name: str
    preprocess: Callable[[np.ndarray], torch.Tensor]

    def as_record(self) -> dict:
        return {"input_hw": list(self.input_hw), "patch_size": self.patch_size, "n_prefix_tokens": self.n_prefix,
                "grid_hw": list(self.grid_hw), "tb_index": self.tb_index, "n_outputs": self.n_outputs,
                "target_layer": self.target_layer_name}


class LogitsOnly(nn.Module):
    """Wraps P2's module so forward returns a plain [B, K] logits tensor (pytorch-grad-cam needs that)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        if isinstance(out, dict):
            out = out["logits"]
        elif hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(out.shape[0], -1)


def _backbone_config(module: nn.Module):
    for name, m in module.named_modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "patch_size"):
            p = cfg.patch_size
            return name, int(p if isinstance(p, int) else p[0]), 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        if hasattr(m, "num_prefix_tokens") and hasattr(m, "patch_embed"):
            p = m.patch_embed.patch_size
            return name, int(p if isinstance(p, int) else p[0]), int(m.num_prefix_tokens)
    raise RuntimeError("cannot find a ViT backbone config (patch_size / num_prefix_tokens) in classifier_module()")


def find_target_layer(module: nn.Module) -> tuple[str, nn.Module]:
    """norm1 of the last block of the longest list of transformer blocks."""
    best = None
    for name, m in module.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) and all(hasattr(b, "norm1") for b in m):
            if best is None or len(m) > len(best[1]):
                best = (name, m)
    if best is None:
        raise RuntimeError("cannot find the transformer block list (blocks with norm1) in classifier_module()")
    name, blocks = best
    return f"{name}.{len(blocks) - 1}.norm1", blocks[-1].norm1


def last_block(module: nn.Module) -> nn.Module:
    name, _ = find_target_layer(module)
    return module.get_submodule(name.rsplit(".", 1)[0])


def _default_preprocess(module: nn.Module) -> tuple[tuple[int, int], Callable]:
    missing = [a for a in ("input_size", "image_mean", "image_std") if not hasattr(module, a)]
    if missing:
        raise RuntimeError(f"classifier_module() exposes neither preprocess() nor {missing}; P3 needs P2's exact "
                           "classifier input preparation (see docs/p3_contract_notes.md)")
    size = module.input_size
    h, w = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    mean = torch.tensor(module.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(module.image_std, dtype=torch.float32).view(3, 1, 1)

    def prep(crop: np.ndarray) -> torch.Tensor:
        x = np.asarray(Image.fromarray(crop).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        t = torch.from_numpy(x).unsqueeze(0).expand(3, -1, -1)
        return ((t - mean) / std).unsqueeze(0)
    return (h, w), prep


def inspect_classifier(module: nn.Module) -> tuple[ClassifierSpec, nn.Module, nn.Module]:
    """Return (spec, wrapped model, target layer). Runs one dry forward to read the output size."""
    module.eval()
    _, patch, n_prefix = _backbone_config(module)
    layer_name, layer = find_target_layer(module)
    if hasattr(module, "preprocess"):
        prep = module.preprocess
        hw = tuple(prep(np.zeros((64, 64), np.uint8)).shape[-2:])
    else:
        hw, prep = _default_preprocess(module)
    if hw[0] % patch or hw[1] % patch:
        raise RuntimeError(f"input {hw} is not a multiple of the patch size {patch}")
    wrapped = LogitsOnly(module).eval()
    dev = next(module.parameters()).device
    with torch.no_grad():
        k = int(wrapped(prep(np.zeros((hw[0], hw[1]), np.uint8)).to(dev)).shape[1])
    if hasattr(module, "tb_index"):
        tb = int(module.tb_index)
    elif k == 1:
        tb = 0
    elif k == 3:
        names = list(getattr(module, "class_names", THREE_CLASS_ORDER))
        tb = names.index("tb")
    else:
        raise RuntimeError(f"cannot tell which of {k} outputs is TB; P2 should set classifier_module().tb_index")
    spec = ClassifierSpec(hw, patch, n_prefix, (hw[0] // patch, hw[1] // patch), tb, k, layer_name, prep)
    return spec, wrapped, layer


def make_reshape(spec: ClassifierSpec) -> Callable[[torch.Tensor], torch.Tensor]:
    gh, gw = spec.grid_hw

    def reshape(t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        b, n, c = t.shape
        if n - spec.n_prefix != gh * gw:
            raise RuntimeError(f"{n} tokens - {spec.n_prefix} prefix != {gh}x{gw} patch grid")
        return t[:, spec.n_prefix:, :].reshape(b, gh, gw, c).permute(0, 3, 1, 2)
    return reshape


def crop_gray(img512: np.ndarray, crop_box) -> np.ndarray:
    img = np.asarray(img512)
    if img.dtype != np.uint8:
        raise ValueError(f"img_for_analysis must be uint8, got {img.dtype}")
    if img.ndim == 3:
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img[..., :3])).convert("L"))
    x0, y0, x1, y1 = crop_box
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def to_canvas(cam_input: np.ndarray, crop_box) -> np.ndarray:
    """CAM at classifier-input size -> crop size -> [0,1] inside the crop -> zero 512x512 canvas, float16."""
    x0, y0, x1, y1 = crop_box
    g = np.asarray(Image.fromarray(cam_input.astype(np.float32), mode="F").resize(
        (x1 - x0, y1 - y0), Image.Resampling.BILINEAR), dtype=np.float32)
    g = np.nan_to_num(g, nan=0.0)
    g -= g.min()
    m = float(g.max())
    g = g / m if m > 0 else np.zeros_like(g)
    out = np.zeros((512, 512), np.float32)
    out[y0:y1, x0:x1] = g
    return out.astype(np.float16)


class GradCamExplainer:
    """Grad-CAM per D19 on one classifier module; reuse it across calls (hooks are set per call)."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.spec, self.model, self.layer = inspect_classifier(module)
        self.device = next(module.parameters()).device

    def __call__(self, img512: np.ndarray, crop_box) -> np.ndarray:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        x = self.spec.preprocess(crop_gray(img512, crop_box)).to(self.device)
        with GradCAM(model=self.model, target_layers=[self.layer], reshape_transform=make_reshape(self.spec)) as cam:
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(self.spec.tb_index)])[0]
        return to_canvas(g, crop_box)


def randomised_copy(module: nn.Module, scope: str = "head", seed: int = 0) -> nn.Module:
    """Deep copy with the classifier head (everything outside the backbone) re-initialised; with
    scope="head+last_block" also the last transformer block. Used by the D19 sanity check."""
    import copy
    m = copy.deepcopy(module)
    backbone_name, _, _ = _backbone_config(m)
    torch.manual_seed(seed)
    n = 0
    for name, sub in m.named_modules():
        inside_backbone = backbone_name and (name == backbone_name or name.startswith(backbone_name + "."))
        if not inside_backbone and sub is not m and hasattr(sub, "reset_parameters"):
            sub.reset_parameters()
            n += 1
    if scope == "head+last_block":
        for sub in last_block(m).modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
                n += 1
    elif scope != "head":
        raise ValueError(scope)
    if n == 0:
        raise RuntimeError("found no head parameters outside the backbone to re-initialise")
    return m
