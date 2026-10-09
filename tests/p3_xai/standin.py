"""TEST-ONLY stand-in for P2's classifier_module() (P2's R2 stub is not in the repo yet).

Same structure as RAD-DINO + a head: a Hugging Face DINOv2 backbone (optionally with register
tokens), random weights, and a linear head on the CLS output (pooler_output). Tiny by default so
CPU tests are fast; ``full_size=True`` gives the RAD-DINO architecture (ViT-B/14 at 518 px) for the
VRAM test. Never used outside tests: the real explain() calls p2_cls.api.classifier_module().
"""
from __future__ import annotations

import torch
from torch import nn


class StandInRadDino(nn.Module):
    is_stub = True

    def __init__(self, *, registers: int = 0, n_outputs: int = 1, full_size: bool = False, seed: int = 0):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model, Dinov2WithRegistersConfig, Dinov2WithRegistersModel
        torch.manual_seed(seed)
        size = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                    image_size=518) if full_size else dict(hidden_size=64, num_hidden_layers=2,
                                                           num_attention_heads=4, intermediate_size=128,
                                                           image_size=224)
        if registers:
            self.backbone = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(patch_size=14, num_register_tokens=registers, **size))
        else:
            self.backbone = Dinov2Model(Dinov2Config(patch_size=14, **size))
        self.head = nn.Linear(size["hidden_size"], n_outputs)
        self.input_size = (size["image_size"], size["image_size"])
        self.image_mean = (0.5307, 0.5307, 0.5307)   # placeholder normalisation for the stand-in only
        self.image_std = (0.2583, 0.2583, 0.2583)
        self.model_id = f"standin_dinov2{'_reg' + str(registers) if registers else ''}_{n_outputs}out"

    def forward(self, x):
        return self.head(self.backbone(pixel_values=x).pooler_output)   # CLS token after final LayerNorm
