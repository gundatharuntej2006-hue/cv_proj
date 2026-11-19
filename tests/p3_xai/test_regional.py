from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from p3_xai.regional import RegionalClassifier


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_size=12, patch_size=1)
        self.tokens = nn.Parameter(torch.randn(1, 145, 3))

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.tokens.expand(len(pixel_values), -1, -1))


def test_regional_score_matches_p2_mean_of_seed_maxima_and_has_gradients():
    torch.manual_seed(3)
    backbone = Backbone()
    w = np.array([[[1, 0, 0]], [[-1, 0, 0]]], np.float32)
    b = np.array([[0.2], [-0.1]], np.float32)
    model = RegionalClassifier(backbone, None, w, b)
    score = model(torch.zeros(1, 3, 12, 12))
    x = backbone.tokens.detach().numpy()[0, -144:].copy()
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    x = x.astype(np.float16).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    x = x.astype(np.float16).astype(np.float32)
    logits = np.stack([x @ w[s, 0] + b[s, 0] for s in range(2)])
    assert np.isclose(score.item(), logits.max(axis=1).mean(), atol=1e-6)
    assert not np.isclose(score.item(), logits.mean(axis=0).max())
    score.sum().backward()
    assert backbone.tokens.grad.abs().sum() > 0
