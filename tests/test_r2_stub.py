# -*- coding: utf-8 -*-
"""P2-02 acceptance: "the stub classifier_module() runs forward + backward".

Everything here uses pretrained=False, so the tests exercise the real architecture with
random weights and CI does not pull a 330 MB checkpoint on every push. The one test that
needs real weights is marked `slow` and skips itself when the cache is cold.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from p2_cls import api_r2 as cls_api               # noqa: E402


@pytest.fixture(scope="module")
def module():
    return cls_api.classifier_module(pretrained=False)


def img(h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w), dtype=np.uint8)


def test_geometry_matches_what_p2_01_measured(module):
    assert module.hidden_size == 768
    assert module.image_size == 518 and module.patch_size == 14
    assert module.grid == 37


def test_forward_gives_two_logits(module):
    x = module.preprocess(img())
    assert x.shape[1:] == (3, module.image_size, module.image_size)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 2) and torch.isfinite(out).all()


def test_forward_and_backward_reach_the_activations(module):
    """The acceptance criterion, and the trap from P2-01: with a frozen backbone the
    activations carry no gradient unless something makes them differentiable. If this
    test ever fails, Grad-CAM returns a heatmap of zeros without raising."""
    store = {}

    def grab(_m, _i, out):
        store["act"] = out[0] if isinstance(out, tuple) else out

    h = module.target_layer.register_forward_hook(grab)
    try:
        x = module.preprocess(img(seed=1))
        logits = module(x)
        act = store["act"]
        assert act.requires_grad, "the target layer's output must be differentiable"
        act.retain_grad()
        logits[0, 1].backward()                       # the TB logit
        assert act.grad is not None
        assert torch.isfinite(act.grad).all()
        assert float(act.grad.abs().sum()) > 0, "gradients reached the activations but are all zero"
    finally:
        h.remove()


def test_reshape_transform_folds_tokens_into_a_patch_grid(module):
    x = module.preprocess(img(seed=2))
    with torch.no_grad():
        tokens = module.backbone(pixel_values=x).last_hidden_state
    assert tokens.shape[1] == module.grid ** 2 + 1                 # 1370 for 518 / 14
    grid = module.reshape_transform(tokens)
    assert grid.shape == (1, module.hidden_size, module.grid, module.grid)
    # the prefix token must be dropped, not folded in: check a known patch survives
    assert torch.allclose(grid[0, :, 0, 0], tokens[0, 1, :], atol=1e-5)


def test_reshape_transform_handles_register_tokens_if_a_later_revision_adds_them(module):
    """Do not hard-code "drop 1". A DINOv2 variant with register tokens would silently
    shift every heatmap by a patch."""
    fake = torch.zeros(1, module.grid ** 2 + 5, module.hidden_size)
    fake[0, 5:, :] = 1.0
    out = module.reshape_transform(fake)
    assert out.shape == (1, module.hidden_size, module.grid, module.grid)
    assert float(out.min()) == 1.0


def test_stub_is_labelled_as_a_stub(module):
    assert module.is_stub is True
    assert module.model_id == cls_api.STUB_MODEL_ID


def test_head_is_deterministic_across_calls():
    """P4's fixtures are regenerated on four machines; a reseeded head would churn them."""
    cls_api._CACHE.clear()
    a = cls_api.classifier_module(pretrained=False).head.weight.detach().clone()
    cls_api._CACHE.clear()
    b = cls_api.classifier_module(pretrained=False).head.weight.detach().clone()
    assert torch.allclose(a, b)


def test_backbone_stays_frozen(module):
    assert not any(p.requires_grad for p in module.backbone.parameters())
def test_classify_with_real_weights_returns_a_wellformed_result():
    import os
    from pathlib import Path
    cache = Path(os.environ.get("HF_HOME", "")) / "hub" / "models--microsoft--rad-dino"
    if not cache.exists():
        pytest.skip("RAD-DINO weights not in the local cache")
    r = cls_api.classify(img(seed=3))
    assert r.head == "binary" and len(r.logits) == 2
    assert 0.0 <= r.p_tb_raw <= 1.0
    assert r.three_class is None
    assert r.is_stub is True, "an untrained head must never be reported as a real result"
