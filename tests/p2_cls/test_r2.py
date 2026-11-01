# -*- coding: utf-8 -*-
"""R2: the classification entry points Person 2 owes Person 3 (Grad-CAM) and Person 4.

Originally the P2-02 acceptance test for the *stub* ("classifier_module() runs forward +
backward"). The head is trained now, so these also pin the thing that separates a real
result from a stub one: `is_stub`, which P4-09's gate keys off. Both paths are covered — the
stub path is forced with TB_P2_FORCE_STUB=1 so its guarantees stay tested on a machine that
has a bundle, and the real-head tests skip themselves where there is none.

Everything uses pretrained=False unless noted, so the tests exercise the real architecture
with random backbone weights and CI does not pull a 330 MB checkpoint on every push.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from p2_cls import api as cls_api               # noqa: E402

needs_bundle = pytest.mark.skipif(
    cls_api.trained_head() is None, reason="no trained head at artifacts/p2/bundle/")


@pytest.fixture(scope="module")
def module():
    return cls_api.classifier_module(pretrained=False)


@pytest.fixture
def stub_module(monkeypatch):
    """The Week-1 stub, forced, so what it promised P3 and P4 stays true."""
    monkeypatch.setenv("TB_P2_FORCE_STUB", "1")
    cls_api._CACHE.clear()
    yield cls_api.classifier_module(pretrained=False)
    cls_api._CACHE.clear()


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


def test_a_forced_stub_still_says_so(stub_module):
    """P4-09 must refuse numbers from an untrained head, so the flag has to be honest."""
    assert stub_module.is_stub is True
    assert stub_module.model_id == cls_api.STUB_MODEL_ID
    assert cls_api.classify(img(seed=7), pretrained=False).is_stub is True


def test_stub_head_is_deterministic_across_calls(monkeypatch):
    """P4's fixtures are regenerated on four machines; a reseeded head would churn them."""
    monkeypatch.setenv("TB_P2_FORCE_STUB", "1")
    cls_api._CACHE.clear()
    a = cls_api.classifier_module(pretrained=False).head.weight.detach().clone()
    cls_api._CACHE.clear()
    b = cls_api.classifier_module(pretrained=False).head.weight.detach().clone()
    cls_api._CACHE.clear()
    assert torch.allclose(a, b)


def test_the_random_backbone_is_deterministic_too(stub_module):
    """Not just the head. An unseeded `from_config` gives different weights on every cache
    clear, so anything cross-checked against it passes or fails depending on test order."""
    w = stub_module.backbone.encoder.layer[-1].state_dict()
    key = sorted(w)[0]
    first = w[key].detach().clone()
    cls_api._CACHE.clear()
    again = cls_api.classifier_module(pretrained=False) \
        .backbone.encoder.layer[-1].state_dict()[key].detach()
    assert torch.allclose(first, again)


@needs_bundle
def test_a_trained_head_is_not_reported_as_a_stub(module):
    assert module.is_stub is False
    assert module.model_id != cls_api.STUB_MODEL_ID
    assert cls_api.is_stub() is False


@needs_bundle
def test_the_trained_weights_are_the_bundles_weights(module):
    W, b, _mid, _var, _n = cls_api.trained_head()
    assert np.allclose(module.head.weight.detach().numpy(), W)
    assert np.allclose(module.head.bias.detach().numpy(), b)
    assert not any(p.requires_grad for p in module.head.parameters()), \
        "the head is frozen at inference; a trainable copy would drift from the registry"


@needs_bundle
def test_tb_index_follows_the_head_width(module):
    """Grad-CAM must target the TB column, which moves with the head type."""
    assert module.tb_index == (1 if module.head.out_features == 2 else 2)


@needs_bundle
def test_features_are_l2_normalised_as_the_head_expects(module):
    """The one wiring mistake that yields confident wrong answers instead of an error."""
    x = module.preprocess(img(seed=4))
    with torch.no_grad():
        f = cls_api._features(module.backbone, x)
    assert torch.allclose(f.norm(dim=1), torch.ones(1), atol=1e-5)


@needs_bundle
def test_classify_agrees_with_the_module_forward(module):
    """Two entry points, one number: P3 explains what P4 reports."""
    a = img(seed=5)
    r = cls_api.classify(a, pretrained=False)
    with torch.no_grad():
        logits = module(module.preprocess(a))
        p = torch.softmax(logits, dim=-1)[0, module.tb_index]
    assert abs(r.p_tb_raw - float(p)) < 1e-5


def test_backbone_stays_frozen(module):
    assert not any(p.requires_grad for p in module.backbone.parameters())


@pytest.mark.slow
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
    # is_stub tracks whether a trained head was found, not whether the backbone is real
    assert r.is_stub is (cls_api.trained_head() is None)
