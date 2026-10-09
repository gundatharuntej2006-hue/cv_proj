"""P3-09: Grad-CAM for the RAD-DINO classifier (D19), on a test-only stand-in until P2's R2 lands."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pytorch_grad_cam")
pytest.importorskip("scipy")
np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "tests" / "p3_decisions"))

from standin import StandInRadDino  # noqa: E402

from p3_xai import api as xai_api  # noqa: E402
from p3_xai import d19, sanity  # noqa: E402
from p3_xai.gradcam import GradCamExplainer, find_target_layer, inspect_classifier, make_reshape, randomised_copy  # noqa: E402

CROP = [40, 20, 470, 490]


@pytest.fixture(scope="module")
def img():
    return (np.random.default_rng(0).random((512, 512)) * 255).astype(np.uint8)


@pytest.fixture(scope="module", params=[(0, 1), (4, 3)], ids=["cls_binary", "registers_3class"])
def model(request):
    reg, k = request.param
    return StandInRadDino(registers=reg, n_outputs=k)


# --- inspection: layer, prefix tokens, grid, TB output -----------------------------------------------

def test_inspection_from_config(model):
    spec, _, layer = inspect_classifier(model)
    reg = model.backbone.config.__dict__.get("num_register_tokens", 0) or 0
    assert spec.n_prefix == 1 + reg                       # CLS + registers, read from the config
    assert spec.patch_size == 14 and spec.grid_hw == (16, 16) and spec.input_hw == (224, 224)
    assert spec.target_layer_name == "backbone.encoder.layer.1.norm1"
    assert layer is model.backbone.encoder.layer[-1].norm1
    assert spec.tb_index == (0 if spec.n_outputs == 1 else 2)


def test_reshape_drops_prefix_and_checks_token_count(model):
    spec, _, _ = inspect_classifier(model)
    t = torch.randn(2, spec.n_prefix + 256, 8)
    out = make_reshape(spec)(t)
    assert out.shape == (2, 8, 16, 16)
    assert torch.equal(out[0, :, 0, 0], t[0, spec.n_prefix, :])          # first patch token -> top-left
    with pytest.raises(RuntimeError, match="patch grid"):
        make_reshape(spec)(torch.randn(1, spec.n_prefix + 255, 8))


def test_block_output_gives_zero_patch_gradient_so_norm1_is_used(model):
    """Why D19 targets norm1: the CLS-pooled head sees the last block output only through a
    per-token LayerNorm, so patch tokens there get exactly zero gradient."""
    spec, wrapped, _ = inspect_classifier(model)
    x = spec.preprocess(np.full((300, 300), 120, np.uint8))
    captured = {}
    h = model.backbone.encoder.layer[-1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("out", (o[0] if isinstance(o, tuple) else o)))
    try:
        out = model(x)
        g = torch.autograd.grad(out[0, spec.tb_index], captured["out"])[0]
    finally:
        h.remove()
    assert g[0, spec.n_prefix:].abs().max().item() == 0.0                 # patch tokens: no gradient
    assert g[0, 0].abs().max().item() > 0                                   # CLS token: gradient


# --- heatmap contract ------------------------------------------------------------------------------

def test_heatmap_shape_dtype_range_and_zero_outside_crop(model, img):
    h = GradCamExplainer(model)(img, CROP)
    assert h.shape == (512, 512) and h.dtype == np.float16
    assert float(h.min()) >= 0.0 and float(h.max()) == 1.0
    outside = np.ones((512, 512), bool)
    outside[CROP[1]:CROP[3], CROP[0]:CROP[2]] = False
    assert not h[outside].any()


def test_deterministic_and_crop_dependent(model, img):
    e = GradCamExplainer(model)
    a, b = e(img, CROP), e(img.copy(), CROP)
    assert np.array_equal(a, b)
    c = e(img, [0, 0, 512, 512])
    assert not np.array_equal(a, c)


def test_rgb_and_dtype_handling(model, img):
    e = GradCamExplainer(model)
    assert np.array_equal(e(img, CROP), e(np.stack([img] * 3, -1), CROP))
    with pytest.raises(ValueError, match="uint8"):
        e(img.astype(np.float32), CROP)


# --- randomisation sanity ----------------------------------------------------------------------------

def test_randomised_copy_changes_only_intended_parts(model):
    head = randomised_copy(model, "head", seed=0)
    assert not torch.equal(head.head.weight, model.head.weight)
    for (n, p), (_, q) in zip(model.backbone.named_parameters(), head.backbone.named_parameters()):
        assert torch.equal(p, q), n                                         # backbone untouched
    both = randomised_copy(model, "head+last_block", seed=0)
    assert not torch.equal(both.backbone.encoder.layer[-1].mlp.fc1.weight,
                           model.backbone.encoder.layer[-1].mlp.fc1.weight)
    assert torch.equal(both.backbone.encoder.layer[0].mlp.fc1.weight, model.backbone.encoder.layer[0].mlp.fc1.weight)


def test_sanity_check_on_fake_development_images(model, tmp_path):
    from p3_det.data import DataPaths
    from p3_det.data.fake_p1 import make_fake_p1_tree
    root = make_fake_p1_tree(tmp_path / "p1", n_per_split=50)
    items = sanity.sample_images(DataPaths(root), seed=0, decisions_dir=tmp_path / "nodec")
    assert len(items) == 20 and sum(i[1] for i in items) == 10
    assert items == sanity.sample_images(DataPaths(root), seed=0, decisions_dir=tmp_path / "nodec")
    res = sanity.run(model, items, seed=0)
    for scope in ("head", "head+last_block"):
        sm = res["summary"][scope]
        assert sm["n_images"] == 20 and -1 <= sm["mean"] <= 1 and sm["mean_abs"] >= abs(sm["mean"]) - 1e-9
    assert res["passed"] == (res["summary"]["head"]["mean"] < 0.5)        # the pre-registered rule, as written
    assert all("spearman_head+last_block" in r for r in res["per_image"])
    # NOTE: the outcome on a random-weight stand-in says nothing about RAD-DINO; only the machinery is tested.


def test_spearman_uses_inside_crop_only():
    a = np.zeros((512, 512), np.float16)
    b = np.zeros((512, 512), np.float16)
    r = np.random.default_rng(1)
    a[10:100, 10:100] = r.random((90, 90))
    b[10:100, 10:100] = r.random((90, 90))
    assert abs(sanity.spearman_inside(a, b, [10, 10, 100, 100])) < 0.1     # unrelated maps
    assert sanity.spearman_inside(a, a, [10, 10, 100, 100]) == pytest.approx(1.0)


# --- D19 file ----------------------------------------------------------------------------------------

def test_d19_written_from_results_and_lock_refused_on_standin(model, img, tmp_path):
    from c16_standin import validate_decision
    from p3_det import decisions as dec
    path = tmp_path / "p3_gradcam_config.json"
    dec.write_criterion(path, d19.criterion_template("2026-10-09"))
    validate_decision(json.loads(path.read_text()))
    res = {"spec": inspect_classifier(model)[0].as_record(), "seed": 0, "passed": True,
           "summary": {"head": {"mean": 0.1}, "head+last_block": {"mean": 0.05}}}
    rp = tmp_path / "res.json"
    rp.write_text("{}")
    doc = sanity.write_d19(res, model, rp, dry_run=True, path=path, vram_file=tmp_path / "none.json")
    v = doc["value"]
    assert v["target_layer"].endswith(".norm1") and v["classifier_is_stub"] is True
    assert doc["data_used"] == ["development"] and doc["locked"] is False
    validate_decision(doc)
    with pytest.raises(dec.Refused, match="stand-in"):
        sanity.write_d19(res, model, rp, dry_run=True, lock=True, path=path, vram_file=tmp_path / "none.json")


# --- api -------------------------------------------------------------------------------------------

def test_api_explain_uses_cached_provider(img, monkeypatch):
    monkeypatch.delenv("TB_MOCK", raising=False)
    monkeypatch.setenv("TB_XAI_DEVICE", "cpu")
    calls = []

    def provider():
        calls.append(1)
        return StandInRadDino()
    xai_api.set_classifier_provider(provider)
    try:
        h1 = xai_api.explain(img, CROP)
        h2 = xai_api.explain(img, CROP)
        assert len(calls) == 1 and np.array_equal(h1, h2) and h1.dtype == np.float16
    finally:
        xai_api.set_classifier_provider(None)


def test_api_without_p2_fails_clearly_and_mock_still_works(img, monkeypatch):
    monkeypatch.delenv("TB_MOCK", raising=False)
    xai_api.set_classifier_provider(None)
    if __import__("importlib").util.find_spec("p2_cls") is None:
        with pytest.raises(RuntimeError, match="classifier_module"):
            xai_api.explain(img, CROP)
    monkeypatch.setenv("TB_MOCK", "1")
    h = xai_api.explain(img, CROP)
    assert h.shape == (512, 512) and h.dtype == np.float16


# --- VRAM on the laptop GPU --------------------------------------------------------------------------

@pytest.mark.gpu
def test_vram_peak_explain_rtx4050(img):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    try:
        from p2_cls.api import classifier_module
        module, source = classifier_module(), "p2_cls.api.classifier_module()"
    except ImportError:
        module, source = StandInRadDino(full_size=True), "stand-in RAD-DINO architecture (ViT-B/14, 518 px)"
    e = GradCamExplainer(module.cuda())
    e(img, CROP)                                                            # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e(img, CROP)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved() / 1e9
    rec = {"peak_gb": round(peak, 3), "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
           "device": torch.cuda.get_device_name(0), "classifier": source, "spec": e.spec.as_record()}
    out = REPO / "artifacts" / "p3" / "xai" / "vram_explain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"\nexplain() peak reserved {peak:.2f} GB on {rec['device']} ({source})")
    assert peak <= sanity.VRAM_LIMIT_GB
