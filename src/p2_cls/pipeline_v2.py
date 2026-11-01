# -*- coding: utf-8 -*-
"""v2 inference: the regional 12x12 max head, exact region attribution, optional clinical fusion.

    from p2_cls.pipeline_v2 import load_bundle_v2, run_case_v2
    b = load_bundle_v2()                                   # artifacts/p2/bundle_v2_regional/
    r = run_case_v2("xray.png", b)                         # image only
    r = run_case_v2("xray.png", b, clinical={"past_tb": 1, "c": 1, ...})

Why a second pipeline rather than a change to pipeline.py: the served v1 bundle is the
2026-10-14 global head. Every result since 2026-10-20 is about the regional head, which
needs a different embedder (lungs-masked 12x12 regions) and a different head shape. v1 is
left untouched so nothing already validated moves; v2 is additive.

Stages:  standardise -> quality gate -> lung mask -> regional embed -> head -> triage
         (-> clinical fusion, only if every field is supplied)

What this returns that v1 cannot:
  * region_logits  - the 12x12 map of per-region logits (seed mean). The image logit is the
                     max over regions, so the arg-max region IS the decision region, exactly.
                     This is attribution by construction, not a post-hoc saliency map.
  * fusion         - a fused probability when the clinical form is supplied; EXPLORATORY,
                     fitted on an exposed cohort, never a replacement for the image score.

What it deliberately does not do: no OOD gate (the v1 bank is for the global embedder), no
temperature scaling (none was fitted for this head), no TB / NOT_TB label - only
`screen_positive` at a stated, development-fitted operating point.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_V2 = ROOT / "artifacts/p2/bundle_v2_regional"
_SCRIPTS = ROOT / "scripts"


def _scripts_on_path():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))


def load_bundle_v2(path=BUNDLE_V2):
    p = Path(path)
    meta = json.loads((p / "bundle.json").read_text(encoding="utf-8"))
    with np.load(p / "head.npz") as z:
        W, b = z["W"].astype(np.float32), z["b"].astype(np.float32)
    fusion = None
    if (p / "fusion.json").is_file():
        fusion = json.loads((p / "fusion.json").read_text(encoding="utf-8"))
    return {"meta": meta, "W": W, "b": b, "fusion": fusion,
            "_embedder": None, "_segmenter": None}


def _segmenter(bundle):
    """torchxrayvision PSPNet, float32, on the GPU when one is present (CPU otherwise). The
    training-time cache was built on CPU; GPU float32 reproduces its masks (p2_41 benchmark)."""
    if bundle["_segmenter"] is None:
        import torch
        import torchxrayvision as xrv
        torch.backends.cudnn.benchmark = False
        m = xrv.baseline_models.chestx_det.PSPNet()
        m.eval()
        dev = "cuda" if torch.cuda.is_available() and not bundle.get("_segmenter_cpu") else "cpu"
        m.to(dev)
        for q in m.parameters():
            q.requires_grad_(False)
        bundle["_segmenter"] = m
        bundle["_segmenter_device"] = dev
    return bundle["_segmenter"]


def _lung_mask(model, img_u8, dev):
    """p1_build_c04_c05.lung_mask with the tensor placed on the model's device."""
    import torch
    import torchxrayvision as xrv
    x = xrv.datasets.normalize(img_u8.astype(np.float32), 255)
    t = torch.from_numpy(x)[None, None, ...].to(dev)
    with torch.inference_mode():
        out = model(t)[0]
        tgt = {n: i for i, n in enumerate(model.targets)}
        li, ri = tgt.get("Left Lung"), tgt.get("Right Lung")
        prob = torch.sigmoid(out)
        m = ((prob[li] > 0.5) | (prob[ri] > 0.5)).cpu().numpy().astype(np.uint8)
    return m


class HFRegionalEmbedder:
    """Any frozen HF ViT encoder -> 12x12 map of L2-normalised region vectors (prefix tokens
    dropped). Same geometry as p2_31.RegionalEmbedder; used for bundles whose backbone is
    not RAD-DINO (candidate v3: RAD-DINO-MAIRA-2)."""

    def __init__(self, model_id, revision, batch=16):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        torch.backends.cudnn.benchmark = False
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        m = AutoModel.from_pretrained(model_id, revision=revision)
        self.half = self.dev == "cuda"
        self.model = (m.half() if self.half else m.float()).eval().to(self.dev)
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.g = int(m.config.image_size // m.config.patch_size)
        self.batch = batch

    def embed(self, images):
        torch = self.torch
        import torch.nn.functional as F
        rgb = [np.stack([im] * 3, axis=-1) for im in images]
        x = self.proc(images=rgb, return_tensors="pt")["pixel_values"]
        x = x.to(self.dev, dtype=torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            tok = self.model(pixel_values=x).last_hidden_state
            pat = tok[:, tok.shape[1] - self.g * self.g:, :].float()
            b, _, c = pat.shape
            grid = pat.reshape(b, self.g, self.g, c).permute(0, 3, 1, 2)
            reg = F.adaptive_avg_pool2d(grid, 12).permute(0, 2, 3, 1).reshape(b, 144, c)
            reg = reg / reg.norm(dim=2, keepdim=True).clamp_min(1e-12)
        return reg.cpu().numpy().astype(np.float16)


def _embedder(bundle):
    """The encoder named in bundle.json: RAD-DINO uses the training-time embedder verbatim;
    any other pinned HF backbone goes through HFRegionalEmbedder."""
    if bundle["_embedder"] is None:
        backbone = bundle["meta"].get("backbone", "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936")
        model_id, revision = backbone.split("@")
        if model_id == "microsoft/rad-dino" and revision.startswith("110cbc18"):
            _scripts_on_path()
            import p2_31_regional_mil as R
            bundle["_embedder"] = R.RegionalEmbedder()
        else:
            bundle["_embedder"] = HFRegionalEmbedder(model_id, revision)
    return bundle["_embedder"]


def lungs_only(img_u8, bundle):
    """Exactly the masking the regional cache was built with (scripts/p2_31_regional_mil.py)."""
    seg = _segmenter(bundle)
    m = _lung_mask(seg, img_u8, bundle["_segmenter_device"]).astype(bool)
    if m.shape != img_u8.shape:
        import cv2
        m = cv2.resize(m.astype(np.uint8), (img_u8.shape[1], img_u8.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    if not m.any():
        return img_u8, {"lung_fraction": 0.0, "masked": False}
    o = img_u8.astype(np.float32)
    o[~m] = float(o.mean())
    return np.clip(o, 0, 255).astype(np.uint8), {"lung_fraction": round(float(m.mean()), 4),
                                                 "masked": True}


def regional_features(img_masked, bundle):
    """[144, 768] float32, L2 per region - the same tensor the head was trained on."""
    _scripts_on_path()
    import p2_32_mil_sweep as S
    X = _embedder(bundle).embed([img_masked])          # [1,144,768] float16
    return S.repool(X, 12, 12)[0].astype(np.float32)


def head(X, bundle):
    """Per-seed region logits -> image score (mean over seeds of max over regions)."""
    W, b = bundle["W"], bundle["b"]
    logits = np.stack([X @ W[s][0] + b[s][0] for s in range(W.shape[0])])   # [S,144]
    per_seed_max = logits.max(axis=1)                                         # [S]
    return float(per_seed_max.mean()), logits.mean(axis=0), per_seed_max


def fuse(score, clinical, bundle):
    """Clinical-context fusion. Refuses, rather than imputes, if any field is missing."""
    f = bundle.get("fusion")
    if f is None:
        return None, "no fusion block in bundle"
    order = f["features_order"][1:]
    missing = [k for k in order if k not in clinical]
    if missing:
        return None, "clinical fields missing: %s" % ", ".join(missing)
    try:
        x = np.array([float(clinical[k]) for k in order])
    except (TypeError, ValueError) as e:
        return None, "clinical field not numeric: %s" % e
    if not np.isin(x, [0.0, 1.0]).all():
        return None, "clinical fields must be 0/1 after encoding (see fusion.json)"
    z = (score - f["score_mean"]) / f["score_sd"]
    lg = f["intercept"] + f["image_score_std"] * z + float(np.dot(
        [f["clinical"][k] for k in order], x))
    return {"p_fused": round(float(1 / (1 + np.exp(-lg))), 6), "logit": round(float(lg), 4),
            "status": f["status"], "source_run": f["source_run"]}, None


def encode_clinical(raw):
    """Map a screening-form dict (Dryad field names and codes) onto the fusion features."""
    try:
        out = {"sex": int(raw["sex"]), "age_group": int(int(raw["age_group"]) == 2),
               "c": int(raw["c"]), "f": int(raw["f"]), "wl": int(raw["wl"]), "ns": int(raw["ns"]),
               "contact": int(raw["contact"]), "past_tb": int(raw["past_tb"]),
               "dm": int(int(raw["dm"]) == 1), "dm_unknown": int(int(raw["dm"]) == 99),
               "scan1": int(int(raw["rad_system"]) == 1), "scan2": int(int(raw["rad_system"]) == 2)}
    except (KeyError, TypeError, ValueError) as e:
        return None, "cannot encode clinical form: %s" % e
    return out, None


def run_stages_v2(img, bundle, clinical=None, image_label=None, t0=None):
    from p2_gate import quality as Q
    t0 = time.perf_counter() if t0 is None else t0
    meta = bundle["meta"]
    out = {"image": image_label, "model_id": meta["model"]["model_id"],
           "bundle_version": meta.get("bundle_version", 2),
           "provisional": bool(meta.get("provisional", True))}
    qr = Q.assess(img)
    out["quality"] = {"pass": bool(qr.passed), "reasons": list(qr.reasons)}
    if not qr.passed:
        out.update(status="REJECTED", stage="quality", reasons=list(qr.reasons), analysis=None,
                   latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return out
    masked, minfo = lungs_only(img, bundle)
    X = regional_features(masked, bundle)
    score, region_mean, per_seed = head(X, bundle)
    thr = meta["thresholds"]
    arg = int(np.argmax(region_mean))
    analysis = {
        "image_logit": round(score, 6),
        "operating_points": {
            "screen_positive_at_spec_0.70": bool(score >= thr["logit_at_spec_0.70"]),
            "positive_at_spec_0.90": bool(score >= thr["logit_at_spec_0.90"]),
            "thresholds_fitted_on": thr["fitted_on"],
            "logit_at_spec_0.70": thr["logit_at_spec_0.70"],
            "logit_at_spec_0.90": thr["logit_at_spec_0.90"]},
        "region_logits": [[round(float(v), 4) for v in row] for row in region_mean.reshape(12, 12)],
        "argmax_region": {"row": arg // 12, "col": arg % 12, "logit": round(float(region_mean[arg]), 4),
                          "note": "the image logit is the max over regions; this is the decision region, exactly"},
        "per_seed_image_logit": [round(float(v), 4) for v in per_seed],
        "lung_mask": minfo,
        "fusion": None,
    }
    if clinical is not None:
        enc, err = encode_clinical(clinical)
        if err:
            analysis["fusion"] = {"status": "not applied", "reason": err}
        else:
            fz, err = fuse(score, enc, bundle)
            analysis["fusion"] = fz if fz else {"status": "not applied", "reason": err}
    out.update(status="ANALYSED", stage="complete", reasons=[], analysis=analysis,
               latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return out


def run_case_v2(image_path, bundle, clinical=None):
    from p2_cls.pipeline import standardise
    t0 = time.perf_counter()
    img, err = standardise(image_path)
    if err:
        return {"image": str(image_path), "model_id": bundle["meta"]["model"]["model_id"],
                "status": "REJECTED", "stage": "decode", "reasons": [err], "analysis": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return run_stages_v2(img, bundle, clinical=clinical, image_label=str(image_path), t0=t0)
