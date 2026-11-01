# -*- coding: utf-8 -*-
"""Frozen-encoder feature extraction with resumable checkpointing and shape assertions.

Both encoders are used FROZEN. No backbone is fine-tuned here.

Comparability, stated rather than hidden:
  RAD-DINO  ViT-B/14, 768-d, official input 518 px -> 37x37 patch grid.
  RayDINO   ViT-L/16 (XRay-DINO), expected 1024-d, official input resolution must be read
            from the model card once access is granted. 16 does not divide 518, so the
            patch grids CANNOT match by construction.

Consequence: the PRIMARY comparison is matched GLOBAL heads on each encoder's own pooled
output under its own model-card preprocessing. A regional comparison is SECONDARY and
requires `pool_to_grid`, a declared deterministic adaptive average pool to a common GxG
grid. Preprocessing differences are a documented factor, not something to pretend away.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from guards import assert_path_allowed, assert_shape

BLOCK = 32


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------- loaders
def load_rad_dino(revision="110cbc18d5133582e320b43d53bf5c44e410c936", device="cuda"):
    """microsoft/rad-dino, MIT licence. Returns (model, processor, meta)."""
    import torch
    from transformers import AutoModel, AutoImageProcessor
    name = "microsoft/rad-dino"
    proc = AutoImageProcessor.from_pretrained(name, revision=revision)
    model = AutoModel.from_pretrained(name, revision=revision).to(device).eval()
    meta = {"encoder": "RAD-DINO", "hf_id": name, "revision": revision,
            "licence": "MIT", "arch": "ViT-B/14",
            "dim": int(model.config.hidden_size),
            "input_px": int(getattr(proc, "crop_size", {}).get("height", 518)),
            "patch_size": int(model.config.patch_size)}
    return model, proc, meta


def load_raydino(weights_path_or_url, repo_dir=None, device="cuda"):
    """XRay-DINO / RayDINO via torch.hub from facebookresearch/dinov2.

    **The weights are gated.** Request access at
    https://ai.meta.com/resources/models-and-libraries/raydino-downloads/ and pass either
    the local checkpoint path or the temporary URL from the access email. Weights are
    released under the **FAIR Noncommercial Research License** — see the repository's
    LICENSE_XRAY_DINO_MODEL. The DINOv2 code is Apache-2.0.

    Nothing here downloads weights without an explicit path/URL supplied by the user.
    """
    import torch
    if not weights_path_or_url:
        raise RuntimeError(
            "RayDINO weights not supplied. They are gated: request access at "
            "https://ai.meta.com/resources/models-and-libraries/raydino-downloads/ "
            "and pass the local path or the emailed URL. This package will not and "
            "cannot fetch them for you.")
    src = repo_dir or "facebookresearch/dinov2"
    model = torch.hub.load(src, "xray_dino_vitl16", weights=weights_path_or_url)
    model = model.to(device).eval()
    dim = int(getattr(model, "embed_dim", 1024))
    meta = {"encoder": "RayDINO (XRay-DINO)", "source": src,
            "entrypoint": "xray_dino_vitl16",
            "licence": "FAIR Noncommercial Research License (weights); "
                       "Apache-2.0 (DINOv2 code)",
            "arch": "ViT-L/16", "dim": dim,
            "input_px": "READ FROM MODEL CARD - do not guess",
            "patch_size": 16,
            "weights_source": "user-supplied path or gated temporary URL"}
    return model, None, meta


# --------------------------------------------------------------------- pooling
def pool_to_grid(tokens, src_grid, dst_grid):
    """[N, src*src, C] -> [N, dst*dst, C] by adaptive average pooling, then L2-normalise.

    This is the DECLARED deterministic harmonisation. It is applied identically to both
    encoders when a regional comparison is attempted. `src_grid` must be stated by the
    caller from the model's own patch grid; it is never inferred.
    """
    import torch
    n, t, c = tokens.shape
    if t != src_grid * src_grid:
        raise AssertionError("tokens have %d positions, not %d x %d"
                             % (t, src_grid, src_grid))
    g = tokens.float().reshape(n, src_grid, src_grid, c).permute(0, 3, 1, 2)
    g = torch.nn.functional.adaptive_avg_pool2d(g, dst_grid)
    g = g.permute(0, 2, 3, 1).reshape(n, dst_grid * dst_grid, c)
    g = g / g.norm(dim=2, keepdim=True).clamp_min(1e-12)
    return g.half()


# --------------------------------------------------------------------- extraction
def extract_features(image_paths, case_ids, encode_batch, out_npz, expected_dim,
                     expected_tokens=None, batch=BLOCK, log=print):
    """Resumable streaming extraction with a row-ID manifest and shape assertions.

    encode_batch(list_of_paths) -> np.ndarray [B, D] (global) or [B, T, D] (regional).
    Progress is checkpointed every block, so an interrupted Kaggle session resumes without
    recomputing. Rows are keyed by `case_ids`, never by positional order.
    """
    out_npz = assert_path_allowed(out_npz)
    for p in image_paths:
        assert_path_allowed(p)
    if len(image_paths) != len(case_ids):
        raise AssertionError("paths and ids differ in length")
    if len(set(case_ids)) != len(case_ids):
        raise AssertionError("duplicate case ids")

    ckpt = Path(str(out_npz) + ".partial.npz")
    done_ids, chunks = [], []
    if ckpt.exists():
        with np.load(ckpt, allow_pickle=False) as z:
            done_ids = [str(x) for x in z["case_id"]]
            chunks = [z["feat"]]
        log("  resuming: %d of %d already encoded" % (len(done_ids), len(case_ids)))

    todo = [(p, i) for p, i in zip(image_paths, case_ids) if i not in set(done_ids)]
    t0 = time.perf_counter()
    bs = batch
    k = 0
    while k < len(todo):
        sub = todo[k:k + bs]
        try:
            arr = encode_batch([p for p, _ in sub])
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                log("  OOM: halving batch to %d (semantics unchanged)" % bs)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
        arr = np.asarray(arr)
        if arr.shape[0] != len(sub):
            raise AssertionError("encoder returned %d rows for %d images"
                                 % (arr.shape[0], len(sub)))
        if arr.shape[-1] != expected_dim:
            raise AssertionError("embedding dim %d, expected %d"
                                 % (arr.shape[-1], expected_dim))
        if expected_tokens is not None and arr.ndim == 3 and arr.shape[1] != expected_tokens:
            raise AssertionError("token count %d, expected %d"
                                 % (arr.shape[1], expected_tokens))
        chunks.append(arr.astype(np.float16))
        done_ids.extend(i for _, i in sub)
        np.savez(ckpt, case_id=np.asarray(done_ids, dtype="<U64"),
                 feat=np.concatenate(chunks))
        chunks = [np.concatenate(chunks)]
        k += len(sub)
        log("  %d/%d  (%.1f min)" % (len(done_ids), len(case_ids),
                                     (time.perf_counter() - t0) / 60))

    feat = np.concatenate(chunks) if chunks else np.zeros((0, expected_dim), np.float16)
    order = {i: n for n, i in enumerate(done_ids)}
    idx = [order[i] for i in case_ids]            # restore the caller's ID order exactly
    feat = feat[idx]
    assert_shape(feat, (len(case_ids), None) if feat.ndim == 2
                 else (len(case_ids), None, None), "features")
    np.savez_compressed(out_npz, case_id=np.asarray(case_ids, dtype="<U64"), feat=feat)
    if ckpt.exists():
        os.remove(ckpt)
    meta = {"n": len(case_ids), "shape": list(feat.shape), "dtype": str(feat.dtype),
            "sha256": sha_file(out_npz), "minutes": round((time.perf_counter() - t0) / 60, 2),
            "final_batch_size": bs}
    Path(str(out_npz) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("  wrote %s %s" % (out_npz.name, feat.shape))
    return feat, meta
