# -*- coding: utf-8 -*-
"""P2-13: the model registry and the R2 reproduction guarantee (D14).

A registry entry is everything needed to reproduce a model's TB probability from an image
or an embedding: the trained head weights, the variant it was trained on (which fixes the
preprocessing), the RAD-DINO revision, and a SHA over the weights. The registry is the five
models the methodology compares — primary, alternate head, full-image variant,
CLAHE-alternate, and the EfficientNet-B3 baseline — each with its D14 SHA.

The R2 contract (`p2_cls.api.classify`) must reproduce the development C08 scores to 1e-4.
That is what `reproduces_scores` checks: run the registered head over the same embeddings
and confirm the probabilities match the exported parquet. A registry entry that fails this
is not allowed into C08.

The head is a plain linear layer on frozen embeddings, so "load the head + matmul the
embedding" is an exact reproduction by construction — the 1e-4 tolerance only absorbs
float32/float16 storage, not modelling drift.
"""
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RegistryEntry:
    model_id: str
    head_type: str            # binary | three_class
    variant: str              # e.g. crop_m0.10_clahe0 — fixes the preprocessing
    backbone: str             # microsoft/rad-dino@<rev> or effnet_b3
    weight_path: str          # .npz of the head weights
    weight_sha256: str
    n_classes: int
    is_primary: bool = False

    def to_json(self):
        return asdict(self)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def save_head(result, path):
    """Persist a trained head (from heads.train_head) as a plain npz: W, b. Returns SHA."""
    import torch
    sd = result.state_dict
    W = sd["weight"].numpy().astype(np.float32)
    b = sd["bias"].numpy().astype(np.float32)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = p.with_suffix(".tmp")
    with open(buf, "wb") as fh:
        np.savez(fh, W=W, b=b, n_classes=np.array(result.n_classes))
    import os
    os.replace(buf, p)
    return sha256_bytes(p.read_bytes())


def load_head(path):
    """(W, b, n_classes) from a saved head npz."""
    with np.load(path) as z:
        return z["W"].astype(np.float32), z["b"].astype(np.float32), int(z["n_classes"])


def head_logits(emb, W, b):
    """Linear head forward: [N, dim] -> [N, n_classes]."""
    return np.asarray(emb, dtype=np.float32) @ W.T + b


def p_tb_from_logits(logits, n_classes):
    """TB probability from head logits, matching the training convention: softmax, TB is
    class 1 (binary) or class 2 (three-class)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)
    return sm[:, 1 if n_classes == 2 else 2]


def classify_embeddings(entry: RegistryEntry, emb):
    """R2 core: TB probability for a batch of embeddings under a registry entry."""
    W, b, n = load_head(entry.weight_path)
    return p_tb_from_logits(head_logits(emb, W, b), n)


def reproduces_scores(entry: RegistryEntry, emb, image_ids, c08_df, tol=1e-4):
    """The D14 / R2 acceptance check: does re-running the head reproduce the C08 p_tb_raw
    for this model to within `tol`? Returns (ok, max_abs_diff, n_checked)."""
    sub = c08_df[c08_df.model_id == entry.model_id].set_index("image_id")
    idx = {i: k for k, i in enumerate(image_ids)}
    common = [i for i in sub.index if i in idx]
    if not common:
        return False, float("inf"), 0
    p_new = classify_embeddings(entry, emb[[idx[i] for i in common]])
    p_old = sub.loc[common, "p_tb_raw"].to_numpy(dtype=np.float32)
    d = float(np.max(np.abs(p_new - p_old)))
    return d <= tol, d, len(common)


class Registry:
    """The set of registered models, persisted as one JSON. Enforces at most one primary."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def add(self, entry: RegistryEntry):
        if entry.is_primary and any(e.is_primary for e in self.entries):
            raise ValueError("a primary model is already registered (D11 picks exactly one)")
        if any(e.model_id == entry.model_id for e in self.entries):
            raise ValueError("duplicate model_id %r" % entry.model_id)
        self.entries.append(entry)

    def primary(self):
        for e in self.entries:
            if e.is_primary:
                return e
        return None

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"models": [e.to_json() for e in self.entries]},
                  open(p, "w", encoding="utf-8"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls([RegistryEntry(**m) for m in d["models"]])

    def verify_shas(self):
        """Every entry's weight file must still hash to its recorded SHA (D14 integrity)."""
        bad = []
        for e in self.entries:
            if not Path(e.weight_path).exists():
                bad.append((e.model_id, "missing weight file"))
            elif sha256_bytes(Path(e.weight_path).read_bytes()) != e.weight_sha256:
                bad.append((e.model_id, "SHA mismatch"))
        return bad
