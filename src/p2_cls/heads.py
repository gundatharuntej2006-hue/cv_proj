# -*- coding: utf-8 -*-
"""P2-07: classifier heads on frozen RAD-DINO embeddings. Produces C08, decides D10.

The backbone is frozen (decided a priori, methodology), so training is a small head on
precomputed C07 embeddings. That makes the whole sweep cheap enough to run on the laptop
and to average over several seeds.

Two heads, exactly as the contract requires:
  - binary: Linear(768 -> 2), the TB logit is logits[:, 1]; `logit_tb` and `p_tb_raw`.
  - three_class: Linear(768 -> 3) over (healthy, sick_non_tb, tb); `logits_3c`, and
    `p_tb_raw` is the softmax TB probability so the two heads are comparable downstream.

Class imbalance (TB is ~9%) is handled by one of **exactly two** approved methods, and the
config refuses anything else — D10 pre-registers that the choice is between these two and
nothing is slipped in later:
  - "weighted_sampler": oversample the minority so each batch is roughly balanced.
  - "logit_adjusted":  add log(class prior) to the logits during training (Menon et al.,
                       2020), which shifts the decision boundary without resampling.

Training length is set by early stopping on the development split, never on calibration or
test. Several seeds are trained; the seed-to-seed spread of development AUC is reported
(D10), and the exported scores are from the seed-averaged probabilities.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

APPROVED_IMBALANCE = ("weighted_sampler", "logit_adjusted")
HEAD_TYPES = ("binary", "three_class")
THREE_CLASS_ORDER = ("healthy", "sick_non_tb", "tb")


@dataclass
class TrainConfig:
    head_type: str = "binary"
    imbalance: str = "logit_adjusted"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20                 # dev-AUC early stopping
    batch_size: int = 256
    seed: int = 17

    def __post_init__(self):
        if self.head_type not in HEAD_TYPES:
            raise ValueError("head_type %r not in %s" % (self.head_type, HEAD_TYPES))
        if self.imbalance not in APPROVED_IMBALANCE:
            raise ValueError(
                "imbalance %r is not an approved method. D10 permits exactly %s; a new "
                "method needs a decision change, not a config edit."
                % (self.imbalance, APPROVED_IMBALANCE))


@dataclass
class TrainResult:
    config: TrainConfig
    best_epoch: int
    dev_auc: float
    state_dict: dict
    n_classes: int
    class_prior: List[float]
    history: List[float] = field(default_factory=list)


# ----------------------------------------------------------------- label helpers
def binary_labels(categories):
    """project category -> {0,1} TB label."""
    return np.array([0 if c in ("healthy", "sick_non_tb") else 1 for c in categories],
                    dtype=np.int64)


def three_class_labels(categories):
    """project category -> {0,1,2} over (healthy, sick_non_tb, tb). Any TB subtype -> tb."""
    m = {"healthy": 0, "sick_non_tb": 1}
    return np.array([m.get(c, 2) for c in categories], dtype=np.int64)


def auc(y, score):
    """Binary AUC (Mann-Whitney U, ties averaged)."""
    import pandas as pd
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------- training
def _make_head(dim, n_classes, torch):
    return torch.nn.Linear(dim, n_classes)


def train_head(emb_train, y_train, emb_dev, y_dev, config: TrainConfig) -> TrainResult:
    """Train one head on frozen embeddings. Deterministic given config.seed."""
    import torch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    n_classes = 2 if config.head_type == "binary" else 3
    Xtr = torch.tensor(np.asarray(emb_train, dtype=np.float32))
    ytr = torch.tensor(np.asarray(y_train, dtype=np.int64))
    Xdv = torch.tensor(np.asarray(emb_dev, dtype=np.float32))

    prior = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    prior = prior / prior.sum()
    head = _make_head(Xtr.shape[1], n_classes, torch)
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # imbalance handling
    sampler = None
    logit_adjust = None
    if config.imbalance == "weighted_sampler":
        w = (1.0 / np.clip(prior, 1e-9, None))[y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(w, dtype=torch.double), num_samples=len(y_train), replacement=True)
    else:  # logit_adjusted
        logit_adjust = torch.tensor(np.log(np.clip(prior, 1e-9, None)), dtype=torch.float32)

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config.batch_size, sampler=sampler, shuffle=sampler is None)
    lossf = torch.nn.CrossEntropyLoss()

    best = {"auc": -1.0, "epoch": -1, "state": None}
    hist = []
    for epoch in range(config.max_epochs):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = head(xb)
            if logit_adjust is not None:
                logits = logits + logit_adjust
            lossf(logits, yb).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            dv = head(Xdv)
            if n_classes == 2:
                p_tb = torch.softmax(dv, dim=1)[:, 1].numpy()
            else:
                p_tb = torch.softmax(dv, dim=1)[:, 2].numpy()
        a = auc((np.asarray(y_dev) >= (1 if n_classes == 2 else 2)).astype(int)
                if n_classes == 3 else np.asarray(y_dev), p_tb)
        hist.append(a)
        if a > best["auc"] + 1e-5:
            best = {"auc": a, "epoch": epoch,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}}
        elif epoch - best["epoch"] >= config.patience:
            break

    head.load_state_dict(best["state"])
    return TrainResult(config=config, best_epoch=best["epoch"], dev_auc=best["auc"],
                       state_dict=best["state"], n_classes=n_classes,
                       class_prior=[float(p) for p in prior], history=hist)


def predict_p_tb(result: TrainResult, emb):
    """Seed-specific TB probability for a set of embeddings."""
    import torch
    head = _make_head(np.asarray(emb).shape[1], result.n_classes, torch)
    head.load_state_dict(result.state_dict)
    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(np.asarray(emb, dtype=np.float32)))
        sm = torch.softmax(logits, dim=1).numpy()
    idx = 1 if result.n_classes == 2 else 2
    return sm[:, idx], logits.numpy(), sm


def train_seeds(emb_train, y_train, emb_dev, y_dev, config: TrainConfig,
                seeds=(17, 29, 43, 61, 79)):
    """Train one head per seed. Returns (results, dev_auc_mean, dev_auc_std) — the spread
    D10 asks for. Five seeds by default; the heads are tiny so this costs seconds."""
    results = []
    for s in seeds:
        c = TrainConfig(**{**config.__dict__, "seed": s})
        results.append(train_head(emb_train, y_train, emb_dev, y_dev, c))
    aucs = np.array([r.dev_auc for r in results])
    return results, float(aucs.mean()), float(aucs.std(ddof=1) if len(aucs) > 1 else 0.0)


def seed_averaged_p_tb(results: List[TrainResult], emb):
    """Mean TB probability across seeds — the value exported to C08."""
    ps = np.stack([predict_p_tb(r, emb)[0] for r in results], axis=0)
    return ps.mean(axis=0)
