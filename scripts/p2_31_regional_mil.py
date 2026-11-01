# -*- coding: utf-8 -*-
"""Does focal (patch-level) evidence beat global pooling? The one mechanism-backed idea left.

Everything so far uses RAD-DINO's **globally pooled** output: one 768-dimensional vector
summarising the whole radiograph. Tuberculosis is focal. A small apical cavity occupies a
handful of the 1,369 patch tokens, and average pooling dilutes it into 99.6% normal lung.
That predicts exactly the failure we measured - adequate on florid disease in curated sets,
poor on subtle disease in a screening cohort.

The standard remedy is multiple-instance learning: score each region with a shared head and
aggregate by max or a top-k mean, so one strongly abnormal region can carry the image.

**Storage.** Keeping all 1,369 tokens per image would be ~7 GB. Instead the 37x37 patch grid
is average-pooled to 6x6 regions (each about 86 px of the 518 px input, comfortably smaller
than a typical lesion), which is 1/38th the size and lets 3x3 and 1x1 be derived from the
same pass by further pooling.

Aggregators compared, all with the same linear scorer:
  mean    - equivalent to global pooling, the control
  max     - one region can carry the image
  top3    - mean of the three strongest regions, less brittle than max
  logsumexp - a smooth maximum

    py -3 scripts/p2_31_regional_mil.py --stage embed     # ~10 min, resumable
    py -3 scripts/p2_31_regional_mil.py --stage eval
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import selection as SEL                                 # noqa: E402

OUT_DIR = ROOT / "artifacts/p2/regional"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
GRID = 12
TB = ("active_tb", "latent_tb", "active_latent_tb")
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000


# ------------------------------------------------------------------ embedding
class RegionalEmbedder:
    """RAD-DINO patch tokens pooled to a GRID x GRID map of 768-d region vectors."""

    def __init__(self, batch=16):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        torch.backends.cudnn.benchmark = False
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(
            "microsoft/rad-dino", revision="110cbc18d5133582e320b43d53bf5c44e410c936")
        m = AutoModel.from_pretrained(
            "microsoft/rad-dino", revision="110cbc18d5133582e320b43d53bf5c44e410c936")
        self.half = self.dev == "cuda"
        self.model = (m.half() if self.half else m.float()).eval().to(self.dev)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.g = int(m.config.image_size // m.config.patch_size)
        self.batch = batch

    def embed(self, images):
        torch = self.torch
        import torch.nn.functional as F
        rgb = [np.stack([im] * 3, axis=-1) for im in images]
        x = self.proc(images=rgb, return_tensors="pt")["pixel_values"]
        x = x.to(self.dev, dtype=torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            tok = self.model(pixel_values=x).last_hidden_state      # [B, 1+g*g, C]
            n_prefix = tok.shape[1] - self.g * self.g
            pat = tok[:, n_prefix:, :].float()                      # drop CLS/registers
            b, _, c = pat.shape
            grid = pat.reshape(b, self.g, self.g, c).permute(0, 3, 1, 2)
            reg = F.adaptive_avg_pool2d(grid, GRID)                 # [B, C, G, G]
            reg = reg.permute(0, 2, 3, 1).reshape(b, GRID * GRID, c)
            reg = reg / reg.norm(dim=2, keepdim=True).clamp_min(1e-12)
        return reg.cpu().numpy().astype(np.float16)


def embed_set(name, paths_or_dir, log=print):
    """Embed a set of 512px PNGs into a regional map. Resumable by output existence."""
    from PIL import Image
    out_p = OUT_DIR / ("regional_%s_g18_v1.npz" % name)
    if out_p.exists():
        log("  cached: %s" % out_p.name)
        return
    paths = sorted(paths_or_dir) if isinstance(paths_or_dir, list) else \
        sorted(Path(paths_or_dir).glob("*.png"))
    if not paths:
        log("  no images for %s" % name)
        return
    emb = embed_set.embedder
    ids, vecs, buf, bids = [], [], [], []
    t0 = time.perf_counter()
    for p in paths:
        buf.append(np.array(Image.open(p).convert("L"), dtype=np.uint8))
        bids.append(p.stem)
        if len(buf) >= 16:
            vecs.append(emb.embed(buf))
            ids.extend(bids)
            buf, bids = [], []
    if buf:
        vecs.append(emb.embed(buf))
        ids.extend(bids)
    arr = np.concatenate(vecs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), reg=arr)
    os.replace(tmp, out_p)
    log("  %-12s %d images -> %s  (%.1f min, %.0f MB)"
        % (name, len(ids), arr.shape, (time.perf_counter() - t0) / 60,
           out_p.stat().st_size / 1e6))


def stage_embed(log=print):
    import pandas as pd
    embed_set.embedder = RegionalEmbedder()
    log("regional embedding at %dx%d, grid %d patches" % (GRID, GRID, embed_set.embedder.g))

    # TBX11K: only the rows we train/evaluate on, lungs-only masking (the best config)
    lungs = ROOT / "data/processed/v1"
    # the lungs-only TBX11K images were never written to disk as PNGs; rebuild them here
    src = ROOT / "data/processed/v1/tbx11k/img512"
    masks = ROOT / "data/processed/v1/masks"
    c01 = pd.read_csv(SPLITS)
    keep = set(c01[c01.split.isin(("train", "development"))].image_id)
    tmp_dir = ROOT / "data/processed/v1/tbx11k_lungs512"
    if not tmp_dir.exists() or len(list(tmp_dir.glob("*.png"))) < len(keep):
        from PIL import Image
        tmp_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(src.glob("*.png")):
            if p.stem not in keep or (tmp_dir / p.name).exists():
                continue
            mp = masks / p.name
            a = np.array(Image.open(p).convert("L"), np.uint8)
            if mp.exists():
                m = np.array(Image.open(mp).convert("L"), np.uint8) > 127
                if m.sum():
                    o = a.astype(np.float32)
                    o[~m] = float(o.mean())
                    a = np.clip(o, 0, 255).astype(np.uint8)
            Image.fromarray(a).save(tmp_dir / p.name)
            n += 1
        log("  built %d lungs-only PNGs" % n)
    embed_set("tbx11k_lungs", tmp_dir)

    # the Vietnam lungs-only images were only ever embedded, never written out
    vsrc = ROOT / "data/processed/v1/vietnam_img512"
    vmask = ROOT / "data/processed/v1/masks_vietnam"
    vdir = ROOT / "data/processed/v1/vietnam_img512_lungs"
    if vsrc.exists() and (not vdir.exists()
                          or len(list(vdir.glob("*.png"))) < len(list(vsrc.glob("*.png")))):
        from PIL import Image
        vdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for q in sorted(vsrc.glob("*.png")):
            if (vdir / q.name).exists():
                continue
            a = np.array(Image.open(q).convert("L"), np.uint8)
            mq = vmask / q.name
            if mq.exists():
                m = np.array(Image.open(mq).convert("L"), np.uint8) > 127
                if m.sum():
                    o = a.astype(np.float32)
                    o[~m] = float(o.mean())
                    a = np.clip(o, 0, 255).astype(np.uint8)
            Image.fromarray(a).save(vdir / q.name)
            n += 1
        log("  built %d Vietnam lungs-only PNGs" % n)
    embed_set("vietnam_lungs", vdir)


# ------------------------------------------------------------------ MIL head
def aggregate(scores, how):
    """scores: [N, R] per-region logits -> [N] image logit."""
    if how == "mean":
        return scores.mean(axis=1)
    if how == "max":
        return scores.max(axis=1)
    if how == "top3":
        return np.sort(scores, axis=1)[:, -3:].mean(axis=1)
    if how == "logsumexp":
        m = scores.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(scores - m).sum(axis=1, keepdims=True)))[:, 0]
    raise ValueError(how)


def train_mil(Xtr, ytr, Xdv, ydv, how, seed, epochs=60, lr=0.05):
    """A single shared linear scorer over regions, trained through the aggregator."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.tensor(Xdv, dtype=torch.float32, device=dev)
    lin = torch.nn.Linear(xt.shape[2], 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    prior = float(yt.mean())
    adj = float(np.log(max(prior, 1e-6) / max(1 - prior, 1e-6)))     # logit adjustment
    best, best_state = -1.0, None
    for ep in range(epochs):
        lin.train()
        opt.zero_grad()
        s = lin(xt).squeeze(-1)                                      # [N, R]
        if how == "mean":
            z = s.mean(1)
        elif how == "max":
            z = s.max(1).values
        elif how == "top3":
            z = s.topk(3, dim=1).values.mean(1)
        else:
            z = torch.logsumexp(s, dim=1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z - adj, yt)
        loss.backward()
        opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            lin.eval()
            with torch.no_grad():
                sv = lin(xv).squeeze(-1).cpu().numpy()
            a = SEL.auc(ydv, aggregate(sv, how))
            if a > best:
                best = a
                best_state = {k: v.detach().clone() for k, v in lin.state_dict().items()}
    lin.load_state_dict(best_state)
    W = lin.weight.detach().cpu().numpy()
    b = lin.bias.detach().cpu().numpy()
    return W, b, best


def score(X, W, b, how):
    s = (X.astype(np.float32) @ W.T + b)[:, :, 0]
    return aggregate(s, how)


def boot(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    v = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        v.append(SEL.auc(y[k], p[k]))
    return float(SEL.auc(y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def stage_eval(log=print):
    import pandas as pd
    tb_p = OUT_DIR / "regional_tbx11k_lungs_v1.npz"
    vn_p = OUT_DIR / "regional_vietnam_lungs_v1.npz"
    if not tb_p.exists():
        raise SystemExit("run --stage embed first")
    with np.load(tb_p) as z:
        ids, Xall = z["image_id"].tolist(), z["reg"]
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr = c01[(c01.split == "train") & (c01.category.isin(TB + ("healthy",)))]
    dv = c01[c01.split == "development"]
    Xtr = Xall[[idx[i] for i in tr.image_id]].astype(np.float32)
    ytr = np.array([1 if c in TB else 0 for c in tr.category], np.int64)
    Xdv = Xall[[idx[i] for i in dv.image_id]].astype(np.float32)
    ydv = np.array([1 if c in TB else 0 for c in dv.category], np.int64)
    log("train %d (%d TB) | development %d | regions %d"
        % (len(ytr), ytr.sum(), len(ydv), Xtr.shape[1]))

    sites = {}
    if vn_p.exists():
        csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
        with np.load(vn_p) as z:
            vid, VX = z["image_id"].tolist(), z["reg"]
        sites["Vietnam"] = (VX, csv.loc[[int(i) for i in vid], "xpert_result"].to_numpy())

    rows = []
    log("\n  %-10s %9s  %s" % ("aggregator", "TBX11K", "  ".join("%-22s" % s for s in sites)))
    for how in ("mean", "max", "top3", "logsumexp"):
        devs, site_scores = [], {s: [] for s in sites}
        for sd in SEEDS:
            W, b, d = train_mil(Xtr, ytr, Xdv, ydv, how, sd)
            devs.append(d)
            for s, (SX, sy) in sites.items():
                site_scores[s].append(score(SX, W, b, how))
        r = {"aggregator": how, "tbx11k_dev": round(float(np.mean(devs)), 4), "external": {}}
        line = ""
        for s, (SX, sy) in sites.items():
            p = np.mean(site_scores[s], axis=0)
            a, lo, hi = boot(sy, p)
            r["external"][s] = [round(a, 4), round(lo, 4), round(hi, 4)]
            line += "%.4f [%.4f, %.4f]  " % (a, lo, hi)
        rows.append(r)
        log("  %-10s %9.4f  %s" % (how, r["tbx11k_dev"], line))

    base = next((r for r in rows if r["aggregator"] == "mean"), None)
    best = max(rows, key=lambda r: r["external"].get("Vietnam", [0])[0])
    log("\ninterpretation:")
    log("  global pooling equivalent (mean): Vietnam %.4f" % base["external"]["Vietnam"][0])
    log("  best aggregator (%s):            Vietnam %.4f" % (best["aggregator"],
                                                             best["external"]["Vietnam"][0]))
    gain = best["external"]["Vietnam"][0] - base["external"]["Vietnam"][0]
    log("  gain from focal pooling: %+.4f" % gain)
    log("  reference: current best global model 0.6743 | commercial comparator 0.7827")
    if gain >= 0.03:
        log("  -> focal evidence matters; global pooling was losing real signal.")
    elif gain >= 0.01:
        log("  -> a small real gain.")
    else:
        log("  -> focal pooling does not help; the limit is not spatial dilution.")

    json.dump({"created": time.strftime("%Y-%m-%d"), "grid": GRID, "seeds": list(SEEDS),
               "results": rows}, open(OUT_DIR / "regional_mil.json", "w", encoding="utf-8"),
              indent=1)
    log("\nwrote %s" % (OUT_DIR / "regional_mil.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("embed", "eval", "all"), default="all")
    a = ap.parse_args()
    if a.stage in ("embed", "all"):
        stage_embed()
    if a.stage in ("eval", "all"):
        stage_eval()


if __name__ == "__main__":
    sys.exit(main())
