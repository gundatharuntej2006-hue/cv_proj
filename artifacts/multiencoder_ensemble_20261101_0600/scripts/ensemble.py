# -*- coding: utf-8 -*-
"""Multi-encoder regional ensemble, exactly as PROTOCOL.md.

    py -3 ensemble.py --stage embed     # D2 and S regional caches (resumable)
    py -3 ensemble.py --stage fit       # heads, ensembles, statistics
    py -3 ensemble.py --stage all
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
CACHE = RUN / "cache"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import p2_32_mil_sweep as S                                         # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

REG = ROOT / "artifacts/p2/regional"
CALEMB = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/regional_tbx11k_calibration_lungs_g12_v1.npz"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
VSPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VCSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
TBX_IMG = ROOT / "data/processed/v1/tbx11k_lungs512"
VIE_IMG = ROOT / "data/processed/v1/vietnam_img512_lungs"
TBX_RAW = ROOT / "data/processed/v1/tbx11k/img512"
MASKS = ROOT / "data/processed/v1/masks"
CAL_IMG = CACHE / "lungs512_calibration"      # calibration has no pre-masked folder; built here


def lungs_png(stem):
    """Same rule as the R12 calibration embedding: outside-lung pixels -> image mean."""
    from PIL import Image
    dst = CAL_IMG / (stem + ".png")
    if dst.exists():
        return dst
    a = np.array(Image.open(TBX_RAW / (stem + ".png")).convert("L"), np.uint8)
    mp = MASKS / (stem + ".png")
    if mp.exists():
        m = np.array(Image.open(mp).convert("L"), np.uint8) > 127
        if m.sum():
            o = a.astype(np.float32)
            o[~m] = float(o.mean())
            a = np.clip(o, 0, 255).astype(np.uint8)
    CAL_IMG.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(dst)
    return dst
ENCODERS = {
    "D2": ("microsoft/rad-dino-maira-2", "a6132efb9268859545e9f68be18a76a6473f174a", "dinov2"),
    "S": ("StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli", "e2ad64eee3c2a2ec746f4c0934b018c718917714", "siglip"),
}
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 4000
BOOT_SEED = 0
MIN_EFFECT = 0.02
G = 12
TB = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE = ("active_tb", "active_latent_tb")
REPRO_D = 0.9498


class Blocker(SystemExit):
    pass


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    p = TAB / "protocol_hash.txt"
    if p.is_file() and p.read_text(encoding="utf-8").split()[0] != h:
        raise Blocker("PROTOCOL.md changed after hashing")
    TAB.mkdir(exist_ok=True)
    p.write_text(h + "  PROTOCOL.md\n", encoding="utf-8")
    return h


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ embedding
class Embedder:
    """Frozen encoder -> G x G map of L2-normalised region vectors (prefix tokens dropped)."""

    def __init__(self, key, batch=16):
        import torch
        from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel
        name, rev, kind = ENCODERS[key]
        self.torch = torch
        torch.backends.cudnn.benchmark = False
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(name, revision=rev)
        m = (SiglipVisionModel if kind == "siglip" else AutoModel).from_pretrained(name, revision=rev)
        self.half = self.dev == "cuda"
        self.model = (m.half() if self.half else m.float()).eval().to(self.dev)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.g = int(m.config.image_size // m.config.patch_size)
        self.batch = batch
        self.n_params = sum(p.numel() for p in m.parameters())

    def embed(self, images):
        torch = self.torch
        import torch.nn.functional as F
        rgb = [np.stack([im] * 3, axis=-1) for im in images]
        x = self.proc(images=rgb, return_tensors="pt")["pixel_values"]
        x = x.to(self.dev, dtype=torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            tok = self.model(pixel_values=x).last_hidden_state
            n_prefix = tok.shape[1] - self.g * self.g
            if n_prefix < 0:
                raise Blocker("token grid mismatch")
            pat = tok[:, n_prefix:, :].float()
            b, _, c = pat.shape
            grid = pat.reshape(b, self.g, self.g, c).permute(0, 3, 1, 2)
            reg = F.adaptive_avg_pool2d(grid, G).permute(0, 2, 3, 1).reshape(b, G * G, c)
            reg = reg / reg.norm(dim=2, keepdim=True).clamp_min(1e-12)
        return reg.cpu().numpy().astype(np.float16)


def embed_set(emb, key, name, paths, log):
    from PIL import Image
    out = CACHE / ("regional_%s_%s_lungs_g12.npz" % (key, name))
    if out.exists():
        log("  cached %s" % out.name)
        return
    ids, vecs, buf, bids = [], [], [], []
    t0 = time.perf_counter()
    for p in paths:
        buf.append(np.array(Image.open(p).convert("L"), dtype=np.uint8))
        bids.append(p.stem)
        if len(buf) >= emb.batch:
            vecs.append(emb.embed(buf))
            ids.extend(bids)
            buf, bids = [], []
    if buf:
        vecs.append(emb.embed(buf))
        ids.extend(bids)
    arr = np.concatenate(vecs)
    CACHE.mkdir(exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), reg=arr)
    os.replace(tmp, out)
    dt = time.perf_counter() - t0
    log("  %s %-8s %5d images %s  %.1f min  %.3f s/img" % (key, name, len(ids), arr.shape, dt / 60, dt / len(ids)))
    return dt / len(ids)


def sets():
    import pandas as pd
    c01 = pd.read_csv(C01)
    forbidden = set(c01[c01.split == "internal_test"].image_id)
    tr = c01[(c01.split == "train") & c01.category.isin(TB + ("healthy",))]
    dv = c01[c01.split == "development"]
    cal = c01[c01.split == "calibration"]
    sp = json.loads(VSPLIT.read_text(encoding="utf-8"))
    tune, report = [str(i) for i in sp["tune"]], {str(i) for i in sp["report"]}
    out = {"tbx_train": [TBX_IMG / (i + ".png") for i in tr.image_id],
           "tbx_dev": [TBX_IMG / (i + ".png") for i in dv.image_id],
           "tbx_cal": [lungs_png(i) for i in cal.image_id],
           "vie_tune": [VIE_IMG / (i + ".png") for i in tune]}
    for k, v in out.items():
        stems = {p.stem for p in v}
        if stems & forbidden or stems & report:
            raise Blocker("forbidden partition in %s" % k)
        missing = [p for p in v if not p.is_file()]
        if missing:
            raise Blocker("%d missing images in %s" % (len(missing), k))
    return out, tr, dv, cal, tune


def stage_embed(log):
    import torch
    paths, *_ = sets()
    timing = []
    for key in ENCODERS:
        if all((CACHE / ("regional_%s_%s_lungs_g12.npz" % (key, n))).exists() for n in paths):
            log("%s: all cached" % key)
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        emb = Embedder(key)
        log("%s loaded: grid %d, %.1f M params" % (key, emb.g, emb.n_params / 1e6))
        for n, ps in paths.items():
            spi = embed_set(emb, key, n, ps, log)
            if spi:
                timing.append({"encoder": key, "set": n, "n": len(ps), "s_per_img": round(spi, 4)})
        if torch.cuda.is_available():
            timing.append({"encoder": key, "set": "peak_gpu_gb", "n": 0,
                           "s_per_img": round(torch.cuda.max_memory_allocated() / 1e9, 3)})
        del emb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if timing:
        write_csv(TAB / "encode_timing.csv", timing)


# ------------------------------------------------------------------ fitting
def train(Xtr, ytr, Xdv, ydv, seed, epochs=120, lr=3e-3, bs=256, wd=1e-4):
    """Identical to the R17 / R19 trainer, which reproduced production."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.tensor(Xdv, device=dev)
    lin = torch.nn.Linear(xt.shape[2], 1).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    prior = float(yt.mean())
    adj = float(np.log(max(prior, 1e-6) / max(1 - prior, 1e-6)))
    n = len(yt)
    best, best_state, patience = -1.0, None, 0
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        lin.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            z = lin(xt[k].float()).squeeze(-1)
            loss = bce(z.max(1).values - adj, yt[k])
            loss.backward()
            opt.step()
        lin.eval()
        with torch.no_grad():
            sv = torch.cat([lin(xv[i:i + 512].float()).squeeze(-1)
                            for i in range(0, len(xv), 512)]).cpu().numpy()
        a = SEL.auc(ydv, sv.max(1))
        if a > best + 1e-5:
            best, patience = a, 0
            best_state = {k2: v.detach().clone() for k2, v in lin.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break
    lin.load_state_dict(best_state)
    del xt, xv
    return lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(), best


def image_logits(X, W, b, chunk=256):
    return np.concatenate([(X[i:i + chunk].astype(np.float32) @ W.T + b)[:, :, 0].max(1)
                           for i in range(0, len(X), chunk)])


def boot_auc(y, p, rng, n=N_BOOT):
    v = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        v.append(SEL.auc(y[k], p[k]))
    return float(SEL.auc(y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired(y, pa, pb, rng, n=N_BOOT):
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], pb[k]) - SEL.auc(y[k], pa[k]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(SEL.auc(y, pb) - SEL.auc(y, pa)), float(lo), float(hi)


def auprc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def thr_at_spec(y, s, spec):
    neg = np.sort(s[y == 0])
    return float(neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)])


def verdict(delta, lo, hi):
    if (hi - lo) / 2 > 0.05:
        return "INCONCLUSIVE"
    return "ACCEPT" if (delta >= MIN_EFFECT and lo > 0) else "REJECT"


def load_cache(path, ids):
    with np.load(path) as z:
        zi, X = z["image_id"].tolist(), z["reg"]
    idx = {i: k for k, i in enumerate(zi)}
    return X[[idx[i] for i in ids]]


def stage_fit(log, proto):
    import pandas as pd
    t0 = time.perf_counter()
    _, tr, dv, cal, tune = sets()
    vdf = pd.read_csv(VCSV).set_index("pid")
    y_tune = vdf.loc[[int(i) for i in tune], "xpert_result"].to_numpy().astype(int)
    ytr = np.array([1 if c in TB else 0 for c in tr.category])
    ydv = np.array([1 if c in TB else 0 for c in dv.category])
    y_cal = np.array([1 if c in ACTIVE else 0 for c in cal.category])
    log("train %d dev %d cal %d (active %d) tune %d (xpert+ %d)" % (
        len(tr), len(dv), len(cal), y_cal.sum(), len(tune), y_tune.sum()))

    # features per encoder: dict set -> [n, 144, d]
    feats = {}
    feats["D"] = {"tr": S.repool(load_cache(REG / "regional_tbx11k_lungs_g12_v1.npz", tr.image_id), 12, G),
                  "dv": S.repool(load_cache(REG / "regional_tbx11k_lungs_g12_v1.npz", dv.image_id), 12, G),
                  "cal": S.repool(load_cache(CALEMB, cal.image_id), 12, G),
                  "tune": S.repool(load_cache(REG / "regional_vietnam_lungs_g12_v1.npz", tune), 12, G)}
    for key in ENCODERS:
        feats[key] = {s: load_cache(CACHE / ("regional_%s_%s_lungs_g12.npz" % (key, n)), ids)
                      for s, n, ids in (("tr", "tbx_train", tr.image_id), ("dv", "tbx_dev", dv.image_id),
                                        ("cal", "tbx_cal", cal.image_id), ("tune", "vie_tune", tune))}
    feats["C3"] = {s: np.concatenate([feats[k][s] for k in ("D", "D2", "S")], axis=2) for s in ("tr", "dv", "cal", "tune")}

    out, per_seed = {}, []
    for arm in ("D", "D2", "S", "C3"):
        F = feats[arm]
        log("\narm %-3s d=%d" % (arm, F["tr"].shape[2]))
        Ws = []
        for sd in SEEDS:
            W, b, best = train(F["tr"], ytr, F["dv"], ydv, sd)
            Ws.append((W, b))
            per_seed.append({"arm": arm, "seed": sd, "early_stop_dev_auc": round(best, 4),
                             "cal_auroc_vs_active": round(SEL.auc(y_cal, image_logits(F["cal"], W, b)), 4)})
            log("   seed %2d dev %.4f" % (sd, best))
        out[arm] = {s: np.mean([image_logits(F[s], W, b) for W, b in Ws], 0) for s in ("dv", "cal", "tune")}
    del feats["C3"]
    write_csv(TAB / "per_seed.csv", per_seed)

    here = SEL.auc(y_cal, out["D"]["cal"])
    ok = abs(here - REPRO_D) <= 5e-4
    write_csv(TAB / "reproduction_gate.csv", [{"arm": "D", "expected": REPRO_D, "here": round(here, 4), "pass": ok}])
    log("gate D expected %.4f here %.4f %s" % (REPRO_D, here, "pass" if ok else "FAIL"))
    if not ok:
        raise Blocker("reproduction gate failed")

    # standardise each single-encoder head by its development mean / sd, then average
    std = {k: (float(out[k]["dv"].mean()), float(out[k]["dv"].std())) for k in ("D", "D2", "S")}
    z = {k: {s: (out[k][s] - std[k][0]) / std[k][1] for s in ("dv", "cal", "tune")} for k in std}
    for name, ks in (("E3", ("D", "D2", "S")), ("E_DS", ("D", "S")), ("E_DD2", ("D", "D2")), ("E_D2S", ("D2", "S"))):
        out[name] = {s: np.mean([z[k][s] for k in ks], 0) for s in ("dv", "cal", "tune")}
    (TAB / "standardisation.json").write_text(json.dumps(
        {k: {"dev_mean": round(m, 4), "dev_sd": round(sd_, 4)} for k, (m, sd_) in std.items()}, indent=1), encoding="utf-8")

    metrics = []
    for arm in ("D", "D2", "S", "E3", "E_DS", "E_DD2", "E_D2S", "C3"):
        ac, lc, hc = boot_auc(y_cal, out[arm]["cal"], np.random.default_rng(BOOT_SEED))
        at, lt, ht = boot_auc(y_tune, out[arm]["tune"], np.random.default_rng(BOOT_SEED))
        thr = thr_at_spec(ydv, out[arm]["dv"], 0.90)
        sc = out[arm]["cal"]
        metrics.append({"arm": arm, "dev_auroc": round(SEL.auc(ydv, out[arm]["dv"]), 4),
                        "cal_auroc_vs_active": round(ac, 4), "cal_ci": "[%.4f, %.4f]" % (lc, hc),
                        "cal_auprc": round(auprc(y_cal, sc), 4),
                        "cal_sens_at_dev_spec0.90": round(float((sc[y_cal == 1] > thr).mean()), 4),
                        "cal_spec_at_dev_spec0.90": round(float((sc[y_cal == 0] <= thr).mean()), 4),
                        "tune_auroc_EXPLORATORY": round(at, 4), "tune_ci": "[%.4f, %.4f]" % (lt, ht)})
        log("%-5s dev %.4f cal %.4f [%.4f, %.4f] tune %.4f [%.4f, %.4f]" % (arm, metrics[-1]["dev_auroc"], ac, lc, hc, at, lt, ht))
    write_csv(TAB / "metrics_by_arm.csv", metrics)

    diffs = []
    for arm in ("E3", "D2", "S", "E_DS", "E_DD2", "E_D2S", "C3"):
        dc, lc, hc = paired(y_cal, out["D"]["cal"], out[arm]["cal"], np.random.default_rng(BOOT_SEED))
        dt, lt, ht = paired(y_tune, out["D"]["tune"], out[arm]["tune"], np.random.default_rng(BOOT_SEED))
        primary = arm == "E3"
        v = verdict(dc, lc, hc) if primary else ""
        diffs.append({"contrast": "%s - D" % arm, "primary": primary, "cal_delta": round(dc, 4),
                      "cal_ci": "[%+.4f, %+.4f]" % (lc, hc), "tune_delta_EXPLORATORY": round(dt, 4),
                      "tune_ci": "[%+.4f, %+.4f]" % (lt, ht), "verdict": v})
        log("%s - D: cal %+.4f [%+.4f, %+.4f] | tune %+.4f [%+.4f, %+.4f] %s" % (arm, dc, lc, hc, dt, lt, ht, v))
    write_csv(TAB / "paired_differences.csv", diffs)

    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "encoders": {k: {"model": v[0], "revision": v[1]} for k, v in ENCODERS.items()},
        "rad_dino_revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
        "seeds": SEEDS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "calibration_n": int(len(y_cal)), "calibration_pos": int(y_cal.sum()),
        "tune_n": int(len(y_tune)), "tune_pos": int(y_tune.sum()),
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "fit_elapsed_s": round(time.perf_counter() - t0, 1)}, indent=1),
        encoding="utf-8")
    log("\nfit done in %.1f s" % (time.perf_counter() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("embed", "fit", "all"), default="all")
    a = ap.parse_args()
    logf = open(RUN / "run.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    proto = check_protocol()
    if a.stage in ("embed", "all"):
        stage_embed(log)
    if a.stage in ("fit", "all"):
        stage_fit(log, proto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
