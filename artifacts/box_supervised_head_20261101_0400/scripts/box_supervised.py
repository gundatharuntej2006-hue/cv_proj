# -*- coding: utf-8 -*-
"""Box-supervised regional head: three arms, one fitting path, exactly as PROTOCOL.md.

    py -3 box_supervised.py           # dry: geometry, label counts, guards
    py -3 box_supervised.py --run
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import p2_32_mil_sweep as S                                         # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

REG = ROOT / "artifacts/p2/regional"
CALEMB = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/regional_tbx11k_calibration_lungs_g12_v1.npz"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
ANN = ROOT / "data/raw/tbx11k/TBX11K/annotations/json"
MASKS = ROOT / "data/processed/v1/masks"
VSPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VCSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 4000
BOOT_SEED = 0
MIN_EFFECT = 0.02
LAMBDA = 1.0
G, PATCH, IN_PX, SRC_PX = 12, 14, 518, 512
BINS = [(int(np.floor(i * 37 / G)), int(np.ceil((i + 1) * 37 / G))) for i in range(G)]
TB = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE = ("active_tb", "active_latent_tb")
REPRO = {"A": 0.9498, "B1": 0.9460}      # R12 calibration AUROC vs active_tb_label


class Blocker(SystemExit):
    pass


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    if h != (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]:
        raise Blocker("PROTOCOL.md changed after hashing")
    return h


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ------------------------------------------------------------------ geometry
def box_to_regions(bbox, w, h):
    """[x, y, bw, bh] in original pixels -> set of region indices (row*12+col) it touches."""
    if (w, h) != (SRC_PX, SRC_PX):
        raise Blocker("expected 512x512 originals, got %dx%d" % (w, h))
    x0, y0, bw, bh = bbox
    x1, y1 = x0 + bw, y0 + bh
    k = IN_PX / SRC_PX
    px0, px1 = int(np.floor(x0 * k / PATCH)), int(np.ceil(x1 * k / PATCH)) - 1
    py0, py1 = int(np.floor(y0 * k / PATCH)), int(np.ceil(y1 * k / PATCH)) - 1
    px0, py0 = max(px0, 0), max(py0, 0)
    px1, py1 = min(px1, 36), min(py1, 36)
    cols = [i for i, (s, e) in enumerate(BINS) if not (e <= px0 or s > px1)]
    rows = [i for i, (s, e) in enumerate(BINS) if not (e <= py0 or s > py1)]
    return {r * G + c for r in rows for c in cols}


def region_lung_fraction(mask):
    """mask: 512x512 bool -> [144] fraction of each region's footprint inside the lungs."""
    k = SRC_PX / IN_PX
    out = np.zeros(G * G)
    for r, (rs, re) in enumerate(BINS):
        for c, (cs, ce) in enumerate(BINS):
            y0, y1 = int(rs * PATCH * k), int(min(re * PATCH, IN_PX) * k)
            x0, x1 = int(cs * PATCH * k), int(min(ce * PATCH, IN_PX) * k)
            sub = mask[y0:y1, x0:x1]
            out[r * G + c] = float(sub.mean()) if sub.size else 0.0
    return out


def load_boxes():
    boxes = defaultdict(list)             # image_id -> [(cat, bbox, w, h)]
    for stem in ("TBX11K_train", "TBX11K_val"):
        d = json.loads((ANN / (stem + ".json")).read_text(encoding="utf-8"))
        name = {im["id"]: im["file_name"].split("/")[-1].rsplit(".", 1)[0] for im in d["images"]}
        for a in d["annotations"]:
            boxes[name[a["image_id"]]].append((a["category_id"], a["bbox"], a["width"], a["height"]))
    return boxes


def region_labels(ids, cats, boxes):
    """-> [n, 144] float with +1 / 0 / nan(masked), per PROTOCOL section 3."""
    Y = np.full((len(ids), G * G), np.nan, np.float32)
    n_act = n_obs = 0
    for i, (iid, cat) in enumerate(zip(ids, cats)):
        if cat == "healthy":
            Y[i] = 0.0
            continue
        act, obs = set(), set()
        for c, bb, w, h in boxes.get(iid, []):
            if c == 3:
                raise Blocker("category 3 box encountered on %s" % iid)
            (act if c == 1 else obs).update(box_to_regions(bb, w, h))
        for r in obs - act:
            Y[i, r] = 0.0
        for r in act:
            Y[i, r] = 1.0
        n_act += len(act)
        n_obs += len(obs - act)
    return Y, n_act, n_obs


# ------------------------------------------------------------------ training
def train(Xtr, ytr, Ytr, Xdv, ydv, seed, lam, epochs=120, lr=3e-3, bs=256, wd=1e-4):
    """S.train with an added region BCE term. lam = 0 must reproduce S.train exactly."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Yr = torch.tensor(np.nan_to_num(Ytr, nan=0.0), dtype=torch.float32, device=dev)
    Mr = torch.tensor(~np.isnan(Ytr), dtype=torch.float32, device=dev)
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
            z = lin(xt[k].float()).squeeze(-1)                    # [b, 144]
            loss = bce(z.max(1).values - adj, yt[k])
            if lam > 0:
                m = Mr[k]
                if m.sum() > 0:
                    lr_ = (bce(z, Yr[k], reduction="none") * m).sum() / m.sum()
                    loss = loss + lam * lr_
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
    return lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(), best


def region_logits(X, W, b, chunk=512):
    return np.concatenate([(X[i:i + chunk].astype(np.float32) @ W.T + b)[:, :, 0]
                           for i in range(0, len(X), chunk)])


# ------------------------------------------------------------------ stats
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    from PIL import Image
    t0 = time.perf_counter()
    proto = check_protocol()

    c01 = pd.read_csv(C01)
    forbidden = set(c01[c01.split == "internal_test"].image_id)
    with np.load(REG / "regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    with np.load(CALEMB) as z:
        cid, CX = z["image_id"].tolist(), z["reg"]
    if forbidden & (set(tid) | set(cid)):
        raise Blocker("internal_test present in a feature file")
    ti, ci = {i: k for k, i in enumerate(tid)}, {i: k for k, i in enumerate(cid)}
    tr = c01[(c01.split == "train") & c01.image_id.isin(ti) & c01.category.isin(TB + ("healthy",))]
    dv = c01[(c01.split == "development") & c01.image_id.isin(ti)]
    cal = c01[(c01.split == "calibration") & c01.image_id.isin(ci)]
    assert (len(tr), len(dv), len(cal)) == (3241, 659, 899)

    boxes = load_boxes()
    Ytr, n_act, n_obs = region_labels(tr.image_id.tolist(), tr.category.tolist(), boxes)
    lab = ~np.isnan(Ytr)
    print("region labels (train): %d images | labelled regions %d | +1 %d | 0 %d | masked %d"
          % (len(tr), lab.sum(), int(np.nansum(Ytr)), int(lab.sum() - np.nansum(Ytr)),
             int((~lab).sum())))
    print("  active-box regions %d, obsolete-only regions %d" % (n_act, n_obs))
    tb_tr = tr[tr.category.isin(TB)]
    n_boxed = sum(1 for i in tb_tr.image_id if boxes.get(i))
    print("  TB train images %d, with boxes %d" % (len(tb_tr), n_boxed))

    # localisation set: development TB images with >= 1 active box
    dv_loc = [(i, c) for i, c in zip(dv.image_id, dv.category)
              if any(b[0] == 1 for b in boxes.get(i, []))]
    print("localisation set: %d development images with an active box" % len(dv_loc))
    geom = {"patch_grid": 37, "pooled_grid": G, "patch_px": PATCH, "input_px": IN_PX,
            "source_px": SRC_PX, "scale": IN_PX / SRC_PX, "bins_37_to_12": BINS,
            "train_region_labels": {"labelled": int(lab.sum()), "positive": int(np.nansum(Ytr)),
                                    "masked": int((~lab).sum())},
            "localisation_n": len(dv_loc)}
    (TAB / "geometry_and_labels.json").write_text(json.dumps(geom, indent=1), encoding="utf-8")
    if not a.run:
        print("\nDRY. Pass --run.")
        return 0

    Xtr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, G)
    Xdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, G)
    Xcal = S.repool(CX[[ci[i] for i in cal.image_id]], 12, G)
    cat_dv, cat_cal = dv.category.tolist(), cal.category.tolist()
    y_cal = np.array([1 if c in ACTIVE else 0 for c in cat_cal])
    y_dv_act = np.array([1 if c in ACTIVE else 0 for c in cat_dv])

    with np.load(REG / "regional_vietnam_lungs_g12_v1.npz") as z:
        vid, VX = z["image_id"].tolist(), z["reg"]
    sp = json.loads(VSPLIT.read_text(encoding="utf-8"))
    tune, report = set(sp["tune"]), set(sp["report"])
    vsel = [k for k, i in enumerate(vid) if int(i) in tune]
    if {int(vid[k]) for k in vsel} & report:
        raise Blocker("REPORT leaked")
    vdf = pd.read_csv(VCSV).set_index("pid")
    y_tune = vdf.loc[[int(vid[k]) for k in vsel], "xpert_result"].to_numpy().astype(int)
    Xtune = S.repool(VX[vsel], 12, G)

    ARMS = {"A": (TB, 0.0), "B1": (ACTIVE, 0.0), "R": (ACTIVE, LAMBDA)}
    out, per_seed = {}, []
    for arm, (pos, lam) in ARMS.items():
        ytr = np.array([1 if c in pos else 0 for c in tr.category])
        ydv = np.array([1 if c in pos else 0 for c in cat_dv])
        Ws = []
        print("\narm %-2s target=%s lambda=%.1f" % (arm, "tb" if pos == TB else "active", lam))
        for sd in SEEDS:
            W, b, best = train(Xtr, ytr, Ytr, Xdv, ydv, sd, lam)
            Ws.append((W, b))
            per_seed.append({"arm": arm, "seed": sd, "early_stop_dev_auc": round(best, 4),
                             "cal_auroc_vs_active": round(SEL.auc(y_cal, region_logits(Xcal, W, b).max(1)), 4)})
            print("   seed %2d dev %.4f" % (sd, best))
        out[arm] = {"Ws": Ws,
                    "cal": np.mean([region_logits(Xcal, W, b).max(1) for W, b in Ws], 0),
                    "dv": np.mean([region_logits(Xdv, W, b).max(1) for W, b in Ws], 0),
                    "tune": np.mean([region_logits(Xtune, W, b).max(1) for W, b in Ws], 0),
                    "dv_regions": np.mean([region_logits(Xdv, W, b) for W, b in Ws], 0)}

    # correctness gate: reproduce R12 arms
    gate = []
    ok = True
    for arm, want in REPRO.items():
        got = SEL.auc(y_cal, out[arm]["cal"])
        p = abs(got - want) <= 0.0005
        ok &= p
        gate.append({"arm": arm, "r12_cal_auroc": want, "here": round(got, 4), "pass": p})
        print("gate %s: R12 %.4f here %.4f %s" % (arm, want, got, "PASS" if p else "FAIL"))
    write_csv(TAB / "reproduction_gate.csv", gate)
    if not ok:
        raise Blocker("arms A/B1 do not reproduce R12; the trainer is not production-equivalent")

    # metrics
    rng = np.random.default_rng(BOOT_SEED)
    metrics = []
    for arm in ARMS:
        p = out[arm]["cal"]
        au, lo, hi = boot_auc(y_cal, p, np.random.default_rng(BOOT_SEED))
        thr = float(np.quantile(out[arm]["dv"][y_dv_act == 0], 0.90))
        pred = p >= thr
        at, lt, ht = boot_auc(y_tune, out[arm]["tune"], np.random.default_rng(BOOT_SEED))
        metrics.append({"arm": arm, "cal_auroc_vs_active": round(au, 4), "cal_ci": "[%.4f, %.4f]" % (lo, hi),
                        "cal_auprc": round(auprc(y_cal, p), 4), "cal_prevalence": round(float(y_cal.mean()), 4),
                        "dev_threshold_spec0.90": round(thr, 4),
                        "cal_sens_active": round(float(pred[y_cal == 1].mean()), 4),
                        "cal_spec": round(float((~pred[y_cal == 0]).mean()), 4),
                        "tune_auroc_EXPLORATORY": round(at, 4), "tune_ci": "[%.4f, %.4f]" % (lt, ht)})
    write_csv(TAB / "metrics_by_arm.csv", metrics)
    write_csv(TAB / "per_seed.csv", per_seed)

    pr = []
    for a_, b_ in (("B1", "R"), ("A", "R"), ("A", "B1")):
        d, lo, hi = paired(y_cal, out[a_]["cal"], out[b_]["cal"], np.random.default_rng(BOOT_SEED))
        dt, lt, ht = paired(y_tune, out[a_]["tune"], out[b_]["tune"], np.random.default_rng(BOOT_SEED))
        hw = (hi - lo) / 2
        verdict = ("INCONCLUSIVE" if hw > 0.05 else
                   ("ACCEPT" if (d >= MIN_EFFECT and lo > 0) else "REJECT")) if (a_, b_) == ("B1", "R") else ""
        pr.append({"contrast": "%s - %s" % (b_, a_), "primary": (a_, b_) == ("B1", "R"),
                   "cal_delta": round(d, 4), "cal_ci": "[%+.4f, %+.4f]" % (lo, hi),
                   "cal_halfwidth": round(hw, 4), "excludes_zero": bool(lo > 0 or hi < 0),
                   "meets_0.02": bool(d >= MIN_EFFECT and lo > 0), "verdict": verdict,
                   "tune_delta_EXPLORATORY": round(dt, 4), "tune_ci": "[%+.4f, %+.4f]" % (lt, ht)})
        print("%s - %s: cal %+.4f [%+.4f, %+.4f] | tune %+.4f [%+.4f, %+.4f] %s"
              % (b_, a_, d, lo, hi, dt, lt, ht, verdict))
    write_csv(TAB / "paired_differences.csv", pr)

    # localisation on development TB images with an active box
    dvi = {i: k for k, i in enumerate(dv.image_id)}
    rows_loc, rng_l = [], np.random.default_rng(BOOT_SEED)
    active_sets, lung_regions = [], []
    for iid, _ in dv_loc:
        act = set()
        for c, bb, w, h in boxes[iid]:
            if c == 1:
                act |= box_to_regions(bb, w, h)
        active_sets.append(act)
        mp = MASKS / (iid + ".png")
        m = np.array(Image.open(mp).convert("L"), np.uint8) > 127 if mp.exists() else np.ones((512, 512), bool)
        lung_regions.append(np.where(region_lung_fraction(m) > 0.5)[0])
    idx_loc = [dvi[i] for i, _ in dv_loc]

    def hits(top_regions_per_image, k):
        return np.array([len(set(tr_[:k]) & act) > 0 for tr_, act in zip(top_regions_per_image, active_sets)], float)

    def ci(v, n=2000):
        b = [v[rng_l.integers(0, len(v), len(v))].mean() for _ in range(n)]
        return round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)

    for arm in ARMS:
        Rl = out[arm]["dv_regions"][idx_loc]
        tops = [list(np.argsort(-r)) for r in Rl]
        for k in (1, 3):
            h = hits(tops, k)
            rows_loc.append({"arm": arm, "metric": "top-%d hit" % k, "n": len(h),
                             "hit_rate": round(float(h.mean()), 3), "ci95": "[%s, %s]" % ci(h)})
    # baselines
    rand_tops = [list(rng_l.permutation(G * G)) for _ in dv_loc]
    centre = [[6 * G + 6, 5 * G + 5, 6 * G + 5] for _ in dv_loc]
    lung_tops = [list(rng_l.permutation(lr)) if len(lr) else list(rng_l.permutation(G * G)) for lr in lung_regions]
    for name, tops in (("random region", rand_tops), ("centre region", centre), ("random lung region", lung_tops)):
        for k in (1, 3):
            h = hits(tops, k)
            rows_loc.append({"arm": "baseline: " + name, "metric": "top-%d hit" % k, "n": len(h),
                             "hit_rate": round(float(h.mean()), 3), "ci95": "[%s, %s]" % ci(h)})
    box_frac = float(np.mean([len(s) / (G * G) for s in active_sets]))
    rows_loc.append({"arm": "chance: mean active-box coverage", "metric": "fraction of regions",
                     "n": len(dv_loc), "hit_rate": round(box_frac, 3), "ci95": ""})
    write_csv(TAB / "localisation.csv", rows_loc)
    print("\nlocalisation (n=%d):" % len(dv_loc))
    for r in rows_loc:
        print("  %-34s %-10s %s %s" % (r["arm"], r["metric"], r["hit_rate"], r["ci95"]))

    import platform, torch
    (TAB / "run_manifest.json").write_text(json.dumps({
        "run": RUN.name, "protocol_sha256": proto, "lambda": LAMBDA, "seeds": list(SEEDS),
        "n_boot": N_BOOT, "geometry": geom,
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "primary_verdict": [r["verdict"] for r in pr if r["primary"]][0],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }, indent=1), encoding="utf-8")
    print("\ndone in %.1f min" % ((time.perf_counter() - t0) / 60))


if __name__ == "__main__":
    sys.exit(main())
