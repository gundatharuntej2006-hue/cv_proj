# -*- coding: utf-8 -*-
"""Efficiency and site-adaptation benchmark, exactly as PROTOCOL.md declares.

    py -3 benchmark.py                # dry: splits, counts, guards, no fitting
    py -3 benchmark.py --run          # everything, ~30 min
    py -3 benchmark.py --run --skip-cpu
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import p2_32_mil_sweep as S                                         # noqa: E402
import p2_31_regional_mil as R                                      # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

REG = ROOT / "artifacts/p2/regional"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
NLM_IMG = ROOT / "data/processed/v1/nlm_img512_lungs"
TB = ("active_tb", "latent_tb", "active_latent_tb")
SEED = 20261101
HEAD_SEED = 17
N_BOOT = 2000
DRAWS = (0, 1, 2)
SPEC_TARGET = 0.70
GRID = {"shz": (0, 10, 25, 50, 100, 200, "all"), "mcu": (0, 10, 25, 50, "all")}
SITE_NAME = {"shz": "Shenzhen", "mcu": "Montgomery"}


class Blocker(SystemExit):
    pass


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    if h != (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]:
        raise Blocker("PROTOCOL.md changed after it was hashed")
    return h


def write_csv(path, rows):
    if not rows:
        return
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


def thr_at_spec(y, s, spec=SPEC_TARGET):
    neg = np.asarray(s, float)[np.asarray(y, int) == 0]
    if len(neg) == 0:
        return float("nan")
    return float(np.quantile(neg, spec))


def operating(y, s, thr):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pred = s >= thr
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return sens, spec, float(pred.mean())


def boot(y, s, thr, rng, n=N_BOOT):
    y, s = np.asarray(y, int), np.asarray(s, float)
    a, se, sp = [], [], []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        a.append(SEL.auc(y[k], s[k]))
        x, z, _ = operating(y[k], s[k], thr)
        se.append(x)
        sp.append(z)
    q = lambda v: (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))
    return q(a), q(se), q(sp)


def paired_boot_sens(y, s1, s2, t1, t2, rng, n=N_BOOT):
    y = np.asarray(y, int)
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if not (y[k] == 1).any():
            continue
        a, _, _ = operating(y[k], s2[k], t2)
        b, _, _ = operating(y[k], s1[k], t1)
        d.append(a - b)
    return round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)


def oracle_sens_at_spec(y, s, spec=SPEC_TARGET):
    t = thr_at_spec(y, s, spec)
    return operating(y, s, t)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    a = ap.parse_args()
    import torch
    import pandas as pd
    t_all = time.perf_counter()
    proto = check_protocol()

    # ------------------------------------------------------------- data
    with np.load(REG / "regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    with np.load(REG / "regional_nlm_lungs_g12_v1.npz") as z:
        nid, NX = z["image_id"].tolist(), z["reg"]
    ti = {i: k for k, i in enumerate(tid)}
    c01 = pd.read_csv(C01)
    c01 = c01[c01.image_id.isin(ti)]
    forbidden = set(pd.read_csv(C01).query("split == 'internal_test'").image_id)
    assert not (forbidden & set(tid)), "internal_test present in the feature cache"
    tr = c01[(c01.split == "train") & (c01.category.isin(TB + ("healthy",)))]
    dv = c01[c01.split == "development"]
    Xtr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, 12)
    Xdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, 12)
    ytr = np.array([1 if c in TB else 0 for c in tr.category], np.int64)
    ydv = np.array([1 if c in TB else 0 for c in dv.category], np.int64)
    assert (len(ytr), len(ydv)) == (3241, 659)

    site_of = np.array([i.split("__")[0] for i in nid])
    y_nlm = np.array([int(i.rsplit("_", 1)[1]) for i in nid])
    NXp = S.repool(NX, 12, 12)
    rng_split = np.random.default_rng(SEED)
    splits = {}
    for site in ("shz", "mcu"):
        idx = np.where(site_of == site)[0]
        adapt, evalu = [], []
        for lab in (0, 1):
            ii = idx[y_nlm[idx] == lab]
            ii = ii[rng_split.permutation(len(ii))]
            h = len(ii) // 2
            adapt.extend(ii[:h].tolist())
            evalu.extend(ii[h:].tolist())
        adapt, evalu = np.array(sorted(adapt)), np.array(sorted(evalu))
        assert not set(adapt) & set(evalu)
        splits[site] = (adapt, evalu)
        print("%-11s adapt %3d (TB %3d) | eval %3d (TB %3d)" % (
            SITE_NAME[site], len(adapt), y_nlm[adapt].sum(), len(evalu), y_nlm[evalu].sum()))
    split_rec = {site: {"adaptation_n": int(len(a_)), "adaptation_tb": int(y_nlm[a_].sum()),
                        "evaluation_n": int(len(e_)), "evaluation_tb": int(y_nlm[e_].sum()),
                        "split_seed": SEED, "stratified": True, "patient_disjoint": True,
                        "note": "one image per patient in NLM"}
                 for site, (a_, e_) in splits.items()}
    (TAB / "site_splits.json").write_text(json.dumps(split_rec, indent=1), encoding="utf-8")

    if not a.run:
        print("\nDRY. Pass --run to benchmark.")
        return 0

    # ------------------------------------------------------------- timing: GPU
    timing = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    t = time.perf_counter()
    emb = R.RegionalEmbedder()
    t_load_gpu = time.perf_counter() - t
    paths = sorted(NLM_IMG.glob("*.png"))
    assert len(paths) == 800
    from PIL import Image

    def encode(paths, embedder, batch=16):
        out, buf = [], []
        for p in paths:
            buf.append(np.array(Image.open(p).convert("L"), np.uint8))
            if len(buf) == batch:
                out.append(embedder.embed(buf))
                buf = []
        if buf:
            out.append(embedder.embed(buf))
        return np.concatenate(out)

    encode(paths[:16], emb)                                   # warm-up, not timed
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    E_gpu = encode(paths, emb)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_enc_gpu = time.perf_counter() - t
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None
    # reproduction check against the cached embeddings
    ni = {i: k for k, i in enumerate(nid)}
    got = E_gpu[[k for k, p in enumerate(paths)]]
    ref = NX[[ni[p.stem] for p in paths]]
    mad = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).mean())
    print("GPU encode 800 images: %.1f s (%.3f s/img) | MAD vs cache %.2e | peak %.2f GB"
          % (t_enc_gpu, t_enc_gpu / 800, mad, peak_gb or 0))
    if mad > 5e-3:
        raise Blocker("re-encoding does not reproduce the cached NLM features (MAD %.3g)" % mad)
    t = time.perf_counter()
    W0, b0, best0 = S.train(Xtr, ytr, Xdv, ydv, "max", HEAD_SEED)
    t_fit_gpu = time.perf_counter() - t
    timing.append({"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                   "encoder_load_s": round(t_load_gpu, 2),
                   "encode_n_images": 800, "encode_total_s": round(t_enc_gpu, 2),
                   "encode_s_per_image": round(t_enc_gpu / 800, 4),
                   "encode_images_per_s": round(800 / t_enc_gpu, 2),
                   "head_fit_n_images": int(len(ytr)), "head_fit_s": round(t_fit_gpu, 2),
                   "head_fit_dev_auroc": round(best0, 4),
                   "trainable_parameters": int(W0.size + b0.size),
                   "frozen_backbone_parameters": int(sum(p.numel() for p in emb.model.parameters())),
                   "peak_gpu_memory_gb": round(peak_gb, 3) if peak_gb else "",
                   "reproduces_cached_features_mad": mad,
                   "precision": "fp16" if emb.half else "fp32"})
    del emb
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- timing: CPU
    if not a.skip_cpu:
        real_avail = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            t = time.perf_counter()
            emb_c = R.RegionalEmbedder()
            t_load_cpu = time.perf_counter() - t
            encode(paths[:4], emb_c)
            t = time.perf_counter()
            encode(paths[:32], emb_c, batch=8)
            t_enc_cpu = time.perf_counter() - t
            del emb_c
            t = time.perf_counter()
            Wc, bc, bestc = S.train(Xtr, ytr, Xdv, ydv, "max", HEAD_SEED)
            t_fit_cpu = time.perf_counter() - t
            print("CPU encode 32 images: %.1f s (%.3f s/img) | head fit %.1f s"
                  % (t_enc_cpu, t_enc_cpu / 32, t_fit_cpu))
            timing.append({"device": "cpu (%d torch threads)" % torch.get_num_threads(),
                           "encoder_load_s": round(t_load_cpu, 2),
                           "encode_n_images": 32, "encode_total_s": round(t_enc_cpu, 2),
                           "encode_s_per_image": round(t_enc_cpu / 32, 4),
                           "encode_images_per_s": round(32 / t_enc_cpu, 3),
                           "head_fit_n_images": int(len(ytr)), "head_fit_s": round(t_fit_cpu, 2),
                           "head_fit_dev_auroc": round(bestc, 4),
                           "trainable_parameters": int(Wc.size + bc.size),
                           "frozen_backbone_parameters": timing[0]["frozen_backbone_parameters"],
                           "peak_gpu_memory_gb": "", "reproduces_cached_features_mad": "",
                           "precision": "fp32"})
        finally:
            torch.cuda.is_available = real_avail
    write_csv(TAB / "timing.csv", timing)

    # ------------------------------------------------------------- adaptation curve
    thr0 = thr_at_spec(ydv, S.score(Xdv, W0, b0, "max"))
    rows, paired = [], []
    rng = np.random.default_rng(SEED)
    enc_rate = timing[0]["encode_s_per_image"]
    for site, (adapt, evalu) in splits.items():
        Xe, ye = NXp[evalu], y_nlm[evalu]
        s0_e = S.score(Xe, W0, b0, "max")
        a0 = operating(ye, s0_e, thr0)
        ci_a, ci_se, ci_sp = boot(ye, s0_e, thr0, np.random.default_rng(SEED))
        rows.append({"site": SITE_NAME[site], "n_local": 0, "draw": "-", "arm": "A0",
                     "arm_desc": "frozen head, TBX11K threshold",
                     "eval_n": int(len(ye)), "eval_tb": int(ye.sum()),
                     "auroc": round(SEL.auc(ye, s0_e), 4), "auroc_ci": "[%s, %s]" % ci_a,
                     "sensitivity": round(a0[0], 4), "sens_ci": "[%s, %s]" % ci_se,
                     "specificity": round(a0[1], 4), "spec_ci": "[%s, %s]" % ci_sp,
                     "referral_rate": round(a0[2], 4),
                     "oracle_sens_at_0.70_spec_on_eval": round(oracle_sens_at_spec(ye, s0_e), 4),
                     "threshold": round(thr0, 4), "fit_s": "", "encode_s_est": "",
                     "total_adaptation_s": ""})
        for n in GRID[site]:
            if n == 0:
                continue
            nn = len(adapt) if n == "all" else int(n)
            for d_ in DRAWS:
                rd = np.random.default_rng(SEED * 10 + d_)
                loc = []
                for lab in (0, 1):
                    ii = adapt[y_nlm[adapt] == lab]
                    k = int(round(nn * (y_nlm[adapt] == lab).mean()))
                    loc.extend(rd.choice(ii, size=min(k, len(ii)), replace=False).tolist())
                loc = np.array(sorted(loc))
                Xl, yl = NXp[loc], y_nlm[loc]
                if len(np.unique(yl)) < 2:
                    continue
                # A1: frozen head, local threshold
                s0_l = S.score(Xl, W0, b0, "max")
                t1 = thr_at_spec(yl, s0_l)
                o1 = operating(ye, s0_e, t1)
                ci_a1, ci_se1, ci_sp1 = boot(ye, s0_e, t1, np.random.default_rng(SEED + d_))
                rows.append({"site": SITE_NAME[site], "n_local": nn, "draw": d_, "arm": "A1",
                             "arm_desc": "frozen head, threshold refit on local",
                             "eval_n": int(len(ye)), "eval_tb": int(ye.sum()),
                             "auroc": round(SEL.auc(ye, s0_e), 4), "auroc_ci": "[%s, %s]" % ci_a1,
                             "sensitivity": round(o1[0], 4), "sens_ci": "[%s, %s]" % ci_se1,
                             "specificity": round(o1[1], 4), "spec_ci": "[%s, %s]" % ci_sp1,
                             "referral_rate": round(o1[2], 4),
                             "oracle_sens_at_0.70_spec_on_eval": round(oracle_sens_at_spec(ye, s0_e), 4),
                             "threshold": round(t1, 4), "fit_s": 0.0,
                             "encode_s_est": round(enc_rate * len(loc), 2),
                             "total_adaptation_s": round(enc_rate * len(loc), 2)})
                # A2: pooled head refit, local threshold
                Xp = np.concatenate([Xtr, Xl])
                yp = np.concatenate([ytr, yl])
                t = time.perf_counter()
                W2, b2, _ = S.train(Xp, yp, Xdv, ydv, "max", HEAD_SEED)
                fit_s = time.perf_counter() - t
                s2_l = S.score(Xl, W2, b2, "max")
                s2_e = S.score(Xe, W2, b2, "max")
                t2 = thr_at_spec(yl, s2_l)
                o2 = operating(ye, s2_e, t2)
                ci_a2, ci_se2, ci_sp2 = boot(ye, s2_e, t2, np.random.default_rng(SEED + d_))
                rows.append({"site": SITE_NAME[site], "n_local": nn, "draw": d_, "arm": "A2",
                             "arm_desc": "head refit on TBX11K + local, threshold on local",
                             "eval_n": int(len(ye)), "eval_tb": int(ye.sum()),
                             "auroc": round(SEL.auc(ye, s2_e), 4), "auroc_ci": "[%s, %s]" % ci_a2,
                             "sensitivity": round(o2[0], 4), "sens_ci": "[%s, %s]" % ci_se2,
                             "specificity": round(o2[1], 4), "spec_ci": "[%s, %s]" % ci_sp2,
                             "referral_rate": round(o2[2], 4),
                             "oracle_sens_at_0.70_spec_on_eval": round(oracle_sens_at_spec(ye, s2_e), 4),
                             "threshold": round(t2, 4), "fit_s": round(fit_s, 2),
                             "encode_s_est": round(enc_rate * len(loc), 2),
                             "total_adaptation_s": round(fit_s + enc_rate * len(loc), 2)})
                lo, hi = paired_boot_sens(ye, s0_e, s2_e, t1, t2, np.random.default_rng(SEED + d_))
                both_up = (o2[0] >= o1[0]) and (o2[1] >= o1[1])
                paired.append({"site": SITE_NAME[site], "n_local": nn, "draw": d_,
                               "delta_sens_A2_minus_A1": round(o2[0] - o1[0], 4),
                               "delta_sens_ci95": "[%s, %s]" % (lo, hi),
                               "delta_spec_A2_minus_A1": round(o2[1] - o1[1], 4),
                               "delta_auroc_A2_minus_A1": round(SEL.auc(ye, s2_e) - SEL.auc(ye, s0_e), 4),
                               "sens_and_spec_both_not_worse": both_up,
                               "sens_ci_excludes_zero": bool(lo > 0 or hi < 0),
                               "protocol_verdict": ("head refit helps" if (both_up and lo > 0)
                                                    else ("harm" if (hi < 0 or o2[1] < o1[1] - 0.02)
                                                          else "no clear benefit"))})
                print("  %-10s n=%3d d=%d  A1 sens %.3f spec %.3f | A2 sens %.3f spec %.3f  d_sens %+.3f [%+.3f,%+.3f]  fit %.0fs"
                      % (SITE_NAME[site], nn, d_, o1[0], o1[1], o2[0], o2[1], o2[0] - o1[0], lo, hi, fit_s))
    write_csv(TAB / "adaptation_curve.csv", rows)
    write_csv(TAB / "head_refit_vs_threshold_only.csv", paired)

    # summary across draws
    summ = {}
    for r in rows:
        if r["arm"] == "A0":
            continue
        k = (r["site"], r["n_local"], r["arm"])
        summ.setdefault(k, []).append(r)
    srows = []
    for (site, n, arm), rs in sorted(summ.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        se = [r["sensitivity"] for r in rs]
        sp = [r["specificity"] for r in rs]
        au = [r["auroc"] for r in rs]
        srows.append({"site": site, "n_local": n, "arm": arm, "draws": len(rs),
                      "sens_mean": round(float(np.mean(se)), 4), "sens_min": min(se), "sens_max": max(se),
                      "spec_mean": round(float(np.mean(sp)), 4), "spec_min": min(sp), "spec_max": max(sp),
                      "auroc_mean": round(float(np.mean(au)), 4),
                      "total_adaptation_s_mean": round(float(np.mean([r["total_adaptation_s"] for r in rs])), 1)})
    write_csv(TAB / "adaptation_summary.csv", srows)

    import platform
    (TAB / "run_manifest.json").write_text(json.dumps({
        "run": RUN.name, "protocol_sha256": proto,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.perf_counter() - t_all) / 60, 2),
        "backbone": {"repo": "microsoft/rad-dino", "revision": "110cbc18d5133582e320b43d53bf5c44e410c936"},
        "head": {"trainable_parameters": int(W0.size + b0.size), "seed": HEAD_SEED,
                 "tbx11k_dev_auroc": round(best0, 4), "tbx11k_threshold_at_0.70_spec": round(thr0, 4)},
        "sites": split_rec, "grid": {k: list(v) for k, v in GRID.items()},
        "draws": list(DRAWS), "n_boot": N_BOOT, "seed": SEED,
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False,
                       "nlm_exposure": "both sites previously used for external validation; exploratory"},
        "training_of_backbone": False, "production_changed": False,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "cpu_threads": torch.get_num_threads()},
    }, indent=1), encoding="utf-8")
    print("\ndone in %.1f min" % ((time.perf_counter() - t_all) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
