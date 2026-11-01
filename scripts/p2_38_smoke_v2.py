# -*- coding: utf-8 -*-
"""Smoke test for the v2 pipeline: does the runtime mask -> embed -> head path reproduce
the offline cached regional score on real images?

Runs 8 TBX11K development images and 8 Vietnam TUNE images through run_case_v2 and compares
the image logit against the score recomputed from the cached regional features with the same
bundle head. Agreement proves the runtime segmenter + embedder match the training-time cache.

    py -3 scripts/p2_38_smoke_v2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p2_cls import pipeline_v2 as P2                                 # noqa: E402
import p2_32_mil_sweep as S                                          # noqa: E402


def cached_score(X144, bundle):
    X = S.repool(X144[None], 12, 12)[0].astype(np.float32)
    return P2.head(X, bundle)[0]


def main():
    import pandas as pd
    b = P2.load_bundle_v2()
    rows = []
    # TBX11K development: raw img512 PNGs (unmasked) -> v2 masks them at runtime
    with np.load(ROOT / "artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    ti = {i: k for k, i in enumerate(tid)}
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    dev = c01[(c01.split == "development") & c01.image_id.isin(ti)]
    pick = list(dev[dev.category == "active_tb"].image_id[:4]) + list(dev[dev.category == "healthy"].image_id[:4])
    for iid in pick:
        p = ROOT / "data/processed/v1/tbx11k/img512" / (iid + ".png")
        t = time.perf_counter()
        r = P2.run_case_v2(p, b)
        dt = (time.perf_counter() - t) * 1000
        off = cached_score(TX[ti[iid]], b)
        rows.append(("tbx11k", iid, dev[dev.image_id == iid].category.iloc[0], r["status"],
                     r["analysis"]["image_logit"] if r["analysis"] else None, off, dt,
                     r["analysis"]["lung_mask"]["lung_fraction"] if r["analysis"] else None))
    # Vietnam TUNE: raw img512 -> runtime mask; compare with cached vietnam regional
    with np.load(ROOT / "artifacts/p2/regional/regional_vietnam_lungs_g12_v1.npz") as z:
        vid, VX = z["image_id"].tolist(), z["reg"]
    vi = {i: k for k, i in enumerate(vid)}
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    tune = [str(i) for i in sp["tune"]][:8]
    m = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    for pid in tune:
        p = ROOT / "data/processed/v1/vietnam_img512" / (pid + ".png")
        row = m.loc[int(pid)]
        clin = {k: int(row[k]) for k in ("sex", "age_group", "c", "f", "wl", "ns", "contact", "past_tb", "dm", "rad_system")}
        t = time.perf_counter()
        r = P2.run_case_v2(p, b, clinical=clin)
        dt = (time.perf_counter() - t) * 1000
        off = cached_score(VX[vi[pid]], b)
        fz = r["analysis"]["fusion"] if r["analysis"] else None
        rows.append(("vietnam", "TUNE-%d" % tune.index(pid), "xpert=%d" % int(row.xpert_result), r["status"],
                     r["analysis"]["image_logit"] if r["analysis"] else None, off, dt,
                     (fz or {}).get("p_fused", fz)))
    print("%-8s %-10s %-12s %-9s %9s %9s %7s %s" % ("set", "image", "label", "status", "runtime", "cached", "ms", "mask/fusion"))
    diffs = []
    for s, i, c, st, rt, off, dt, extra in rows:
        d = abs(rt - off) if rt is not None else float("nan")
        diffs.append(d)
        print("%-8s %-10s %-12s %-9s %9s %9.4f %7.0f %s" % (s, i, c, st, ("%.4f" % rt) if rt is not None else "-", off, dt, extra))
    diffs = np.array([d for d in diffs if np.isfinite(d)])
    print("\nruntime vs cached: max |diff| %.4f, median %.4f over %d analysed" % (diffs.max(), np.median(diffs), len(diffs)))
    print("VERDICT:", "runtime path reproduces the cache" if diffs.max() < 0.05 else "MISMATCH - investigate")


if __name__ == "__main__":
    main()
