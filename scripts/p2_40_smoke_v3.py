# -*- coding: utf-8 -*-
"""Smoke test for candidate v3: does the runtime mask -> MAIRA-2 embed -> head path reproduce
the R20 cached regional score on real images? 8 TBX11K development + 8 Vietnam TUNE (fusion).

    py -3 scripts/p2_40_smoke_v3.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_cls import pipeline_v2 as P2                                 # noqa: E402

BUNDLE = ROOT / "artifacts/p2/bundle_v3_candidate"
R20 = ROOT / "artifacts/multiencoder_ensemble_20261101_0600/cache"


def cached(path, iid, bundle):
    with np.load(path) as z:
        X = z["reg"][z["image_id"].tolist().index(iid)].astype(np.float32)
    return P2.head(X, bundle)[0]


def main():
    import pandas as pd
    b = P2.load_bundle_v2(BUNDLE)
    assert b["meta"]["bundle_version"] == 3
    rows = []
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    dev = c01[c01.split == "development"]
    pick = list(dev[dev.category == "active_tb"].image_id[:4]) + list(dev[dev.category == "healthy"].image_id[:4])
    for iid in pick:
        t = time.perf_counter()
        r = P2.run_case_v2(ROOT / "data/processed/v1/tbx11k/img512" / (iid + ".png"), b)
        dt = (time.perf_counter() - t) * 1000
        rows.append(("tbx11k", iid, dev[dev.image_id == iid].category.iloc[0], r["status"],
                     r["analysis"]["image_logit"] if r["analysis"] else None,
                     cached(R20 / "regional_D2_tbx_dev_lungs_g12.npz", iid, b), dt, ""))
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    tune = [str(i) for i in sp["tune"]][:8]
    m = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    for pid in tune:
        row = m.loc[int(pid)]
        clin = {k: int(row[k]) for k in ("sex", "age_group", "c", "f", "wl", "ns", "contact", "past_tb", "dm", "rad_system")}
        t = time.perf_counter()
        r = P2.run_case_v2(ROOT / "data/processed/v1/vietnam_img512" / (pid + ".png"), b, clinical=clin)
        dt = (time.perf_counter() - t) * 1000
        fz = r["analysis"]["fusion"] if r["analysis"] else None
        rows.append(("vietnam", "TUNE-%d" % tune.index(pid), "xpert=%d" % int(row.xpert_result), r["status"],
                     r["analysis"]["image_logit"] if r["analysis"] else None,
                     cached(R20 / "regional_D2_vie_tune_lungs_g12.npz", pid, b), dt, (fz or {}).get("p_fused", fz)))
    print("%-8s %-10s %-12s %-9s %9s %9s %7s %s" % ("set", "image", "label", "status", "runtime", "cached", "ms", "fusion"))
    diffs = []
    for s, i, c, st, rt, off, dt, extra in rows:
        d = abs(rt - off) if rt is not None else float("nan")
        diffs.append(d)
        print("%-8s %-10s %-12s %-9s %9s %9.4f %7.0f %s" % (s, i, c, st, ("%.4f" % rt) if rt is not None else "-", off, dt, extra))
    diffs = np.array([d for d in diffs if np.isfinite(d)])
    print("\nruntime vs cached: max |diff| %.4f, median %.4f over %d analysed" % (diffs.max(), np.median(diffs), len(diffs)))
    print("VERDICT:", "runtime path reproduces the cache" if diffs.max() < 0.05 else "MISMATCH - investigate")
    return 0 if diffs.max() < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
