# -*- coding: utf-8 -*-
"""Score the frozen focal-pooling configuration once, on everything it has not seen.

Written **before** the sweep results were looked at, so the reporting protocol is fixed in
advance rather than chosen once the winner is known. It reads the selected configuration
from `mil_sweep.json` and does not re-select anything.

What it reports:
  * the Vietnam **REPORT** half - 1,356 participants, 141 TB, never used for any decision;
  * the Vietnam TUNE half alongside it, so the selection optimism is visible rather than
    hidden (if REPORT is much worse than TUNE, the sweep overfitted and we say so);
  * TBX11K development, for the internal-versus-external contrast;
  * the commercial comparator on exactly the REPORT patients.

Curated external sites (Shenzhen, Montgomery, DA/DB) need regional embeddings of their own
and are scored by `--sites` once those exist; the screening cohort is the number that
matters and is reported regardless.

    py -3 scripts/p2_33_mil_final.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p2_cls import selection as SEL                                 # noqa: E402

REG = ROOT / "artifacts/p2/regional"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
SPLIT_JSON = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
SWEEP = REG / "mil_sweep.json"
OUT = REG / "mil_final.json"
TB = ("active_tb", "latent_tb", "active_latent_tb")
SEEDS = (17, 29, 43, 61, 79)          # five seeds for the final number, as everywhere else
N_BOOT = 4000


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


def main():
    import pandas as pd
    import p2_32_mil_sweep as S

    sweep = json.load(open(SWEEP, encoding="utf-8"))
    best = sweep["best"]
    g, how = int(best["grid"]), best["agg"]
    print("frozen configuration from the sweep: grid %d x %d, aggregator %r" % (g, g, how))
    print("  selected on the TUNE half at %.4f; that figure is optimistic by construction\n"
          % best["tune_auc"])

    with np.load(REG / "regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    with np.load(REG / "regional_vietnam_lungs_g12_v1.npz") as z:
        vid, VX = z["image_id"].tolist(), z["reg"]

    ti = {i: k for k, i in enumerate(tid)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(ti)]
    tr = c01[(c01.split == "train") & (c01.category.isin(TB + ("healthy",)))]
    dv = c01[c01.split == "development"]
    Ttr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, g)
    Tdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, g)
    ytr = np.array([1 if c in TB else 0 for c in tr.category], np.int64)
    ydv = np.array([1 if c in TB else 0 for c in dv.category], np.int64)

    Vall = S.repool(VX, 12, g)
    Ws, devs = [], []
    for sd in SEEDS:
        W, b, d = S.train(Ttr, ytr, Tdv, ydv, how, sd)
        Ws.append((W, b))
        devs.append(d)
    p_dev = np.mean([S.score(Tdv, W, b, how) for W, b in Ws], axis=0)
    p_all = np.mean([S.score(Vall, W, b, how) for W, b in Ws], axis=0)

    split = json.load(open(SPLIT_JSON, encoding="utf-8"))
    tune, rep = set(split["tune"]), set(split["report"])
    csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    pid = [int(i) for i in vid]

    rows = {}
    for name, keep in (("Vietnam TUNE", tune), ("Vietnam REPORT", rep)):
        sel = [k for k, p in enumerate(pid) if p in keep]
        y = csv.loc[[pid[k] for k in sel], "xpert_result"].to_numpy()
        a, lo, hi = boot(y, p_all[sel])
        lun, _, _ = boot(y, csv.loc[[pid[k] for k in sel], "lunit_3901"].to_numpy())
        rows[name] = {"n": len(sel), "n_tb": int(y.sum()), "auc": round(a, 4),
                      "ci95": [round(lo, 4), round(hi, 4)], "lunit": round(lun, 4)}

    a, lo, hi = boot(ydv, p_dev)
    rows["TBX11K development"] = {"n": len(ydv), "n_tb": int(ydv.sum()), "auc": round(a, 4),
                                  "ci95": [round(lo, 4), round(hi, 4)], "lunit": None}

    print("  %-20s %5s %5s %9s %-22s %s" % ("", "n", "TB", "AUC", "95% CI", "Lunit"))
    for k in ("TBX11K development", "Vietnam TUNE", "Vietnam REPORT"):
        r = rows[k]
        print("  %-20s %5d %5d %9.4f [%.4f, %.4f] %s"
              % (k, r["n"], r["n_tb"], r["auc"], r["ci95"][0], r["ci95"][1],
                 ("%.4f" % r["lunit"]) if r["lunit"] else ""))

    opt = rows["Vietnam TUNE"]["auc"] - rows["Vietnam REPORT"]["auc"]
    print("\nselection optimism (TUNE minus REPORT): %+.4f" % opt)
    if opt > 0.03:
        print("  -> the sweep overfitted the tuning half; REPORT is the honest figure.")
    else:
        print("  -> small, so the configuration generalises within the cohort.")

    r = rows["Vietnam REPORT"]
    print("\nHEADLINE, read once: %.4f [%.4f, %.4f] on %d held-aside screening participants"
          % (r["auc"], r["ci95"][0], r["ci95"][1], r["n"]))
    print("  commercial comparator on the same patients: %.4f" % r["lunit"])
    print("  previous best global model on the full cohort: 0.6744")
    print("  gap to commercial: %+.4f" % (r["auc"] - r["lunit"]))

    json.dump({"created": time.strftime("%Y-%m-%d"), "config": {"grid": g, "agg": how},
               "seeds": list(SEEDS), "selection_optimism": round(opt, 4),
               "results": rows,
               "note": ("REPORT half read once. Configuration selected on TUNE only, and "
                        "this script was written before the sweep results were seen.")},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
