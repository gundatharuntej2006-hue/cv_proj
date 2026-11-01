# -*- coding: utf-8 -*-
"""Clinical-context fusion on the MAIRA-2 + active-target head, exactly as PROTOCOL.md.

    py -3 fusion_x.py --run
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
R16 = ROOT / "artifacts/clinical_fusion_20261101_0330"
R20 = ROOT / "artifacts/multiencoder_ensemble_20261101_0600"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(R16 / "scripts"))
sys.path.insert(0, str(R20 / "scripts"))
import fusion_cv as F16                                              # noqa: E402  load, cv_oof, auc, auprc, write_csv
import ensemble as E20                                               # noqa: E402  train, image_logits, load_cache
from p2_cls import selection as SEL                                  # noqa: E402

C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
SEEDS = E20.SEEDS
REPRO = {"X_tune_raw": 0.7585, "X_cal": 0.9466, "ours_M2": 0.7809, "lunit_M2": 0.7778}
MIN_EFFECT = 0.02


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


def fit_x_head(pids, log):
    """X_noNLM: MAIRA-2 regional features, active target, TBX11K train only; R20 trainer."""
    import pandas as pd
    c01 = pd.read_csv(C01)
    tr = c01[(c01.split == "train") & c01.category.isin(E20.TB + ("healthy",))]
    dv = c01[c01.split == "development"]
    cal = c01[c01.split == "calibration"]
    d2 = lambda n, ids: E20.load_cache(R20 / "cache" / ("regional_D2_%s_lungs_g12.npz" % n), ids)
    Xtr, Xdv, Xcal, Xtune = d2("tbx_train", tr.image_id), d2("tbx_dev", dv.image_id), d2("tbx_cal", cal.image_id), d2("vie_tune", pids)
    ytr = np.array([1 if c in E20.ACTIVE else 0 for c in tr.category])
    ydv = np.array([1 if c in E20.ACTIVE else 0 for c in dv.category])
    y_cal = np.array([1 if c in E20.ACTIVE else 0 for c in cal.category])
    Ws = []
    for sd in SEEDS:
        W, b, best = E20.train(Xtr, ytr, Xdv, ydv, sd)
        Ws.append((W, b))
        log("   X_noNLM seed %2d dev %.4f" % (sd, best))
    tune = np.mean([E20.image_logits(Xtune, W, b) for W, b in Ws], 0)
    cal_auc = SEL.auc(y_cal, np.mean([E20.image_logits(Xcal, W, b) for W, b in Ws], 0))
    np.savez(TAB / "X_noNLM_head.npz", W=np.stack([w for w, _ in Ws]), b=np.stack([b_ for _, b_ in Ws]))
    return tune, cal_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    proto = check_protocol()
    logf = open(RUN / "run.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    pids, y, X, scores, fp, past = F16.load()
    log("TUNE %d, %d positive, %d previous TB, fingerprint %s (pinned)" % (len(y), y.sum(), past.sum(), fp[:16]))
    if not a.run:
        log("DRY. Pass --run.")
        return 0

    tune_x, cal_x = fit_x_head(pids, log)
    scores = {"ours": scores["ours"], "ours_X": tune_x, "lunit": scores["lunit"]}
    gate = [{"check": "X_noNLM raw TUNE AUROC", "expected": REPRO["X_tune_raw"], "here": round(F16.auc(y, tune_x), 4)},
            {"check": "X_noNLM calibration AUROC", "expected": REPRO["X_cal"], "here": round(cal_x, 4)}]

    results, oof = [], {}
    for sk, s in scores.items():
        for model in ("M0", "M2"):
            oofs, pr_all, aucs = [], [], []
            for seed in F16.SEEDS:
                if model == "M0":
                    o = s.copy()
                    pr = np.zeros(len(y), bool)
                    from sklearn.model_selection import StratifiedKFold
                    for tr, te in StratifiedKFold(F16.FOLDS, shuffle=True, random_state=seed).split(X, y):
                        pr[te] = s[te] >= np.quantile(s[tr][y[tr] == 0], F16.SPEC_T)
                else:
                    o, pr = F16.cv_oof(model, s, X, y, seed)
                oofs.append(o)
                pr_all.append(pr)
                aucs.append(F16.auc(y, o))
            om = np.mean(oofs, 0)
            oof[(sk, model)] = om
            results.append({"image_score": sk, "model": model,
                            "auroc_mean": round(float(np.mean(aucs)), 4), "auroc_sd": round(float(np.std(aucs)), 4),
                            "auroc_prev_tb_n466": round(float(np.mean([F16.auc(y[past == 1], o[past == 1]) for o in oofs])), 4),
                            "auroc_no_prev_tb_n886": round(float(np.mean([F16.auc(y[past == 0], o[past == 0]) for o in oofs])), 4),
                            "auprc_mean": round(float(np.mean([F16.auprc(y, o) for o in oofs])), 4),
                            "sens_at_train_thr_0.70spec": round(float(np.mean([p[y == 1].mean() for p in pr_all])), 4),
                            "spec_at_train_thr_0.70spec": round(float(np.mean([(~p[y == 0]).mean() for p in pr_all])), 4),
                            "n_repeats": len(F16.SEEDS), "folds": F16.FOLDS})
            log("  %-7s %-2s AUROC %.4f ±%.4f | prevTB %.4f | noPrevTB %.4f | sens %.3f spec %.3f" % (
                sk, model, results[-1]["auroc_mean"], results[-1]["auroc_sd"], results[-1]["auroc_prev_tb_n466"],
                results[-1]["auroc_no_prev_tb_n886"], results[-1]["sens_at_train_thr_0.70spec"], results[-1]["spec_at_train_thr_0.70spec"]))
    F16.write_csv(TAB / "fusion_results.csv", results)
    r = {(x["image_score"], x["model"]): x["auroc_mean"] for x in results}
    gate += [{"check": "ours M2 (R16)", "expected": REPRO["ours_M2"], "here": r[("ours", "M2")]},
             {"check": "lunit M2 (R16)", "expected": REPRO["lunit_M2"], "here": r[("lunit", "M2")]}]
    for g in gate:
        g["pass"] = abs(g["here"] - g["expected"]) <= 1e-3
        log("gate %-28s expected %.4f here %.4f %s" % (g["check"], g["expected"], g["here"], "pass" if g["pass"] else "FAIL"))
    F16.write_csv(TAB / "reproduction_gate.csv", gate)
    if not all(g["pass"] for g in gate):
        raise Blocker("reproduction gate failed")

    rng = np.random.default_rng(F16.BOOT_SEED)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(F16.N_BOOT)]

    def paired(pa, pb, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        d = []
        for k in idxs:
            kk = k[m[k]]
            if len(np.unique(y[kk])) < 2:
                continue
            d.append(F16.auc(y[kk], pb[kk]) - F16.auc(y[kk], pa[kk]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return round(F16.auc(y[m], pb[m]) - F16.auc(y[m], pa[m]), 4), round(float(lo), 4), round(float(hi), 4)

    rows = []
    for name, a_, b_, primary in (("ours_X M2 - ours M2", ("ours", "M2"), ("ours_X", "M2"), True),
                                  ("ours_X M2 - lunit M0", ("lunit", "M0"), ("ours_X", "M2"), False),
                                  ("ours_X M2 - lunit M2", ("lunit", "M2"), ("ours_X", "M2"), False),
                                  ("ours_X M0 - ours M0", ("ours", "M0"), ("ours_X", "M0"), False),
                                  ("ours_X M2 - ours_X M0", ("ours_X", "M0"), ("ours_X", "M2"), False),
                                  ("ours_X M0 - lunit M0", ("lunit", "M0"), ("ours_X", "M0"), False)):
        d, lo, hi = paired(oof[a_], oof[b_])
        dp, lop, hip = paired(oof[a_], oof[b_], past == 1)
        v = ""
        if primary:
            v = "INCONCLUSIVE" if (hi - lo) / 2 > 0.05 else ("ACCEPT" if (d >= MIN_EFFECT and lo > 0) else "REJECT")
        rows.append({"contrast": name, "primary": primary, "delta": d, "ci": "[%+.4f, %+.4f]" % (lo, hi),
                     "prev_tb_delta": dp, "prev_tb_ci": "[%+.4f, %+.4f]" % (lop, hip), "verdict": v})
        log("%-24s %+.4f [%+.4f, %+.4f] | prevTB %+.4f [%+.4f, %+.4f] %s" % (name, d, lo, hi, dp, lop, hip, v))
    F16.write_csv(TAB / "paired_contrasts.csv", rows)

    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "cohort_fingerprint": fp, "n": int(len(y)), "positives": int(y.sum()),
        "previous_tb": int(past.sum()), "encoder_X": {"model": E20.ENCODERS["D2"][0], "revision": E20.ENCODERS["D2"][1]},
        "seeds_head": SEEDS, "cv": {"folds": F16.FOLDS, "repeats": len(F16.SEEDS)},
        "n_boot": F16.N_BOOT, "boot_seed": F16.BOOT_SEED,
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=1), encoding="utf-8")
    log("\ndone in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
