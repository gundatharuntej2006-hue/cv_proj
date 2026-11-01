# -*- coding: utf-8 -*-
"""Clinical-field ablation of the R22 fusion, exactly as PROTOCOL.md.

    py -3 ablation.py --run
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
R22 = ROOT / "artifacts/fusion_on_stacked_head_20261101_0800"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(R16 / "scripts"))
sys.path.insert(0, str(R20 / "scripts"))
import fusion_cv as F16                                              # noqa: E402
import ensemble as E20                                               # noqa: E402

CLIN = F16.CLIN
GROUPS = {"dm": ["dm", "dm_unknown"], "scanner": ["scan1", "scan2"]}
SINGLES = [c for c in CLIN if c not in ("dm", "dm_unknown", "scan1", "scan2")]
REPRO = {"X raw": 0.7585, "full": 0.7941, "comparator full": 0.7778}


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


def arms():
    out = {"full": list(CLIN)}
    for c in SINGLES:
        out["without %s" % c] = [k for k in CLIN if k != c]
    for g, cols in GROUPS.items():
        out["without %s" % g] = [k for k in CLIN if k not in cols]
    out["image + past_tb"] = ["past_tb"]
    out["image + scanner"] = ["scan1", "scan2"]
    out["image + symptoms"] = ["c", "f", "wl", "ns"]
    out["image + demographics"] = ["sex", "age_group"]
    out["questionnaire only"] = [k for k in CLIN if k not in ("scan1", "scan2")]
    return out


def cv_oof_cols(s, Xc, y, seed):
    """R16's cv_oof with model M2 on an arbitrary column subset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    oof, pred = np.zeros(len(y)), np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(F16.FOLDS, shuffle=True, random_state=seed).split(Xc, y):
        mu, sd = s[tr].mean(), s[tr].std() + 1e-12
        sz = (s - mu) / sd
        Dtr, Dte = np.column_stack([sz[tr], Xc[tr]]), np.column_stack([sz[te], Xc[te]])
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000).fit(Dtr, y[tr])
        thr = np.quantile(lr.predict_proba(Dtr)[:, 1][y[tr] == 0], F16.SPEC_T)
        oof[te] = lr.predict_proba(Dte)[:, 1]
        pred[te] = oof[te] >= thr
    return oof, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    proto = check_protocol()
    pids, y, X, scores, fp, past = F16.load()
    print("TUNE %d, %d positive, %d previous TB, fingerprint %s (pinned)" % (len(y), y.sum(), past.sum(), fp[:16]))
    if not a.run:
        print("DRY. Pass --run.")
        return 0

    with np.load(R22 / "tables/X_noNLM_head.npz") as z:
        Ws = list(zip(z["W"], z["b"]))
    Xtune = E20.load_cache(R20 / "cache/regional_D2_vie_tune_lungs_g12.npz", pids)
    s_x = np.mean([E20.image_logits(Xtune, W, b) for W, b in Ws], 0)
    gate = [{"check": "X raw", "expected": REPRO["X raw"], "here": round(F16.auc(y, s_x), 4)}]

    A = arms()
    results, oof = [], {}
    jobs = [("ours_X", k, cols, s_x) for k, cols in A.items()] + [("lunit", "comparator full", list(CLIN), scores["lunit"])]
    for sk, name, cols, s in jobs:
        Xc = X[:, [CLIN.index(c) for c in cols]]
        oofs, prs = [], []
        for seed in F16.SEEDS:
            o, p = cv_oof_cols(s, Xc, y, seed)
            oofs.append(o)
            prs.append(p)
        om = np.mean(oofs, 0)
        oof[name] = om
        results.append({"arm": name, "image_score": sk, "n_fields": len(cols), "fields": " ".join(cols),
                        "auroc_mean": round(float(np.mean([F16.auc(y, o) for o in oofs])), 4),
                        "auroc_sd": round(float(np.std([F16.auc(y, o) for o in oofs])), 4),
                        "auroc_prev_tb_n466": round(float(np.mean([F16.auc(y[past == 1], o[past == 1]) for o in oofs])), 4),
                        "auroc_no_prev_tb_n886": round(float(np.mean([F16.auc(y[past == 0], o[past == 0]) for o in oofs])), 4),
                        "auprc_mean": round(float(np.mean([F16.auprc(y, o) for o in oofs])), 4),
                        "sens_at_train_thr_0.70spec": round(float(np.mean([p[y == 1].mean() for p in prs])), 4),
                        "spec_at_train_thr_0.70spec": round(float(np.mean([(~p[y == 0]).mean() for p in prs])), 4)})
        r = results[-1]
        print("  %-24s %2d fields AUROC %.4f ±%.4f | prevTB %.4f | AUPRC %.3f | sens %.3f" % (
            name, len(cols), r["auroc_mean"], r["auroc_sd"], r["auroc_prev_tb_n466"], r["auprc_mean"], r["sens_at_train_thr_0.70spec"]))
    F16.write_csv(TAB / "ablation_results.csv", results)
    r = {x["arm"]: x["auroc_mean"] for x in results}
    gate += [{"check": "full", "expected": REPRO["full"], "here": r["full"]},
             {"check": "comparator full", "expected": REPRO["comparator full"], "here": r["comparator full"]}]
    for g in gate:
        g["pass"] = abs(g["here"] - g["expected"]) <= 1e-3
        print("gate %-16s expected %.4f here %.4f %s" % (g["check"], g["expected"], g["here"], "pass" if g["pass"] else "FAIL"))
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
    for name in A:
        if name == "full":
            continue
        d, lo, hi = paired(oof["full"], oof[name])
        dp, lop, hip = paired(oof["full"], oof[name], past == 1)
        carries = name.startswith("without") and d <= -0.01 and hi < 0
        rows.append({"contrast": "%s - full" % name, "delta": d, "ci": "[%+.4f, %+.4f]" % (lo, hi),
                     "prev_tb_delta": dp, "prev_tb_ci": "[%+.4f, %+.4f]" % (lop, hip),
                     "field_carries_gain": carries if name.startswith("without") else ""})
        print("%-28s %+.4f [%+.4f, %+.4f] | prevTB %+.4f [%+.4f, %+.4f]%s" % (
            rows[-1]["contrast"], d, lo, hi, dp, lop, hip, "  <- carries" if carries else ""))
    F16.write_csv(TAB / "paired_vs_full.csv", rows)

    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "cohort_fingerprint": fp, "n": int(len(y)), "positives": int(y.sum()),
        "previous_tb": int(past.sum()), "arms": {k: v for k, v in A.items()},
        "cv": {"folds": F16.FOLDS, "repeats": len(F16.SEEDS)}, "n_boot": F16.N_BOOT, "boot_seed": F16.BOOT_SEED,
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=1), encoding="utf-8")
    print("\ndone in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
