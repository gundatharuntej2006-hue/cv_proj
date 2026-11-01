# -*- coding: utf-8 -*-
"""Clinical-context fusion by repeated stratified nested CV on Vietnam TUNE.

    py -3 fusion_cv.py
"""
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
PRED = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/predictions"
META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
SPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
PINNED_FP = (ROOT / "artifacts/kaggle_exec_prep_20261023_1130/configs/tune_cohort_fingerprint.json")
SEEDS = range(20)
FOLDS = 10
N_BOOT = 2000
BOOT_SEED = 20261101
MIN_EFFECT = 0.02
SPEC_T = 0.70
CLIN = ["sex", "age_group", "c", "f", "wl", "ns", "contact", "past_tb", "dm", "dm_unknown",
        "scan1", "scan2"]


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


def load():
    import pandas as pd
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    tune = {str(i) for i in sp["tune"]}
    report = {str(i) for i in sp["report"]}
    m = pd.read_csv(META)
    m["pid"] = m.pid.astype(str)
    m = m[m.pid.isin(tune)].sort_values("pid").reset_index(drop=True)
    if len(m) != 1352 or set(m.pid) & report:
        raise Blocker("cohort is not the 1,352 TUNE set")
    fp = hashlib.sha256("\n".join(sorted(m.pid)).encode()).hexdigest()
    pinned = json.loads(PINNED_FP.read_text(encoding="utf-8"))["fingerprint_sha256"]
    if fp != pinned:
        raise Blocker("cohort fingerprint mismatch")
    scores = {}
    for key, f in (("ours", "vietnam_tune_armA.csv"), ("ours_B1", "vietnam_tune_armB1.csv")):
        with open(PRED / f, encoding="utf-8") as fh:
            d = {r["pid"]: float(r["score_seed_mean"]) for r in csv.DictReader(fh)}
        scores[key] = np.array([d[p] for p in m.pid])
    scores["lunit"] = m.lunit_3901.to_numpy(float)
    y = m.xpert_result.to_numpy(int)
    X = np.column_stack([
        m.sex.to_numpy(float), (m.age_group.to_numpy(int) == 2).astype(float),
        m.c, m.f, m.wl, m.ns, m.contact, m.past_tb,
        (m.dm.to_numpy(int) == 1).astype(float), (m.dm.to_numpy(int) == 99).astype(float),
        (m.rad_system.to_numpy(int) == 1).astype(float),
        (m.rad_system.to_numpy(int) == 2).astype(float)]).astype(float)
    for col in range(X.shape[1]):
        if not np.isfinite(X[:, col]).all():
            raise Blocker("non-finite clinical feature %s" % CLIN[col])
    return m.pid.tolist(), y, X, scores, fp, m.past_tb.to_numpy(int)


def design(model, s, X):
    if model == "M1":
        return X
    if model == "M2":
        return np.column_stack([s, X])
    if model == "M3":
        return np.column_stack([s, X, s * X[:, CLIN.index("past_tb")]])
    raise ValueError(model)


def cv_oof(model, s, X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    pred_thr = np.zeros(len(y), bool)
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mu, sd = s[tr].mean(), s[tr].std() + 1e-12
        sz = (s - mu) / sd
        Dtr, Dte = design(model, sz[tr], X[tr]), design(model, sz[te], X[te])
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
        lr.fit(Dtr, y[tr])
        p_tr = lr.predict_proba(Dtr)[:, 1]
        thr = np.quantile(p_tr[y[tr] == 0], SPEC_T)
        oof[te] = lr.predict_proba(Dte)[:, 1]
        pred_thr[te] = oof[te] >= thr
    return oof, pred_thr


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def auprc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def main():
    t0 = time.perf_counter()
    proto = check_protocol()
    pids, y, X, scores, fp, past = load()
    print("TUNE %d, %d positive, %d previous TB, fingerprint %s (pinned)"
          % (len(y), y.sum(), past.sum(), fp[:16]))

    # M0 operating point: threshold at 0.70 spec within each fold's training part
    def m0_thr_preds(s, seed):
        from sklearn.model_selection import StratifiedKFold
        pred = np.zeros(len(y), bool)
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y):
            thr = np.quantile(s[tr][y[tr] == 0], SPEC_T)
            pred[te] = s[te] >= thr
        return pred

    results, oof_mean = [], {}
    for sk, s in scores.items():
        for model in ("M0", "M1", "M2", "M3"):
            if model == "M1" and sk != "ours":
                continue                       # clinical-only is score-independent
            aucs, aucs_pt, aucs_npt, sens, spec, aps = [], [], [], [], [], []
            oofs = []
            for seed in SEEDS:
                if model == "M0":
                    oof, pr = s.copy(), m0_thr_preds(s, seed)
                else:
                    oof, pr = cv_oof(model, s, X, y, seed)
                oofs.append(oof)
                aucs.append(auc(y, oof))
                aucs_pt.append(auc(y[past == 1], oof[past == 1]))
                aucs_npt.append(auc(y[past == 0], oof[past == 0]))
                sens.append(float(pr[y == 1].mean()))
                spec.append(float((~pr[y == 0]).mean()))
                aps.append(auprc(y, oof))
            om = np.mean(oofs, axis=0)
            oof_mean[(sk, model)] = om
            results.append({
                "image_score": sk, "model": model,
                "auroc_mean": round(float(np.mean(aucs)), 4), "auroc_sd": round(float(np.std(aucs)), 4),
                "auroc_prev_tb_n466_mean": round(float(np.mean(aucs_pt)), 4),
                "auroc_prev_tb_sd": round(float(np.std(aucs_pt)), 4),
                "auroc_no_prev_tb_n886_mean": round(float(np.mean(aucs_npt)), 4),
                "auprc_mean": round(float(np.mean(aps)), 4), "prevalence": round(float(y.mean()), 4),
                "sens_at_train_thr_0.70spec": round(float(np.mean(sens)), 4),
                "spec_at_train_thr_0.70spec": round(float(np.mean(spec)), 4),
                "n_repeats": len(SEEDS), "folds": FOLDS,
            })
            print("  %-8s %-3s AUROC %.4f ±%.4f | prevTB %.4f | noPrevTB %.4f | sens %.3f spec %.3f"
                  % (sk, model, np.mean(aucs), np.std(aucs), np.mean(aucs_pt), np.mean(aucs_npt),
                     np.mean(sens), np.mean(spec)))
    write_csv(TAB / "fusion_results.csv", results)

    # paired bootstrap on repeat-averaged OOF predictions
    rng = np.random.default_rng(BOOT_SEED)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]

    def paired(pa, pb, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        d = []
        for k in idxs:
            kk = k[m[k]]
            if len(np.unique(y[kk])) < 2:
                continue
            d.append(auc(y[kk], pb[kk]) - auc(y[kk], pa[kk]))
        point = auc(y[m], pb[m]) - auc(y[m], pa[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return round(point, 4), round(float(lo), 4), round(float(hi), 4)

    contrasts = []
    pt = past == 1
    for sk in ("ours", "ours_B1", "lunit"):
        for a_, b_, label in (("M0", "M2", "fusion vs image alone"),
                              ("M0", "M3", "fusion+interaction vs image alone"),
                              ("M2", "M3", "interaction vs fusion")):
            pa, pb = oof_mean[(sk, a_)], oof_mean[(sk, b_)]
            d, lo, hi = paired(pa, pb)
            dp, lop, hip = paired(pa, pb, pt)
            contrasts.append({"image_score": sk, "contrast": "%s - %s" % (b_, a_), "label": label,
                              "delta_auroc_all": d, "ci95_all": "[%s, %s]" % (lo, hi),
                              "excludes_zero_all": bool(lo > 0 or hi < 0),
                              "meets_min_effect_0.02": bool(d >= MIN_EFFECT and lo > 0),
                              "delta_auroc_prev_tb_n466": dp, "ci95_prev_tb": "[%s, %s]" % (lop, hip),
                              "excludes_zero_prev_tb": bool(lop > 0 or hip < 0)})
    # cross-system contrasts
    for a_, b_, label in ((("lunit", "M0"), ("ours", "M0"), "ours raw vs lunit raw"),
                          (("lunit", "M0"), ("ours", "M2"), "ours fused vs lunit raw"),
                          (("lunit", "M2"), ("ours", "M2"), "ours fused vs lunit fused"),
                          (("lunit", "M3"), ("ours", "M3"), "ours fused+int vs lunit fused+int")):
        pa, pb = oof_mean[a_], oof_mean[b_]
        d, lo, hi = paired(pa, pb)
        dp, lop, hip = paired(pa, pb, pt)
        contrasts.append({"image_score": "cross", "contrast": "%s/%s - %s/%s" % (b_[0], b_[1], a_[0], a_[1]),
                          "label": label, "delta_auroc_all": d, "ci95_all": "[%s, %s]" % (lo, hi),
                          "excludes_zero_all": bool(lo > 0 or hi < 0), "meets_min_effect_0.02": "",
                          "delta_auroc_prev_tb_n466": dp, "ci95_prev_tb": "[%s, %s]" % (lop, hip),
                          "excludes_zero_prev_tb": bool(lop > 0 or hip < 0)})
    write_csv(TAB / "paired_contrasts.csv", contrasts)

    # coefficients of the full-data M2/M3 fit, for interpretation only (not a model to ship)
    from sklearn.linear_model import LogisticRegression
    coefs = []
    for sk in ("ours", "lunit"):
        s = scores[sk]
        sz = (s - s.mean()) / (s.std() + 1e-12)
        for model in ("M2", "M3"):
            D = design(model, sz, X)
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(D, y)
            names = ["image_score"] + CLIN + (["image_score_x_past_tb"] if model == "M3" else [])
            for n_, c_ in zip(names, lr.coef_[0]):
                coefs.append({"image_score": sk, "model": model, "feature": n_,
                              "coef_log_odds": round(float(c_), 4)})
    write_csv(TAB / "coefficients_full_fit.csv", coefs)

    import platform
    (TAB / "run_manifest.json").write_text(json.dumps({
        "run": RUN.name, "protocol_sha256": proto, "cohort_fingerprint": fp,
        "n": int(len(y)), "n_positive": int(y.sum()), "n_prev_tb": int(past.sum()),
        "folds": FOLDS, "repeats": len(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "lr": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        "clinical_features": CLIN, "min_effect": MIN_EFFECT,
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "tune_exposure": "TUNE selected the pooling configuration; absolute AUROCs optimistic; fusion increment fitted inside CV",
        "adopted_into_production": False,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }, indent=1), encoding="utf-8")
    print("\nwrote 3 tables + run_manifest.json in %.1f min" % ((time.perf_counter() - t0) / 60))


if __name__ == "__main__":
    sys.exit(main())
