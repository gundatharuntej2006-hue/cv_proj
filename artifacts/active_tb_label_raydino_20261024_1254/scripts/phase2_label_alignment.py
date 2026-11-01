# -*- coding: utf-8 -*-
"""Phase 2: active-TB label-alignment experiment, exactly as preregistered.

Three arms over one frozen fitting path, differing only in the TRAINING target.
Everything is evaluated against `active_tb_label` on TBX11K calibration.

Refuses to run if PROTOCOL.md has changed since it was hashed.
Refuses to load TBX11K internal_test or Vietnam REPORT under any circumstance.

    py -3 phase2_label_alignment.py
"""
import hashlib
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

from p2_cls import selection as SEL                                     # noqa: E402
import p2_32_mil_sweep as S                                             # noqa: E402

REG = ROOT / "artifacts/p2/regional"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
VSPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VCSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
CALEMB = RUN / "outputs/regional_tbx11k_calibration_lungs_g12_v1.npz"

SEEDS = (17, 29, 43, 61, 79)
GRID = 12
AGG = "max"
N_BOOT = 4000
BOOT_SEED = 0
MIN_EFFECT = 0.05
SPEC_TARGET = 0.90
SENS_DROP_LIMIT = 0.02

FORBIDDEN_SPLITS = ("internal_test",)
TB = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE = ("active_tb", "active_latent_tb")

ARMS = {
    # name: (training positives, extra training negatives beyond healthy)
    "A":  {"pos": TB,     "neg_extra": (),             "desc": "baseline, tb_label"},
    "B1": {"pos": ACTIVE, "neg_extra": ("latent_tb",), "desc": "active_tb_label, latent kept as negative"},
    "B2": {"pos": ACTIVE, "neg_extra": (),             "desc": "active_tb_label, latent excluded"},
}


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    pinned = (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]
    if h != pinned:
        raise SystemExit("PROTOCOL.md changed after it was hashed (%s != %s)" % (h, pinned))
    return h


def auc(y, p):
    return float(SEL.auc(np.asarray(y, int), np.asarray(p, float)))


def auprc(y, p):
    y = np.asarray(y, int)
    o = np.argsort(-np.asarray(p, float))
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(int(y.sum()), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def paired_boot(y, pa, pb, seed=BOOT_SEED, n=N_BOOT):
    """Identical resample indices for both arms. Returns (delta, lo, hi, ci_halfwidth)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], pb[k]) - SEL.auc(y[k], pa[k]))
    d = np.asarray(d)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return float(auc(y, pb) - auc(y, pa)), lo, hi, (hi - lo) / 2.0


def boot_auc(y, p, seed=BOOT_SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    v = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        v.append(SEL.auc(y[k], p[k]))
    return auc(y, p), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def threshold_at_spec(y, p, spec=SPEC_TARGET):
    """Smallest score reaching at least `spec` specificity on this split."""
    neg = np.asarray(p, float)[np.asarray(y, int) == 0]
    return float(np.quantile(neg, spec))


def main():
    import pandas as pd
    proto_hash = check_protocol()
    t0 = time.perf_counter()

    c01 = pd.read_csv(C01)
    forbidden_ids = set(c01[c01.split.isin(FORBIDDEN_SPLITS)].image_id)
    assert len(forbidden_ids) == 901, "unexpected internal_test size: %d" % len(forbidden_ids)

    # train + development come from the frozen cache; calibration from this run's own
    # extension of it (scripts/embed_calibration.py, verified bit-identical on 24 cached
    # development images before it was written).
    with np.load(REG / ("regional_tbx11k_lungs_g%d_v1.npz" % GRID)) as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    with np.load(CALEMB) as z:
        cid, CX = z["image_id"].tolist(), z["reg"]
    assert not (forbidden_ids & (set(tid) | set(cid))), \
        "internal_test images are present in an embedding file"
    ti = {i: k for k, i in enumerate(tid)}
    ci = {i: k for k, i in enumerate(cid)}

    tr_all = c01[(c01.split == "train") & c01.image_id.isin(ti)]
    dv = c01[(c01.split == "development") & c01.image_id.isin(ti)]
    cal = c01[(c01.split == "calibration") & c01.image_id.isin(ci)]
    assert (len(tr_all), len(dv), len(cal)) == (5941, 659, 899), \
        "split sizes changed: %d / %d / %d" % (len(tr_all), len(dv), len(cal))
    used_ids = set(tr_all.image_id) | set(dv.image_id) | set(cal.image_id)
    assert not (used_ids & forbidden_ids), "internal_test rows leaked into the run"

    Xdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, GRID)
    Xcal = S.repool(CX[[ci[i] for i in cal.image_id]], 12, GRID)
    cat_dv = dv.category.tolist()
    cat_cal = cal.category.tolist()

    def ylab(cats, target):
        pos = TB if target == "tb_label" else ACTIVE
        return np.array([1 if c in pos else 0 for c in cats], np.int64)

    y_cal_act = ylab(cat_cal, "active_tb_label")
    y_cal_tb = ylab(cat_cal, "tb_label")
    y_dv_act = ylab(cat_dv, "active_tb_label")

    # --- Vietnam TUNE, exploratory only -------------------------------------------------
    with np.load(REG / ("regional_vietnam_lungs_g%d_v1.npz" % GRID)) as z:
        vid, VX = z["image_id"].tolist(), z["reg"]
    vsplit = json.load(open(VSPLIT, encoding="utf-8"))
    tune, report = set(vsplit["tune"]), set(vsplit["report"])
    vsel = [k for k, i in enumerate(vid) if int(i) in tune]
    assert not ({int(vid[k]) for k in vsel} & report), "Vietnam REPORT leaked into the run"
    vdf = pd.read_csv(VCSV).set_index("pid")
    y_tune = vdf.loc[[int(vid[k]) for k in vsel], "xpert_result"].to_numpy().astype(int)
    Xtune = S.repool(VX[vsel], 12, GRID)
    print("Vietnam TUNE: %d participants, %d Xpert-positive (REPORT untouched)"
          % (len(y_tune), y_tune.sum()))

    # --- fit the arms --------------------------------------------------------------------
    out = {}
    per_seed = []
    for name, spec in ARMS.items():
        keep = tuple(spec["pos"]) + ("healthy",) + tuple(spec["neg_extra"])
        tr = tr_all[tr_all.category.isin(keep)]
        Xtr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, GRID)
        ytr = np.array([1 if c in spec["pos"] else 0 for c in tr.category], np.int64)
        # early stopping target = this arm's own target, as in production
        ydv = np.array([1 if c in spec["pos"] else 0 for c in cat_dv], np.int64)
        print("\narm %-2s  %-44s  n_train %d  pos %d" % (name, spec["desc"], len(ytr), ytr.sum()))
        Ws = []
        for sd in SEEDS:
            W, b, best = S.train(Xtr, ytr, Xdv, ydv, AGG, sd)
            Ws.append((W, b))
            per_seed.append({"arm": name, "seed": sd, "early_stop_dev_auc": round(best, 4)})
            print("   seed %3d  dev AUC (own target, early-stopping) %.4f" % (sd, best))
        p_cal = np.mean([S.score(Xcal, W, b, AGG) for W, b in Ws], axis=0)
        p_dv = np.mean([S.score(Xdv, W, b, AGG) for W, b in Ws], axis=0)
        p_tune = np.mean([S.score(Xtune, W, b, AGG) for W, b in Ws], axis=0)
        # per-case predictions, preserved as required by the protocol
        pred_dir = RUN / "outputs" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        import csv as _p
        with open(pred_dir / ("calibration_arm%s.csv" % name), "w", newline="",
                  encoding="utf-8") as fh:
            w = _p.writer(fh)
            w.writerow(["image_id", "category", "active_tb_label", "tb_label",
                        "score_seed_mean"] + ["score_seed_%d" % s for s in SEEDS])
            seedwise = [S.score(Xcal, W, b, AGG) for W, b in Ws]
            for k, iid in enumerate(cal.image_id):
                w.writerow([iid, cat_cal[k], int(y_cal_act[k]), int(y_cal_tb[k]),
                            round(float(p_cal[k]), 6)]
                           + [round(float(v[k]), 6) for v in seedwise])
        with open(pred_dir / ("vietnam_tune_arm%s.csv" % name), "w", newline="",
                  encoding="utf-8") as fh:
            w = _p.writer(fh)
            w.writerow(["pid", "xpert_result", "score_seed_mean"])
            for k, s in enumerate(vsel):
                w.writerow([int(vid[s]), int(y_tune[k]), round(float(p_tune[k]), 6)])
        # per-seed calibration AUROC against the fixed evaluation target
        for sd, (W, b) in zip(SEEDS, Ws):
            ps = S.score(Xcal, W, b, AGG)
            for r in per_seed:
                if r["arm"] == name and r["seed"] == sd:
                    r["calibration_auroc_vs_active_tb_label"] = round(auc(y_cal_act, ps), 4)
        out[name] = {"desc": spec["desc"], "n_train": int(len(ytr)),
                     "n_train_pos": int(ytr.sum()),
                     "p_cal": p_cal, "p_dv": p_dv, "p_tune": p_tune}

    # --- endpoints -------------------------------------------------------------------------
    dup = set()
    with open(TAB / "near_duplicate_cross_split.csv", encoding="utf-8") as fh:
        import csv as _csv
        for r in _csv.DictReader(fh):
            if r["c01_split_a"] == "calibration":
                dup.add(r["image_id_a"])
            if r["c01_split_b"] == "calibration":
                dup.add(r["image_id_b"])
    keep_mask = np.array([i not in dup for i in cal.image_id], bool)

    metrics, paired = [], []
    for name, d in out.items():
        p = d["p_cal"]
        a, lo, hi = boot_auc(y_cal_act, p)
        thr = threshold_at_spec(y_dv_act, d["p_dv"])
        pred = p >= thr
        sens = float(pred[y_cal_act == 1].mean())
        spec_ = float((~pred[y_cal_act == 0]).mean())
        a_tb, lo_tb, hi_tb = boot_auc(y_cal_tb, p)
        at, lot, hit = boot_auc(y_tune, d["p_tune"])
        metrics.append({
            "arm": name, "desc": d["desc"], "n_train": d["n_train"],
            "n_train_pos": d["n_train_pos"],
            "cal_n": int(len(y_cal_act)), "cal_pos_active": int(y_cal_act.sum()),
            "cal_prevalence_active": round(float(y_cal_act.mean()), 4),
            "cal_auroc_vs_active": round(a, 4), "cal_auroc_ci_lo": round(lo, 4),
            "cal_auroc_ci_hi": round(hi, 4),
            "cal_auprc_vs_active": round(auprc(y_cal_act, p), 4),
            "dev_threshold_at_spec0.90": round(thr, 4),
            "cal_sensitivity_active": round(sens, 4),
            "cal_specificity": round(spec_, 4),
            "cal_auroc_vs_tb_label_MIRROR": round(a_tb, 4),
            "cal_auroc_vs_tb_label_ci": "[%.4f, %.4f]" % (lo_tb, hi_tb),
            "cal_prevalence_tb_label": round(float(y_cal_tb.mean()), 4),
            "vietnam_tune_auroc_EXPLORATORY": round(at, 4),
            "vietnam_tune_ci": "[%.4f, %.4f]" % (lot, hit),
        })

    for comp in ("B1", "B2"):
        pa, pb = out["A"]["p_cal"], out[comp]["p_cal"]
        dlt, lo, hi, hw = paired_boot(y_cal_act, pa, pb)
        dlt_d, lo_d, hi_d, _ = paired_boot(y_cal_act[keep_mask], pa[keep_mask], pb[keep_mask])
        dlt_m, lo_m, hi_m, _ = paired_boot(y_cal_tb, pa, pb)
        dlt_v, lo_v, hi_v, _ = paired_boot(y_tune, out["A"]["p_tune"], out[comp]["p_tune"])
        thr_a = threshold_at_spec(y_dv_act, out["A"]["p_dv"])
        thr_b = threshold_at_spec(y_dv_act, out[comp]["p_dv"])
        sa = float((pa >= thr_a)[y_cal_act == 1].mean())
        sb = float((pb >= thr_b)[y_cal_act == 1].mean())
        paired.append({
            "comparison": "%s minus A" % comp,
            "endpoint": "calibration AUROC vs active_tb_label (PRIMARY)" if comp == "B1"
                        else "calibration AUROC vs active_tb_label (sensitivity arm)",
            "delta": round(dlt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ci_halfwidth": round(hw, 4),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "meets_min_effect_0.05": bool(dlt >= MIN_EFFECT),
            "halfwidth_within_0.05": bool(hw <= MIN_EFFECT),
            "delta_no_near_duplicates_n%d" % int(keep_mask.sum()): round(dlt_d, 4),
            "ci_no_near_duplicates": "[%.4f, %.4f]" % (lo_d, hi_d),
            "delta_MIRROR_vs_tb_label": round(dlt_m, 4),
            "ci_MIRROR": "[%.4f, %.4f]" % (lo_m, hi_m),
            "delta_vietnam_tune_EXPLORATORY": round(dlt_v, 4),
            "ci_vietnam_tune": "[%.4f, %.4f]" % (lo_v, hi_v),
            "active_tb_sensitivity_A": round(sa, 4),
            "active_tb_sensitivity_arm": round(sb, 4),
            "sensitivity_delta": round(sb - sa, 4),
            "sensitivity_guardrail_ok": bool((sb - sa) >= -SENS_DROP_LIMIT),
        })

    # --- score distribution by source class -------------------------------------------
    dist = []
    for name, d in out.items():
        for c in ("healthy", "sick_non_tb", "latent_tb", "active_tb", "active_latent_tb"):
            m = np.array([x == c for x in cat_cal], bool)
            if not m.any():
                continue
            dist.append({"arm": name, "source_class": c, "n": int(m.sum()),
                         "mean_score": round(float(d["p_cal"][m].mean()), 4),
                         "median_score": round(float(np.median(d["p_cal"][m])), 4),
                         "p90_score": round(float(np.percentile(d["p_cal"][m], 90)), 4)})

    def wcsv(path, rows):
        import csv as _c
        with open(TAB / path, "w", newline="", encoding="utf-8") as fh:
            w = _c.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # --- POST-HOC, declared post-hoc: active vs latent separation on calibration -------
    # Scale-invariant, unlike comparing raw score means across arms. n_latent = 18, so
    # this is a direction, not an estimate. Not in the protocol; descriptive only.
    posthoc = []
    m_act = np.array([c in ACTIVE for c in cat_cal], bool)
    m_lat = np.array([c == "latent_tb" for c in cat_cal], bool)
    m_sick = np.array([c == "sick_non_tb" for c in cat_cal], bool)
    for name, d in out.items():
        for lbl, mneg in (("active vs latent_tb", m_lat), ("active vs sick_non_tb", m_sick)):
            sel = m_act | mneg
            posthoc.append({"arm": name, "contrast": lbl,
                            "n_pos": int(m_act.sum()), "n_neg": int(mneg.sum()),
                            "auroc": round(auc(m_act[sel].astype(int), d["p_cal"][sel]), 4),
                            "status": "POST-HOC, not preregistered"})

    wcsv("posthoc_class_separation.csv", posthoc)
    wcsv("metrics_by_arm.csv", metrics)
    wcsv("paired_differences.csv", paired)
    wcsv("per_seed_results.csv", per_seed)
    wcsv("score_distribution_by_class.csv", dist)

    p = paired[0]
    verdict = ("ACCEPT" if (p["meets_min_effect_0.05"] and p["excludes_zero"]
                            and p["halfwidth_within_0.05"] and p["sensitivity_guardrail_ok"])
               else ("INCONCLUSIVE" if not p["halfwidth_within_0.05"] else "REJECT"))

    man = {
        "run": "active_tb_label_raydino_20261024_1254",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "protocol_sha256": proto_hash,
        "git_head": (ROOT / ".git/HEAD").exists() and
                    (TAB / "git_head_before.txt").read_text(encoding="utf-8").strip(),
        "config": {"grid": GRID, "agg": AGG, "seeds": list(SEEDS), "n_boot": N_BOOT,
                   "boot_seed": BOOT_SEED, "min_effect": MIN_EFFECT,
                   "spec_target": SPEC_TARGET},
        "inputs": {
            "embeddings_tbx11k": str((REG / ("regional_tbx11k_lungs_g%d_v1.npz" % GRID))
                                     .relative_to(ROOT)),
            "embeddings_vietnam": str((REG / ("regional_vietnam_lungs_g%d_v1.npz" % GRID))
                                      .relative_to(ROOT)),
            "splits": str(C01.relative_to(ROOT)),
            "embeddings_calibration": str(CALEMB.relative_to(ROOT)),
        },
        "partitions_accessed": ["tbx11k train", "tbx11k development", "tbx11k calibration",
                                "vietnam TUNE (exploratory)"],
        "internal_test_loaded": False,
        "vietnam_report_loaded": False,
        "primary_endpoint": p,
        "verdict": verdict,
        "note": ("EXPLORATORY. The label change was already measured once as arm L "
                 "(+0.0292 on Vietnam). This is not independent confirmation."),
    }
    (TAB / "run_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")

    print("\n%-3s %-44s %8s %8s %8s %8s" % ("arm", "desc", "calAUROC", "AUPRC", "sens", "spec"))
    for m in metrics:
        print("%-3s %-44s %8.4f %8.4f %8.4f %8.4f"
              % (m["arm"], m["desc"], m["cal_auroc_vs_active"], m["cal_auprc_vs_active"],
                 m["cal_sensitivity_active"], m["cal_specificity"]))
    print("\nPRIMARY  B1 - A  on calibration vs active_tb_label:")
    print("  delta %+.4f  95%% CI [%+.4f, %+.4f]  half-width %.4f"
          % (p["delta"], p["ci_lo"], p["ci_hi"], p["ci_halfwidth"]))
    print("  >= +0.05: %s | excludes zero: %s | half-width <= 0.05: %s | sensitivity ok: %s"
          % (p["meets_min_effect_0.05"], p["excludes_zero"],
             p["halfwidth_within_0.05"], p["sensitivity_guardrail_ok"]))
    print("  VERDICT: %s" % verdict)
    print("\nwrote tables/ (4 csv + run_manifest.json) in %.1f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
