# -*- coding: utf-8 -*-
"""Workstreams B and C: reproduce the correlations, stratify them, decompose B0 vs B1.

Re-runs the pinned CLEAR inference over the approved Vietnam TUNE images only, because
the per-case similarities were deliberately deleted by the previous run. Encoding goes to
a private scratch directory outside this run directory and is deleted at the end.

Writes aggregate tables only. No per-case score, image, clinical row or patient id is
written here, and a test asserts it.

    py -3 b1_image_side.py                 # dry validation
    py -3 b1_image_side.py --run-inference
"""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
PKG = ROOT / "artifacts/kaggle_exec_prep_20261023_1130"
PRED = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/predictions"
PRIOR = ROOT / "artifacts/b1_clear_mechanism_20261031_2328"
SCRATCH = Path(os.environ.get("TB_SCRATCH",
                              r"D:\tb-triage-scratch\consolidation_forensic_20261101_0123"))
sys.path.insert(0, str(PKG / "src"))
import paths as P                                                   # noqa: E402
import concept_audit as A                                           # noqa: E402

SEED = 20261101
N_BOOT = 10000
# Reported in artifacts/b1_clear_mechanism_20261031_2328, to be reproduced here.
TARGETS = {("consolidation", "negatives (all)"): -0.3000,
           ("consolidation", "positives (all)"): -0.3849,
           ("scarring", "negatives (all)"): 0.5661,
           ("fibrosis", "negatives (all)"): 0.4166}
TOL = 0.002


class Blocker(SystemExit):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def cfg():
    return json.loads((RUN / "configs/concept_indices.json").read_text(encoding="utf-8"))


def arm_scores(name, pids):
    with open(PRED / ("vietnam_tune_arm%s.csv" % name), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        if r["pid"] in by:
            raise Blocker("duplicate pid in arm %s" % name)
        by[r["pid"]] = r
    if len(by) != len(pids) or any(i not in by for i in pids):
        raise Blocker("arm %s does not cover the audited cohort exactly" % name)
    s = np.array([float(by[i]["score_seed_mean"]) for i in pids])
    y = np.array([int(by[i]["xpert_result"]) for i in pids])
    if not np.isfinite(s).all():
        raise Blocker("non-finite score in arm %s" % name)
    return s, y


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def design(d, m):
    """The declared covariate baseline, restricted to a boolean mask."""
    return np.column_stack([d["rad_system"][m] == v for v in (1, 2, 3)][:2]
                           + [d["age_group"][m], d["sex"][m], d["symptoms"][m],
                              d["past_tb"][m]]).astype(float)


def r2(X, y):
    Xi = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    res = y - Xi @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((res ** 2).sum()) / sst if sst > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-inference", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    t0 = time.perf_counter()
    from scipy.stats import spearmanr, pearsonr

    conf = cfg()
    names = [c["concept"] for c in conf["audited"]] + \
            [c["concept"] for c in conf["diagnostic_probes"]]
    idxs = [c["concept_idx"] for c in conf["audited"]] + \
           [c["concept_idx"] for c in conf["diagnostic_probes"]]
    probes = {c["concept"] for c in conf["diagnostic_probes"]}
    controls = {c["concept"] for c in conf["audited"] if c["role"] == "negative control"}
    family = conf["bh_family"]
    if len(set(idxs)) != len(idxs):
        raise Blocker("duplicate concept index in the declared set")

    # Shadow the audited package's asset lookup with this run's map, without editing it.
    shadow = RUN / "configs/_shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    (shadow / "concept_indices.json").write_text(json.dumps(
        {"vocabulary_sha256_prefix": "93e975b722422b64f18b57161ac9894f",
         "hf_repo": A.HF_REPO, "pinned_revision": A.HF_REV,
         "rule": conf["rule"], "n_primary": len(names),
         "mapping": [{"concept": n, "exact_vocab_entry": n, "concept_idx": i,
                      "status": "EXACT - single vocabulary entry"}
                     for n, i in zip(names, idxs)]}, indent=1), encoding="utf-8")
    P._ASSET_DIRS = [shadow] + list(P._ASSET_DIRS)

    p = P.resolve(prefer="repo")
    if p["layout"] != "repo":
        raise Blocker("expected the local repo layout, got %r" % p["layout"])
    d = A.load_tune_only(p)
    pids = d["pid"]
    split = json.loads(Path(p["split"]).read_text(encoding="utf-8"))
    report = {str(i) for i in split["report"]}
    if set(pids) & report:
        raise Blocker("a Vietnam REPORT pid entered the cohort")

    s_b0, y0 = arm_scores("A", pids)
    s_b1, y1 = arm_scores("B1", pids)
    if not (np.array_equal(y0, d["y"]) and np.array_equal(y1, d["y"])):
        raise Blocker("arm labels disagree with the audited cohort")

    print("cohort      : %d patients, %d positive, %d previous TB" %
          (len(pids), int(d["y"].sum()), int(d["past_tb"].sum())))
    print("fingerprint : %s (matches pinned)" % d["fingerprint"][:16])
    print("concepts    : %d audited + %d diagnostic probes = %d encoded"
          % (len(conf["audited"]), len(conf["diagnostic_probes"]), len(names)))
    print("REPORT pids : 0 | internal_test referenced: no")
    missing, extra = A.check_images(p, pids)
    print("images      : %d approved, %d missing, %d present but unapproved"
          % (len(pids) - len(missing), len(missing), len(extra)))
    if missing:
        raise Blocker("approved images missing: %d" % len(missing))

    # B0 identity check, cited as evidence rather than asserted
    with open(p["scores"], encoding="utf-8") as fh:
        prod = {r["pid"]: float(r["score_baseline"]) for r in csv.DictReader(fh)
                if r["pid"] in set(pids)}
    pv = np.array([prod[i] for i in pids])
    b0_ident = {"spearman": round(float(spearmanr(pv, s_b0).statistic), 6),
                "pearson": round(float(np.corrcoef(pv, s_b0)[0, 1]), 6),
                "max_abs_difference": float(np.abs(pv - s_b0).max()),
                "source_production": str(Path(p["scores"]).relative_to(ROOT)),
                "source_B0": "active_tb_label_raydino_20261024_1254/.../vietnam_tune_armA.csv"}
    print("B0 vs production score_baseline: spearman %.6f, max abs diff %.2g"
          % (b0_ident["spearman"], b0_ident["max_abs_difference"]))

    if not a.run_inference:
        print("\nDRY VALIDATION ONLY. Prerequisites pass. Pass --run-inference.")
        return 0

    SCRATCH.mkdir(parents=True, exist_ok=True)
    S, prov = A.encode_concepts(pids, device=a.device, batch=a.batch,
                                out_dir=SCRATCH, resume=True)
    if S.shape != (len(pids), len(names)):
        raise Blocker("similarities %s, expected (%d, %d)" % (S.shape, len(pids), len(names)))
    if not np.isfinite(S).all():
        raise Blocker("non-finite similarity")

    arms = {"B0": s_b0, "B1": s_b1}
    y, past, scan = d["y"], d["past_tb"], d["rad_system"]
    neg, pos = y == 0, y == 1

    # ---------- B2: reproduce, with Pearson as a sensitivity analysis -----------------
    groups = [("negatives (all)", neg), ("positives (all)", pos),
              ("negatives, past TB", neg & (past == 1)),
              ("negatives, no past TB", neg & (past == 0)),
              ("positives, past TB", pos & (past == 1)),
              ("positives, no past TB", pos & (past == 0))]
    for sc in sorted(set(scan.tolist())):
        groups.append(("negatives, scanner %d" % sc, neg & (scan == sc)))
        groups.append(("positives, scanner %d" % sc, pos & (scan == sc)))

    rng = np.random.default_rng(SEED)
    assoc, pv_family = [], {}
    for arm, s in arms.items():
        for j, nm in enumerate(names):
            for label, m in groups:
                n = int(m.sum())
                if n < 20:
                    assoc.append({"arm": arm, "concept": nm, "group": label, "n": n,
                                  "note": "suppressed, n < 20"})
                    continue
                c, ss = S[m, j], s[m]
                r = spearmanr(c, ss)
                pr = pearsonr(c, ss)
                boot = []
                for _ in range(N_BOOT if label.endswith("(all)") else 2000):
                    k = rng.integers(0, n, n)
                    if np.ptp(c[k]) == 0 or np.ptp(ss[k]) == 0:
                        continue
                    boot.append(spearmanr(c[k], ss[k]).statistic)
                X = design(d, m)
                base = r2(X, ss)
                row = {"arm": arm, "concept": nm, "group": label, "n": n,
                       "is_diagnostic_probe": nm in probes,
                       "is_negative_control": nm in controls,
                       "spearman_rho": round(float(r.statistic), 4),
                       "rho_ci95_lo": round(float(np.percentile(boot, 2.5)), 4),
                       "rho_ci95_hi": round(float(np.percentile(boot, 97.5)), 4),
                       "p_raw": float(r.pvalue),
                       "pearson_r_sensitivity": round(float(pr.statistic), 4),
                       "r2_covariates_only": round(base, 4),
                       "incr_r2": round(r2(np.column_stack([X, c]), ss) - base, 4),
                       "note": ""}
                assoc.append(row)
                if label == "negatives (all)" and nm in family:
                    pv_family.setdefault(arm, []).append((len(assoc) - 1, float(r.pvalue)))

    for arm, items in pv_family.items():
        pvals = np.array([p for _, p in items])
        o = np.argsort(pvals)
        adj = np.minimum.accumulate(
            (pvals[o] * len(pvals) / (np.arange(len(pvals)) + 1))[::-1])[::-1]
        padj = np.empty(len(pvals))
        padj[o] = np.clip(adj, 0, 1)
        for (i, _), v in zip(items, padj):
            assoc[i]["p_bh_adjusted"] = float(v)
    for r in assoc:
        r.setdefault("p_bh_adjusted", "")
    write_csv(TAB / "b0_b1_concept_associations.csv", assoc)

    # ---------- reproduction check against the previous run ---------------------------
    got = {(r["concept"], r["group"]): r["spearman_rho"] for r in assoc
           if r["arm"] == "B0" and "spearman_rho" in r}
    repro = []
    ok = True
    for (nm, grp), want in TARGETS.items():
        have = got.get((nm, grp))
        passed = have is not None and abs(have - want) <= TOL
        ok = ok and passed
        repro.append({"concept": nm, "group": grp, "previously_reported": want,
                      "reproduced_here": have,
                      "abs_difference": None if have is None else round(abs(have - want), 5),
                      "tolerance": TOL, "pass": passed})
        print("  repro %-14s %-20s reported %+.4f  here %s  %s"
              % (nm, grp, want, ("%+.4f" % have) if have is not None else "n/a",
                 "PASS" if passed else "FAIL"))

    # ---------- C2: ranking versus scale ------------------------------------------------
    from sklearn.metrics import roc_auc_score, average_precision_score
    rank_rows = [{
        "quantity": "B0 vs B1 Spearman (rank agreement, all 1352)",
        "value": round(float(spearmanr(s_b0, s_b1).statistic), 4)},
        {"quantity": "B0 vs B1 Pearson (scale agreement)",
         "value": round(float(np.corrcoef(s_b0, s_b1)[0, 1]), 4)},
        {"quantity": "B0 AUROC", "value": round(float(roc_auc_score(y, s_b0)), 4)},
        {"quantity": "B1 AUROC", "value": round(float(roc_auc_score(y, s_b1)), 4)},
        {"quantity": "B0 AUPRC", "value": round(float(average_precision_score(y, s_b0)), 4)},
        {"quantity": "B1 AUPRC", "value": round(float(average_precision_score(y, s_b1)), 4)},
        {"quantity": "mean rank shift (|delta rank| / n)",
         "value": round(float(np.abs(np.argsort(np.argsort(s_b0))
                                     - np.argsort(np.argsort(s_b1))).mean() / len(y)), 4)},
        {"quantity": "max rank shift (|delta rank| / n)",
         "value": round(float(np.abs(np.argsort(np.argsort(s_b0))
                                     - np.argsort(np.argsort(s_b1))).max() / len(y)), 4)},
    ]
    # monotone recalibration of B1 onto B0's marginal distribution: rank-preserving by
    # construction, so any concept association that survives it is a ranking effect.
    order = np.argsort(np.argsort(s_b1))
    s_b1_cal = np.sort(s_b0)[order]
    rank_rows.append({"quantity": "B1 after monotone recalibration onto B0 scale: Spearman vs B1",
                      "value": round(float(spearmanr(s_b1, s_b1_cal).statistic), 4)})
    for nm, j in (("consolidation", names.index("consolidation")),
                  ("scarring", names.index("scarring"))):
        rank_rows.append({
            "quantity": "%s rho among negatives, B1 recalibrated to B0 scale" % nm,
            "value": round(float(spearmanr(S[neg, j], s_b1_cal[neg]).statistic), 4)})
    write_csv(TAB / "b0_b1_ranking_vs_scale.csv", rank_rows)

    # ---------- B3: paired, shared-index differences -----------------------------------
    rng2 = np.random.default_rng(SEED)
    nneg = int(neg.sum())
    idxs_b = [rng2.integers(0, nneg, nneg) for _ in range(N_BOOT)]
    Sneg, a0, a1 = S[neg], s_b0[neg], s_b1[neg]
    Xn = design(d, neg)
    paired = []
    for j, nm in enumerate(names):
        c = Sneg[:, j]
        r0 = float(spearmanr(c, a0).statistic)
        r1 = float(spearmanr(c, a1).statistic)
        dd = []
        for k in idxs_b:
            ck = c[k]
            if np.ptp(ck) == 0:
                continue
            dd.append(spearmanr(ck, a1[k]).statistic - spearmanr(ck, a0[k]).statistic)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        b0r, b1r = r2(Xn, a0), r2(Xn, a1)
        paired.append({"concept": nm, "n_negatives": nneg,
                       "is_diagnostic_probe": nm in probes,
                       "is_negative_control": nm in controls,
                       "rho_B0": round(r0, 4), "rho_B1": round(r1, 4),
                       "delta_rho": round(r1 - r0, 4),
                       "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
                       "excludes_zero": bool(lo > 0 or hi < 0),
                       "incr_r2_B0": round(r2(np.column_stack([Xn, c]), a0) - b0r, 4),
                       "incr_r2_B1": round(r2(np.column_stack([Xn, c]), a1) - b1r, 4),
                       "n_replicates": len(dd)})
    write_csv(TAB / "b0_b1_paired_differences.csv", paired)

    # ---------- B4/B5: consolidation stratified, with influence diagnostics ------------
    jc = names.index("consolidation")
    strat = []
    for arm, s in arms.items():
        for label, m in groups:
            n = int(m.sum())
            if n < 20:
                strat.append({"arm": arm, "group": label, "n": n, "note": "suppressed, n < 20"})
                continue
            c, ss = S[m, jc], s[m]
            r = float(spearmanr(c, ss).statistic)
            # leave-one-out influence on the rank correlation
            loo = [float(spearmanr(np.delete(c, i), np.delete(ss, i)).statistic)
                   for i in range(n)] if n <= 1300 else []
            X = design(d, m)
            base = r2(X, ss)
            strat.append({"arm": arm, "group": label, "n": n,
                          "spearman_rho": round(r, 4),
                          "pearson_r": round(float(pearsonr(c, ss).statistic), 4),
                          "incr_r2": round(r2(np.column_stack([X, c]), ss) - base, 4),
                          "loo_min_rho": round(min(loo), 4) if loo else "",
                          "loo_max_rho": round(max(loo), 4) if loo else "",
                          "sign_stable_under_loo": (bool(max(loo) < 0) if loo and r < 0
                                                    else (bool(min(loo) > 0) if loo else "")),
                          "note": ""})
    write_csv(TAB / "consolidation_stratified_results.csv", strat)

    # ---------- the probe comparison, the point of this run -----------------------------
    probe_rows = []
    for arm, s in arms.items():
        for nm in ("consolidation", "no consolidation", "scarring", "no scarring",
                   "clear lungs", "lungs are clear", "no acute cardiopulmonary process",
                   "normal chest"):
            j = names.index(nm)
            for label, m in (("negatives (all)", neg), ("positives (all)", pos)):
                c, ss = S[m, j], s[m]
                probe_rows.append({
                    "arm": arm, "concept": nm, "group": label, "n": int(m.sum()),
                    "role": "diagnostic probe" if nm in probes else "audited concept",
                    "spearman_rho": round(float(spearmanr(c, ss).statistic), 4)})
    # image-side correlation between each probe and its audited partner
    for a_, b_ in (("consolidation", "no consolidation"), ("scarring", "no scarring"),
                   ("consolidation", "clear lungs"), ("consolidation", "normal chest")):
        probe_rows.append({
            "arm": "-", "concept": "%s  vs  %s" % (a_, b_), "group": "image-side, all 1352",
            "n": len(pids), "role": "concept-to-concept",
            "spearman_rho": round(float(spearmanr(S[:, names.index(a_)],
                                                  S[:, names.index(b_)]).statistic), 4)})
    write_csv(TAB / "negation_probe_results.csv", probe_rows)

    # ---------- provenance ----------------------------------------------------------------
    import platform
    import torch
    man = {
        "run": RUN.name,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "execution": "local GPU; no Kaggle session; nothing uploaded",
        "cohort": {"name": "vietnam_tune", "n": len(pids),
                   "n_positive": int(y.sum()), "n_negative": int(neg.sum()),
                   "n_past_tb": int(past.sum()),
                   "fingerprint_sha256": d["fingerprint"],
                   "ordered_id_hash_sha256": d["ordered_hash"],
                   "images_missing": len(missing), "images_unapproved_present": len(extra)},
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False,
                       "report_pids_in_cohort": 0},
        "model": {"repo": A.HF_REPO, "revision": A.HF_REV, "clear_commit": A.CLEAR_COMMIT,
                  "input_px": A.CLEAR_INPUT_PX, **prov},
        "concepts": {"audited": len(conf["audited"]),
                     "diagnostic_probes": len(conf["diagnostic_probes"]),
                     "encoded": len(names), "bh_family": family,
                     "bh_family_is_not_independent": True},
        "statistics": {"seed": SEED, "n_boot_primary": N_BOOT, "n_boot_subgroup": 2000,
                       "interval": "percentile",
                       "resampling_unit": "patient (one image per patient in Vietnam)"},
        "b0_identity_with_production": b0_ident,
        "reproduction": {"targets": {"%s|%s" % k: v for k, v in TARGETS.items()},
                         "all_passed": ok},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "numpy": np.__version__},
        "per_case_export": "none; shards computed in private scratch and deleted",
        "training_performed": False,
        "production_unchanged": "Vietnam screening AUROC remains 0.7242; nothing was modified",
    }
    (TAB / "metric_reproduction.json").write_text(json.dumps(man, indent=1), encoding="utf-8")

    shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.rmtree(shadow, ignore_errors=True)
    print("\nscratch deleted: %s" % SCRATCH)
    print("wrote 6 aggregate tables in %.1f min" % ((time.perf_counter() - t0) / 60))
    if not ok:
        print("\nREPRODUCTION MISMATCH - see tables/metric_reproduction.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
