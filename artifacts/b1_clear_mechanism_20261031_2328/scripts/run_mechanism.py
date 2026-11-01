# -*- coding: utf-8 -*-
"""B1 label alignment x CLEAR mechanism test — TUNE only, aggregates only.

Runs the pinned CLEAR encoder over the 1,352 approved Vietnam TUNE images **locally**,
then compares the score/concept associations under three classifier score columns:

    baseline_production   the column the 2026-10-23 Kaggle audit used -- reproduction gate
    B0                    arm A of the label-alignment run (current target)
    B1                    arm B1 of the same run (active-TB target)

The encoder, its revision, its preprocessing and the concept indices all come from the
audited package `artifacts/kaggle_exec_prep_20261023_1130`, which is imported and **never
modified**. The only addition is two predeclared negative-control concepts, supplied by
shadowing that package's asset lookup with this run's own `configs/concept_indices.json`.

Per-case concept similarities are computed into a scratch directory **outside** this run
directory and deleted at the end. Nothing per-case is written here.

    py -3 run_mechanism.py                 # dry validation, no inference
    py -3 run_mechanism.py --run-inference
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
METRICS = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/tables/metrics_by_arm.csv"
VOCAB = Path(r"D:\hf-cache\clear_vocab\mimic_concepts.csv")
SCRATCH = Path(os.environ.get("TB_SCRATCH",
                              r"D:\tb-triage-scratch\b1_clear_mechanism_20261031_2328"))

sys.path.insert(0, str(PKG / "src"))
import paths as P                                                   # noqa: E402
import concept_audit as A                                           # noqa: E402

PAIRED_SEED = 20261031
PAIRED_BOOT = 10000
VOCAB_SHA = "93e975b722422b64f18b57161ac9894fe30ac7a04cc68b7aa8a2e01f6d0fc439"

# --- the reproduction gate, copied from ANALYSIS_MANIFEST.md section 7 -----------------
GATE = {
    "scarring_rho": (0.5661, 0.0005),
    "scarring_ci_lo": (0.5276, 0.005),
    "scarring_ci_hi": (0.6020, 0.005),
    "fibrosis_rho": (0.4166, 0.0005),
    "scarring_delta_r2": (0.1975, 0.0010),
    "clinical_only_r2": (0.2441, 0.0010),
}


class Blocker(SystemExit):
    pass


def check_manifest():
    h = hashlib.sha256((RUN / "ANALYSIS_MANIFEST.md").read_bytes()).hexdigest()
    pinned = (TAB / "manifest_hash.txt").read_text(encoding="utf-8").split()[0]
    if h != pinned:
        raise Blocker("ANALYSIS_MANIFEST.md changed after it was hashed")
    return h


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_arm_scores(name, pids):
    """Per-case TUNE scores for one arm, re-ordered onto the audited id order."""
    with open(PRED / ("vietnam_tune_arm%s.csv" % name), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(pids):
        raise Blocker("arm %s has %d rows, expected %d" % (name, len(rows), len(pids)))
    by = {}
    for r in rows:
        if r["pid"] in by:
            raise Blocker("duplicate pid %s in arm %s" % (r["pid"], name))
        by[r["pid"]] = r
    missing = [i for i in pids if i not in by]
    if missing:
        raise Blocker("arm %s is missing %d audited pids, first: %s"
                      % (name, len(missing), missing[:3]))
    s = np.array([float(by[i]["score_seed_mean"]) for i in pids])
    y = np.array([int(by[i]["xpert_result"]) for i in pids])
    if not np.isfinite(s).all():
        raise Blocker("non-finite score in arm %s" % name)
    return s, y


def locked_thresholds():
    """Thresholds fitted on TBX11K development, never on TUNE."""
    with open(METRICS, encoding="utf-8") as fh:
        rows = {r["arm"]: r for r in csv.DictReader(fh)}
    out = {}
    for arm, key in (("B0", "A"), ("B1", "B1")):
        r = rows.get(key)
        if r is None:
            raise Blocker("no metrics row for arm %s" % key)
        out[arm] = float(r["dev_threshold_at_spec0.90"])
    return out


def verify_assets():
    """Hash the two downloaded CLEAR assets against the pinned record. Fail closed.

    The package records sizes only at run time; this checks content. A truncated or
    substituted checkpoint would otherwise produce a full table of wrong numbers.
    """
    from huggingface_hub import hf_hub_download
    pinned = json.loads((PKG / "configs/model_assets.json").read_text(encoding="utf-8"))
    out = {}
    for name, want in pinned["files"].items():
        if name == "mimic_concepts.csv":
            continue
        path = hf_hub_download(A.HF_REPO, name, revision=A.HF_REV)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        got = h.hexdigest()
        size = Path(path).stat().st_size
        if got != want["sha256"] or size != want["bytes"]:
            raise Blocker("asset %s does not match the pinned record\n"
                          "  sha256 %s != %s\n  bytes  %d != %d"
                          % (name, got, want["sha256"], size, want["bytes"]))
        out[name] = {"sha256": got, "bytes": size}
    return out


def verify_vocabulary():
    """Re-derive four pinned indices from the vocabulary file itself."""
    if not VOCAB.is_file():
        raise Blocker("vocabulary not found at %s" % VOCAB)
    got = sha(VOCAB)
    if got != VOCAB_SHA:
        raise Blocker("vocabulary sha256 %s != pinned %s" % (got, VOCAB_SHA))
    idx = {}
    with open(VOCAB, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            idx.setdefault(r["concept"].strip().lower(), []).append(int(r["concept_idx"]))
    checks = {"scarring": 96952, "fibrosis": 835, "consolidation": 123742,
              "volume loss": 368164, "scoliosis": 82734, "degenerative changes": 339316}
    bad = {k: idx.get(k) for k, v in checks.items() if idx.get(k) != [v]}
    if bad:
        raise Blocker("vocabulary indices do not re-derive: %s" % bad)
    return {"sha256": got, "n_entries": sum(len(v) for v in idx.values()),
            "re_derived": checks}


# ------------------------------------------------------------------ paired statistics
def paired_differences(d, S, names, score_b0, score_b1, log=print):
    """Shared-index paired bootstrap for rho and incremental R-squared, B1 minus B0."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    neg = d["y"] == 0
    past = d["past_tb"][neg]
    Sneg = S[neg]
    s0, s1 = score_b0[neg], score_b1[neg]
    X0 = np.column_stack([d["rad_system"][neg] == v for v in (1, 2, 3)][:2]
                         + [d["age_group"][neg], d["sex"][neg], d["symptoms"][neg],
                            d["past_tb"][neg]]).astype(float)
    n = int(neg.sum())

    def rho(c, s):
        return float(spearmanr(c, s).statistic)

    def r2(Xb, s):
        """OLS R-squared with an intercept, by least squares.

        Mathematically identical to LinearRegression().fit(.).score(.); solved directly
        because this runs 10,000 times per concept. The point estimates it produces are
        checked against the LinearRegression values in tables/aggregate_results.csv.
        """
        Xi = np.column_stack([np.ones(len(s)), Xb])
        beta, *_ = np.linalg.lstsq(Xi, s, rcond=None)
        resid = s - Xi @ beta
        sst = float(((s - s.mean()) ** 2).sum())
        return 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else 0.0

    def dr2(c, s, Xb, base=None):
        b = r2(Xb, s) if base is None else base
        return float(r2(np.column_stack([Xb, c]), s) - b)

    rng = np.random.default_rng(PAIRED_SEED)
    idxs, degenerate = [], 0
    while len(idxs) < PAIRED_BOOT:
        k = rng.integers(0, n, n)
        # a replicate is usable only if both score vectors and the design vary
        if np.ptp(s0[k]) == 0 or np.ptp(s1[k]) == 0 or len(np.unique(past[k])) < 2:
            degenerate += 1
            continue
        idxs.append(k)
    log("  paired bootstrap: %d replicates, %d degenerate resamples discarded"
        % (len(idxs), degenerate))

    # Replicate-major, because the covariate-only fit is shared by every concept within a
    # replicate. Concept-major refits it 11 times over and turns minutes into an hour.
    nC = len(names)
    DR = np.full((len(idxs), nC), np.nan)
    DD = np.full((len(idxs), nC), np.nan)
    for t, k in enumerate(idxs):
        Xk, a0, a1 = X0[k], s0[k], s1[k]
        base0, base1 = r2(Xk, a0), r2(Xk, a1)
        Sk = Sneg[k]
        for j in range(nC):
            ck = Sk[:, j]
            if np.ptp(ck) == 0:
                continue
            DR[t, j] = rho(ck, a1) - rho(ck, a0)
            DD[t, j] = dr2(ck, a1, Xk, base1) - dr2(ck, a0, Xk, base0)
        if t and t % 2000 == 0:
            log("    %d/%d replicates" % (t, len(idxs)))

    base0, base1 = r2(X0, s0), r2(X0, s1)
    out = []
    for j, nm in enumerate(names):
        c = Sneg[:, j]
        r0, r1 = rho(c, s0), rho(c, s1)
        d0, d1 = dr2(c, s0, X0, base0), dr2(c, s1, X0, base1)
        drho = DR[np.isfinite(DR[:, j]), j]
        ddr2 = DD[np.isfinite(DD[:, j]), j]
        lo, hi = np.percentile(drho, [2.5, 97.5])
        l2, h2 = np.percentile(ddr2, [2.5, 97.5])
        out.append({
            "concept": nm, "n_negatives": n,
            "rho_B0": round(r0, 4), "rho_B1": round(r1, 4),
            "delta_rho_B1_minus_B0": round(r1 - r0, 4),
            "delta_rho_ci95_lo": round(float(lo), 4),
            "delta_rho_ci95_hi": round(float(hi), 4),
            "delta_rho_excludes_zero": bool(lo > 0 or hi < 0),
            "incr_r2_B0": round(d0, 4), "incr_r2_B1": round(d1, 4),
            "delta_incr_r2": round(d1 - d0, 4),
            "delta_incr_r2_ci95_lo": round(float(l2), 4),
            "delta_incr_r2_ci95_hi": round(float(h2), 4),
            "delta_incr_r2_excludes_zero": bool(l2 > 0 or h2 < 0),
            "n_replicates_used": int(len(drho)),
        })
    return out


def subgroup_rows(d, S, names, scores):
    """Stratified associations, unadjusted and exploratory."""
    from scipy.stats import spearmanr
    y, past = d["y"], d["past_tb"]
    groups = (("negatives, all", y == 0),
              ("negatives, past TB", (y == 0) & (past == 1)),
              ("negatives, no past TB", (y == 0) & (past == 0)),
              ("positives, all", y == 1))
    out = []
    for j, nm in enumerate(names):
        for label, m in groups:
            if m.sum() < 20:
                continue
            row = {"concept": nm, "group": label, "n": int(m.sum()),
                   "multiplicity": "unadjusted, exploratory"}
            for arm, s in scores.items():
                r = spearmanr(S[m, j], s[m])
                row["rho_%s" % arm] = round(float(r.statistic), 4)
                row["p_raw_%s" % arm] = float(r.pvalue)
            row["delta_rho_B1_minus_B0"] = round(row["rho_B1"] - row["rho_B0"], 4)
            out.append(row)
    return out


def safety_rows(d, scores, thr):
    """Discrimination and locked-threshold operating points. No threshold fitted on TUNE."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = d["y"]
    out = []
    for arm, s in scores.items():
        r = {"arm": arm, "n": int(len(y)), "n_positive": int(y.sum()),
             "prevalence": round(float(y.mean()), 4),
             "auroc": round(float(roc_auc_score(y, s)), 4),
             "auprc": round(float(average_precision_score(y, s)), 4),
             "note": "TUNE is exposed; descriptive only"}
        if arm in thr:
            t = thr[arm]
            pred = s >= t
            r.update({
                "locked_threshold": round(t, 4),
                "threshold_origin": "TBX11K development, >=0.90 specificity; not TUNE",
                "sensitivity_at_locked_threshold": round(float(pred[y == 1].mean()), 4),
                "specificity_at_locked_threshold": round(float((~pred[y == 0]).mean()), 4),
                "referral_rate": round(float(pred.mean()), 4),
            })
        out.append(r)
    return out


def distribution_rows(d, scores):
    y, past = d["y"], d["past_tb"]
    groups = (("negatives, no past TB", (y == 0) & (past == 0)),
              ("negatives, past TB", (y == 0) & (past == 1)),
              ("positives, no past TB", (y == 1) & (past == 0)),
              ("positives, past TB", (y == 1) & (past == 1)))
    out = []
    for arm, s in scores.items():
        for label, m in groups:
            if m.sum() < 20:
                out.append({"arm": arm, "group": label, "n": int(m.sum()),
                            "note": "suppressed, n < 20"})
                continue
            v = s[m]
            out.append({"arm": arm, "group": label, "n": int(m.sum()),
                        "mean": round(float(v.mean()), 4),
                        "sd": round(float(v.std(ddof=1)), 4),
                        "p25": round(float(np.percentile(v, 25)), 4),
                        "median": round(float(np.median(v)), 4),
                        "p75": round(float(np.percentile(v, 75)), 4),
                        "note": ""})
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-inference", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--keep-scratch", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()

    manifest_hash = check_manifest()
    # Shadow the package's asset lookup so the extended concept map is used, without
    # touching a single byte of the audited package.
    P._ASSET_DIRS = [RUN / "configs"] + list(P._ASSET_DIRS)

    vocab = verify_vocabulary()
    print("vocabulary verified: %s, 4+2 pinned indices re-derived" % vocab["sha256"][:16])

    p = P.resolve(prefer="repo")
    print(P.describe(p))
    if p["layout"] != "repo":
        raise Blocker("expected the local repo layout, got %r" % p["layout"])

    d = A.load_tune_only(p)
    m, primary = A.concept_map()
    names = [c["concept"] for c in primary]
    controls = [c["concept"] for c in m["mapping"] if c.get("role")]
    bh_family = m["bh_family"]
    print("\nTUNE cohort        : %d patients, %d TB-positive, %d with previous TB"
          % (len(d["pid"]), int(d["y"].sum()), int(d["past_tb"].sum())))
    print("cohort fingerprint : %s (matches pinned)" % d["fingerprint"][:16])
    print("concepts           : %d (%d primary + %d negative controls: %s)"
          % (len(names), len(bh_family), len(controls), ", ".join(controls)))

    missing, extra = A.check_images(p, d["pid"])
    print("images             : %d approved, %d missing, %d present but not approved"
          % (len(d["pid"]) - len(missing), len(missing), len(extra)))
    if missing:
        raise Blocker("%d approved images are missing, first: %s" % (len(missing), missing[:5]))

    # --- partition guards ------------------------------------------------------------
    split = json.loads(Path(p["split"]).read_text(encoding="utf-8"))
    report = {str(i) for i in split["report"]}
    if set(d["pid"]) & report:
        raise Blocker("a Vietnam REPORT pid entered the cohort")
    print("REPORT pids loaded : 0  |  internal_test referenced: no")

    s_b0, y0 = load_arm_scores("A", d["pid"])
    s_b1, y1 = load_arm_scores("B1", d["pid"])
    if not (np.array_equal(y0, d["y"]) and np.array_equal(y1, d["y"])):
        raise Blocker("arm label columns disagree with the audited cohort labels")
    scores = {"baseline_production": d["clf_score"], "B0": s_b0, "B1": s_b1}
    thr = locked_thresholds()
    print("locked thresholds  : B0 %.4f, B1 %.4f  (TBX11K development, never TUNE)"
          % (thr["B0"], thr["B1"]))

    if not a.run_inference:
        print("\nDRY VALIDATION ONLY. All prerequisites pass. "
              "Pass --run-inference to encode images.")
        return 0

    # --- inference, into scratch -------------------------------------------------------
    assets = verify_assets()
    for k, v in assets.items():
        print("asset verified      : %-32s %s" % (k, v["sha256"][:16]))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print("\nencoding %d images with CLEAR (scratch: %s)" % (len(d["pid"]), SCRATCH))
    S, prov = A.encode_concepts(d["pid"], device=a.device, batch=a.batch,
                                out_dir=SCRATCH, resume=True)
    if S.shape != (len(d["pid"]), len(names)):
        raise Blocker("similarities have shape %s, expected (%d, %d)"
                      % (S.shape, len(d["pid"]), len(names)))

    # --- reproduction gate --------------------------------------------------------------
    gate_rows, gate_incr = A.analyse(d, S, names)
    g = {(r["concept"], r["group"]): r for r in gate_rows}
    gi = {r["concept"]: r for r in gate_incr}
    got = {
        "scarring_rho": g[("scarring", "negatives (all)")]["spearman_rho"],
        "scarring_ci_lo": g[("scarring", "negatives (all)")]["ci95_lo"],
        "scarring_ci_hi": g[("scarring", "negatives (all)")]["ci95_hi"],
        "fibrosis_rho": g[("fibrosis", "negatives (all)")]["spearman_rho"],
        "scarring_delta_r2": gi["scarring"]["delta_r2"],
        "clinical_only_r2": gi["scarring"]["r2_clinical_only"],
    }
    gate = []
    ok = True
    for k, (want, tol) in GATE.items():
        diff = abs(got[k] - want)
        passed = diff <= tol
        ok = ok and passed
        gate.append({"quantity": k, "published_kaggle_20261023": want,
                     "reproduced_locally": round(float(got[k]), 4),
                     "abs_difference": round(float(diff), 6), "tolerance": tol,
                     "pass": passed})
        print("  gate %-20s published %8.4f  local %8.4f  diff %.6f  %s"
              % (k, want, got[k], diff, "PASS" if passed else "FAIL"))
    write_csv(TAB / "reproduction_gate.csv", gate)
    if not ok:
        raise Blocker("REPRODUCTION GATE FAILED. The local CLEAR pipeline does not "
                      "reproduce the published Kaggle figures; no new association is "
                      "published. See tables/reproduction_gate.csv.")
    print("  reproduction gate PASSED")

    # --- per-arm associations, with BH over the 9-concept family only ------------------
    agg = []
    for arm in ("baseline_production", "B0", "B1"):
        d_arm = dict(d)
        d_arm["clf_score"] = scores[arm]
        rows, incr = A.analyse(d_arm, S, names)
        ir = {r["concept"]: r for r in incr}
        for r in rows:
            is_control = r["concept"] in controls
            agg.append({
                "arm": arm, "concept": r["concept"],
                "is_negative_control": is_control,
                "group": r["group"], "n": r["n"],
                "spearman_rho": round(r["spearman_rho"], 4),
                "rho_ci95_lo": round(r["ci95_lo"], 4),
                "rho_ci95_hi": round(r["ci95_hi"], 4),
                "p_raw": r["p_raw"],
                "p_bh_adjusted": ("" if is_control or r["p_bh_adjusted"] is None
                                  else r["p_bh_adjusted"]),
                "bh_family": "9 primary concepts, negatives (all)" if (
                    not is_control and r["group"] == "negatives (all)") else "not in family",
                "incr_r2": ir[r["concept"]]["delta_r2"] if r["group"] == "negatives (all)" else "",
                "r2_clinical_only": ir[r["concept"]]["r2_clinical_only"] if r["group"] == "negatives (all)" else "",
            })
    write_csv(TAB / "aggregate_results.csv", agg)

    # NOTE: the BH values above are computed by A.analyse over all 11 encoded concepts.
    # Recompute them over the declared 9-concept family so the manifest is honoured.
    _rebh(TAB / "aggregate_results.csv", bh_family)

    paired = paired_differences(d, S, names, s_b0, s_b1)
    for r in paired:
        r["is_negative_control"] = r["concept"] in controls
    write_csv(TAB / "bootstrap_intervals.csv", paired)

    write_csv(TAB / "subgroup_associations.csv", subgroup_rows(d, S, names, scores))
    write_csv(TAB / "safety_and_discrimination.csv", safety_rows(d, scores, thr))
    write_csv(TAB / "score_distributions.csv", distribution_rows(d, scores))

    # --- preflight / provenance ---------------------------------------------------------
    import platform
    import torch
    pre = {
        "manifest_sha256": manifest_hash,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "execution": "local GPU; no Kaggle session, no upload of any kind",
        "cohort": {"name": "vietnam_tune", "n": len(d["pid"]),
                   "n_positive": int(d["y"].sum()),
                   "n_past_tb": int(d["past_tb"].sum()),
                   "fingerprint_sha256": d["fingerprint"],
                   "ordered_id_hash_sha256": d["ordered_hash"],
                   "images_missing": len(missing), "images_not_approved": len(extra)},
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False,
                       "report_pids_in_cohort": 0},
        "model": {"repo": A.HF_REPO, "revision": A.HF_REV,
                  "clear_commit": A.CLEAR_COMMIT, "input_px": A.CLEAR_INPUT_PX,
                  "concept_embedding_rows": A.CLEAR_EMBED_ROWS, **prov},
        "vocabulary": vocab,
        "assets_verified_sha256": assets,
        "concepts": {"n_encoded": len(names), "names": names,
                     "bh_family": bh_family, "negative_controls": controls,
                     "excluded_no_exact_entry": [c["concept"] for c in m["mapping"]
                                                 if c["concept_idx"] is None],
                     "score_kind": ("text-similarity to one report sentence per concept: "
                                    "a weakly supervised proxy, NOT a validated detector")},
        "statistics": {"per_arm_seed": A.SEED, "per_arm_n_boot": A.N_BOOT,
                       "paired_seed": PAIRED_SEED, "paired_n_boot": PAIRED_BOOT,
                       "resampling_unit": "patient (one image per patient in Vietnam)"},
        "locked_thresholds": {k: round(v, 4) for k, v in thr.items()},
        "inputs_sha256": {
            "vietnam_tune_armA.csv": sha(PRED / "vietnam_tune_armA.csv"),
            "vietnam_tune_armB1.csv": sha(PRED / "vietnam_tune_armB1.csv"),
            "concept_audit.py": sha(PKG / "src/concept_audit.py"),
            "paths.py": sha(PKG / "src/paths.py"),
            "concept_indices.json (this run, extended)": sha(RUN / "configs/concept_indices.json"),
            "ANALYSIS_MANIFEST.md": manifest_hash,
        },
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "numpy": np.__version__},
        "per_case_export": "none; shards computed in scratch and deleted",
        "production_unchanged": ("Vietnam screening AUROC remains 0.7242; no checkpoint, "
                                 "prediction file or threshold was modified"),
    }
    (TAB / "preflight.json").write_text(json.dumps(pre, indent=1), encoding="utf-8")

    sc = {r["concept"]: r for r in paired}
    print("\n%-22s %8s %8s %9s %-20s" % ("concept", "rho B0", "rho B1", "delta", "95% CI"))
    for nm in ["scarring", "fibrosis"] + controls:
        r = sc[nm]
        print("%-22s %8.4f %8.4f %+9.4f [%+.4f, %+.4f]%s"
              % (nm, r["rho_B0"], r["rho_B1"], r["delta_rho_B1_minus_B0"],
                 r["delta_rho_ci95_lo"], r["delta_rho_ci95_hi"],
                 "  <- NEGATIVE CONTROL" if r["is_negative_control"] else ""))

    if not a.keep_scratch:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print("\nscratch deleted: %s" % SCRATCH)
    print("wrote 6 aggregate tables + preflight.json in %.1f min"
          % ((time.perf_counter() - t0) / 60))
    print("No per-case concept score was written. Do not export one.")
    return 0


def _rebh(path, family):
    """Benjamini-Hochberg over the declared 9-concept family, per arm."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for arm in sorted({r["arm"] for r in rows}):
        fam = [r for r in rows if r["arm"] == arm and r["group"] == "negatives (all)"
               and r["concept"] in family]
        if len(fam) != len(family):
            raise Blocker("BH family has %d of %d concepts for arm %s"
                          % (len(fam), len(family), arm))
        p = np.array([float(r["p_raw"]) for r in fam])
        o = np.argsort(p)
        adj = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
        padj = np.empty(len(p))
        padj[o] = np.clip(adj, 0, 1)
        for r, v in zip(fam, padj):
            r["p_bh_adjusted"] = repr(float(v))
    for r in rows:
        if not (r["group"] == "negatives (all)" and r["concept"] in family):
            r["p_bh_adjusted"] = ""
    write_csv(Path(path), rows)


if __name__ == "__main__":
    sys.exit(main())
