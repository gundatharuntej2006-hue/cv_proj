# -*- coding: utf-8 -*-
"""One-shot head-to-head of the frozen candidate v3 against a commercial CAD score on an
untouched cohort, exactly as PROTOCOL.md. No training path exists in this script.

    py -3 headtohead.py --build-manifest cohort.csv --images DIR --holder-statement "..."
    py -3 headtohead.py --dry-run           # validate manifest + images, score nothing
    py -3 headtohead.py --run               # the single scoring pass; writes LOCK; refuses to repeat

cohort.csv columns: pid, image, xpert_result (0/1), past_tb (0/1), scanner, comparator_score,
                    comparator_version, [cluster], [sex, age_group, c, f, wl, ns, contact, dm]
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
LOCK = RUN / "LOCK"
MANIFEST = RUN / "cohort_manifest.json"
sys.path.insert(0, str(ROOT / "src"))

BUNDLE = ROOT / "artifacts/p2/bundle_v3_candidate"
HEAD_SHA = "7f61a082"
N_BOOT, BOOT_SEED = 4000, 0
NI_MARGIN = 0.03
MIN_CASES = 400
REQUIRED = ("pid", "image", "xpert_result", "past_tb", "scanner", "comparator_score", "comparator_version")
QUESTIONNAIRE = ("sex", "age_group", "c", "f", "wl", "ns", "contact", "dm")


class Blocker(SystemExit):
    pass


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_protocol():
    h = sha(RUN / "PROTOCOL.md")
    p = RUN / "protocol_hash.txt"
    if p.is_file() and p.read_text(encoding="utf-8").split()[0] != h:
        raise Blocker("PROTOCOL.md changed after hashing; write a new numbered protocol version instead")
    p.write_text(h + "  PROTOCOL.md\n", encoding="utf-8")
    return h


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def paired(y, pa, pb, units, rng, n=N_BOOT):
    """Paired bootstrap of AUC(pb) - AUC(pa). Resampling unit = cluster when units vary, else participant."""
    uniq = np.unique(units)
    groups = [np.where(units == u)[0] for u in uniq]
    d = []
    for _ in range(n):
        if len(uniq) < len(y):
            k = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        else:
            k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        d.append(auc(y[k], pb[k]) - auc(y[k], pa[k]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(auc(y, pb) - auc(y, pa), 4), round(float(lo), 4), round(float(hi), 4), ("cluster" if len(uniq) < len(y) else "participant")


def decision(lo):
    if lo > 0:
        return "SUPERIOR (lower bound > 0)"
    if lo > -NI_MARGIN:
        return "NON-INFERIOR at margin %.2f" % NI_MARGIN
    return "NOT NON-INFERIOR at margin %.2f" % NI_MARGIN


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise Blocker("empty cohort file")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise Blocker("cohort file missing columns: %s" % missing)
    for r in rows:
        if r["xpert_result"] not in ("0", "1") or r["past_tb"] not in ("0", "1"):
            raise Blocker("pid %s: xpert_result and past_tb must be 0/1" % r["pid"])
    if len({r["pid"] for r in rows}) != len(rows):
        raise Blocker("duplicate pid in cohort file")
    return rows


def build_manifest(cohort_csv, images_dir, holder_statement):
    rows = load_rows(cohort_csv)
    images_dir = Path(images_dir)
    missing = [r["pid"] for r in rows if not (images_dir / r["image"]).is_file()]
    if missing:
        raise Blocker("%d images missing, e.g. %s" % (len(missing), missing[:3]))
    n_pos = sum(int(r["xpert_result"]) for r in rows)
    m = {"cohort_csv": str(Path(cohort_csv).resolve()), "cohort_csv_sha256": sha(cohort_csv),
         "images_dir": str(images_dir.resolve()), "n": len(rows), "positives": n_pos,
         "previous_tb": sum(int(r["past_tb"]) for r in rows),
         "scanners": sorted({r["scanner"] for r in rows}),
         "comparator_versions": sorted({r["comparator_version"] for r in rows}),
         "cluster_column_present": "cluster" in rows[0],
         "questionnaire_present": all(q in rows[0] for q in QUESTIONNAIRE),
         "holder_statement_no_prior_team_access": holder_statement,
         "powered_for_protocol": n_pos >= MIN_CASES,
         "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    MANIFEST.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print("manifest: n %d, positives %d (%s), scanners %s" % (
        m["n"], n_pos, "powered" if m["powered_for_protocol"] else "UNDERPOWERED", m["scanners"]))
    return m


def check_bundle():
    if not (BUNDLE / "head.npz").is_file():
        raise Blocker("candidate bundle missing")
    if not sha(BUNDLE / "head.npz").startswith(HEAD_SHA):
        raise Blocker("candidate head is not the frozen one (%s)" % HEAD_SHA)
    meta = json.loads((BUNDLE / "bundle.json").read_text(encoding="utf-8"))
    if "NOT PROMOTED" not in meta["status"]:
        raise Blocker("bundle status changed")
    return meta


def score_cohort(rows, images_dir, log):
    from p2_cls import pipeline_v2 as P2
    b = P2.load_bundle_v2(BUNDLE)
    out = []
    for i, r in enumerate(rows):
        clin = None
        if all(q in r for q in QUESTIONNAIRE):
            clin = {q: int(r[q]) for q in QUESTIONNAIRE}
            clin["rad_system"] = 3                                    # reference scanner: no site term (PROTOCOL 5)
        res = P2.run_case_v2(Path(images_dir) / r["image"], b, clinical=clin)
        a = res["analysis"]
        out.append({"pid": r["pid"], "status": res["status"], "reasons": ";".join(res.get("reasons", [])),
                    "image_logit": a["image_logit"] if a else None,
                    "screen_positive_0.70": a["operating_points"]["screen_positive_at_spec_0.70"] if a else None,
                    "argmax_row": a["argmax_region"]["row"] if a else None, "argmax_col": a["argmax_region"]["col"] if a else None,
                    "p_fused_no_scanner": (a["fusion"] or {}).get("p_fused") if a else None})
        if (i + 1) % 100 == 0:
            log("  scored %d / %d" % (i + 1, len(rows)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", metavar="COHORT_CSV")
    ap.add_argument("--images")
    ap.add_argument("--holder-statement", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    TAB.mkdir(exist_ok=True)
    proto = check_protocol()
    logf = open(RUN / "run.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    if a.build_manifest:
        if not a.images or not a.holder_statement:
            raise Blocker("--images and --holder-statement are required to build the manifest")
        build_manifest(a.build_manifest, a.images, a.holder_statement)
        return 0
    if not MANIFEST.is_file():
        raise Blocker("no cohort manifest; run --build-manifest first")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(m["cohort_csv"]) != m["cohort_csv_sha256"]:
        raise Blocker("cohort file changed since the manifest was built")
    meta = check_bundle()
    rows = load_rows(m["cohort_csv"])
    log("protocol %s | bundle %s | cohort n %d positives %d%s" % (
        proto[:12], meta["model"]["weight_sha256"][:12], len(rows), m["positives"],
        "" if m["powered_for_protocol"] else " | UNDERPOWERED (< %d cases)" % MIN_CASES))
    if a.dry_run or not a.run:
        log("DRY RUN: manifest and bundle verified; nothing scored. Pass --run for the single scoring pass.")
        return 0
    if LOCK.is_file():
        raise Blocker("LOCK present: the single scoring pass has already been made (%s)" % LOCK.read_text(encoding="utf-8").strip())
    LOCK.write_text("scoring started %s protocol %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), proto[:12]), encoding="utf-8")

    scored = score_cohort(rows, m["images_dir"], log)
    with open(TAB / "per_case_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    keep = [i for i, s in enumerate(scored) if s["status"] == "ANALYSED"]
    excl = {"quality_rejected": len(scored) - len(keep)}
    y = np.array([int(rows[i]["xpert_result"]) for i in keep])
    past = np.array([int(rows[i]["past_tb"]) for i in keep])
    sc = np.array([rows[i]["scanner"] for i in keep])
    units = np.array([rows[i].get("cluster", rows[i]["pid"]) for i in keep])
    ours = np.array([scored[i]["image_logit"] for i in keep], float)
    comp = np.array([float(rows[i]["comparator_score"]) for i in keep])
    rng = np.random.default_rng(BOOT_SEED)

    results = {"eligible_n": int(len(y)), "positives": int(y.sum()), "exclusions": excl,
               "powered": bool(y.sum() >= MIN_CASES), "protocol_sha256": proto}
    d, lo, hi, unit = paired(y, comp, ours, units, rng)
    results["primary_image_only_minus_comparator"] = {"delta": d, "ci95": [lo, hi], "resampling_unit": unit,
                                                      "decision": decision(lo) if y.sum() >= MIN_CASES else "INCONCLUSIVE (underpowered): " + decision(lo)}
    results["auroc"] = {"candidate_image_only": round(auc(y, ours), 4), "comparator": round(auc(y, comp), 4)}
    if past.sum() > 0 and y[past == 1].sum() > 0:
        d2, lo2, hi2, _ = paired(y[past == 1], comp[past == 1], ours[past == 1], units[past == 1], rng)
        results["secondary_previous_tb"] = {"n": int(past.sum()), "positives": int(y[past == 1].sum()),
                                            "delta": d2, "ci95": [lo2, hi2], "status": "EXPLORATORY (see PROTOCOL 7)"}
    fused = [scored[i]["p_fused_no_scanner"] for i in keep]
    if all(v is not None for v in fused):
        f = np.array(fused, float)
        d3, lo3, hi3, _ = paired(y, comp, f, units, rng)
        results["secondary_fused_no_scanner_minus_comparator"] = {"delta": d3, "ci95": [lo3, hi3]}
    per_sc = {}
    for s in np.unique(sc):
        k = sc == s
        if len(np.unique(y[k])) == 2:
            per_sc[str(s)] = {"n": int(k.sum()), "candidate": round(auc(y[k], ours[k]), 4), "comparator": round(auc(y[k], comp[k]), 4)}
    results["secondary_within_scanner"] = per_sc
    thr = meta["thresholds"]["logit_at_spec_0.70"]
    ref = ours >= thr
    results["secondary_operating_point_dev_spec0.70"] = {
        "threshold_logit": thr, "sensitivity": round(float(ref[y == 1].mean()), 4),
        "specificity": round(float((~ref[y == 0]).mean()), 4),
        "referral_burden_by_scanner": {str(s): round(float(ref[sc == s].mean()), 4) for s in np.unique(sc)}}
    from sklearn.metrics import average_precision_score
    results["secondary_auprc"] = {"candidate": round(float(average_precision_score(y, ours)), 4),
                                  "comparator": round(float(average_precision_score(y, comp)), 4)}
    (TAB / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    outs = sorted(p for p in TAB.rglob("*") if p.is_file())
    (RUN / "OUTPUT_MANIFEST.sha256").write_text("".join("%s  %s\n" % (sha(p), p.relative_to(RUN)) for p in outs), encoding="utf-8")
    log("primary: %+.4f [%+.4f, %+.4f] (%s resampling) -> %s" % (d, lo, hi, unit, results["primary_image_only_minus_comparator"]["decision"]))
    log("done; LOCK written; outputs hashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
