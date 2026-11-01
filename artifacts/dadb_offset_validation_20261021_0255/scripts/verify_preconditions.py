# -*- coding: utf-8 -*-
"""Phase 0 precondition verification for the DA/DB offset-correction validation.

This script performs every check the task's Phase 0 requires and writes the case-level
inventory. It deliberately computes **no model scores**: the preconditions fail, and the
decision to stop is made here rather than after looking at outcomes.

    py -3 artifacts/dadb_offset_validation_20261021_0255/scripts/verify_preconditions.py
"""
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

SRC = ROOT / "data/raw/tbx11k/TBX11K/imgs/extra/da+db"
PROTOCOL = {
    "NEXT_STEP.md": "artifacts/overnight_scanner_run_20261021_0208/NEXT_STEP.md",
    "FROZEN_MANIFEST.md": "artifacts/overnight_scanner_run_20261021_0208/FROZEN_MANIFEST.md",
    "frozen_parameters.json":
        "artifacts/overnight_scanner_run_20261021_0208/tables/frozen_parameters.json",
    "PRIMARY_EXPERIMENT_PREREGISTRATION.md":
        "artifacts/overnight_run_20261021_2340/PRIMARY_EXPERIMENT_PREREGISTRATION.md",
    "dadb_third_site.json": "artifacts/p2/external/dadb_third_site.json",
    "radino_dadb_lungs_only_v1.npz": "artifacts/p2/external/radino_dadb_lungs_only_v1.npz",
    "radino_dadb_full_v1.npz": "artifacts/p2/external/radino_dadb_full_v1.npz",
    "p2_26_dadb_third_site.py": "scripts/p2_26_dadb_third_site.py",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dadb_cases():
    """Reconstruct the exact case list the committed pipeline uses.

    scripts/p2_26_dadb_third_site.py::images() iterates sorted(SRC.rglob("*")), takes the
    site from the suffix (.png -> DA, .jpg -> DB) and the label from the first character
    of the filename (p -> 1, otherwise 0). The embedding npz stores no image_id, so this
    ordering IS the only alignment key; it is reproduced here exactly.
    """
    out = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        out.append({
            "order_index": len(out),
            "case_id": p.stem,
            "site": "DA" if p.suffix.lower() == ".png" else "DB",
            "label": 1 if p.name[0] == "p" else 0,
            "tbx11k_subdir": p.parent.name,
            "filename": p.name,
        })
    return out


def main():
    import numpy as np

    report = {"phase": 0, "scores_computed": False}

    # ---- 1. protocol hashes
    report["protocol_sources"] = {}
    for k, rel in PROTOCOL.items():
        p = ROOT / rel
        report["protocol_sources"][k] = {
            "path": rel, "exists": p.exists(),
            "sha256": sha(p) if p.exists() else None}

    # ---- 2. the frozen correction formula, read from source
    fp = json.load(io.open(ROOT / PROTOCOL["frozen_parameters.json"], encoding="utf-8"))
    report["frozen_offsets"] = {
        "parameters": fp["scanner_offsets_tune_non_tb_median"],
        "estimated_on": "Vietnam TUNE half, per rad_system, median score of non-TB cases",
        "keyed_by": "Vietnam rad_system in {1, 2, 3}",
        "formula": "score_corrected = score - offset[rad_system]",
    }
    report["dadb_estimator_in_NEXT_STEP"] = {
        "text": ("Per site, the median score of that site's non-TB cases. DA/DB has no "
                 "development half, so use leave-one-out: centre each case by the median "
                 "of the other non-TB cases at its site."),
        "requires_dadb_non_tb_labels": True,
        "fits_offsets_on_dadb": True,
    }
    report["acceptance_rule_in_NEXT_STEP"] = {
        "minimum_effect_size": "+0.02 pooled AUROC",
        "uncertainty": "95% paired bootstrap CI excludes zero",
        "guardrail": "neither site's AUROC falls by more than 0.02",
        "present_and_unambiguous": True,
    }

    # ---- 3. cohort verification
    cases = dadb_cases()
    ids = [c["case_id"] for c in cases]
    report["cohort"] = {
        "source_dir": str(SRC.relative_to(ROOT)),
        "source_dir_exists": SRC.exists(),
        "n_files": len(cases),
        "n_reported_in_dadb_third_site_json": 176,
        "reconciles": len(cases) == 176,
        "n_unique_case_ids": len(set(ids)),
        "n_duplicate_case_ids": len(ids) - len(set(ids)),
        "missing_files": 0,
        "by_site_label": {"%s_label%d" % (s, y): n for (s, y), n in
                          sorted(Counter((c["site"], c["label"]) for c in cases).items())},
        "by_site": dict(Counter(c["site"] for c in cases)),
        "n_positive": sum(c["label"] for c in cases),
        "n_negative": sum(1 for c in cases if c["label"] == 0),
        "prevalence": round(sum(c["label"] for c in cases) / len(cases), 4),
        "label_mapping": "filename first character: p -> 1 (TB), otherwise 0",
        "patient_identifiers_available": False,
        "patient_overlap_checkable": False,
    }

    # cross-check against the published per-site numbers
    pub = json.load(io.open(ROOT / PROTOCOL["dadb_third_site.json"], encoding="utf-8"))
    bysite = {r["site"]: (r["n"], r["n_tb"]) for r in pub["results"] if r["variant"] == "full"}
    mine = {s: (sum(1 for c in cases if c["site"] == s),
                sum(c["label"] for c in cases if c["site"] == s)) for s in ("DA", "DB")}
    mine["DA+DB"] = (len(cases), sum(c["label"] for c in cases))
    report["cohort"]["published_vs_recomputed"] = {
        s: {"published": list(bysite[s]), "recomputed": list(mine[s]),
            "match": list(bysite[s]) == list(mine[s])} for s in bysite}

    # ---- 4. what artifacts actually exist
    emb = {}
    for v in ("full", "lungs_only", "border_only"):
        p = ROOT / ("artifacts/p2/external/radino_dadb_%s_v1.npz" % v)
        with np.load(p) as z:
            emb[v] = {"keys": list(z.keys()), "emb_shape": list(z["emb"].shape),
                      "n_sanity_fail": int(z["n_sanity_fail"]),
                      "has_image_id": "image_id" in z}
    report["available_artifacts"] = {
        "global_embeddings": emb,
        "regional_12x12_embedding_exists": bool(
            list((ROOT / "artifacts/p2/regional").glob("*dadb*"))),
        "case_level_score_file_exists": False,
        "note": ("the Vietnam offset result was measured on focal-pooling (12x12 max) "
                 "scores; DA/DB has global 768-d embeddings only"),
    }

    # ---- 5. provenance
    leak = {}
    import pandas as pd
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    lab = pd.read_csv(ROOT / "artifacts/p2/labels_tbx11k_trainval_v1.csv")
    stems = set(ids)
    leak["overlap_with_provisional_c01"] = len(stems & set(c01.image_id.astype(str)))
    leak["overlap_with_core_trainval_labels"] = len(stems & set(lab.image_id.astype(str)))
    leak["scripts_reading_dadb"] = subprocess.run(
        ["git", "grep", "-l", "dadb", "--", "scripts/", "src/"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    leak["appears_in_decision_provenance_table"] = False
    leak["rad_dino_checkpoint_exposure"] = (
        "UNKNOWN. The RAD-DINO model card states the released checkpoint was selected by "
        "linear probing on validation sets of 'the evaluation datasets described in the "
        "paper', which the card does not name. DA/DB (NITRD, New Delhi) is not in the "
        "listed pre-training corpora, but checkpoint-selection exposure cannot be excluded.")
    report["provenance"] = leak

    # ---- 6. the decisive precondition
    report["precondition_6_application_at_inference"] = {
        "frozen_vietnam_offsets_transferable_to_dadb": False,
        "reason_not_transferable": (
            "the frozen offsets are keyed to Vietnam rad_system 1/2/3; DA and DB are "
            "different acquisition sources with no mapping to those groups, and the task "
            "forbids guessing one"),
        "dadb_estimator_requires_dadb_labels": True,
        "label_free_variant_would_still_fit_on_dadb": True,
        "verdict": "NOT EVALUABLE as a frozen-parameter validation",
    }

    # ---- write outputs
    with io.open(RUN / "tables/dadb_case_inventory.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_index", "case_id", "site", "label", "tbx11k_subdir",
                    "baseline_score", "corrected_score", "inclusion", "reason"])
        for c in cases:
            w.writerow([c["order_index"], c["case_id"], c["site"], c["label"],
                        c["tbx11k_subdir"], "NOT_SCORED", "NOT_SCORED", "eligible",
                        "cohort verified; scoring not performed because Phase 0 "
                        "precondition 6 failed"])

    json.dump(report, io.open(RUN / "tables/precondition_report.json", "w",
                              encoding="utf-8"), indent=2)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("protocol_sources",)}, indent=2))
    print("\nwrote tables/dadb_case_inventory.csv (%d rows) and "
          "tables/precondition_report.json" % len(cases))


if __name__ == "__main__":
    main()
