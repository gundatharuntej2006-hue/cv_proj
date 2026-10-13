# -*- coding: utf-8 -*-
"""P2-08 / P2-09 command line: select preprocessing (D08/D09) and head (D11) on development.

    py -3 scripts/p2_08_09_select.py \
        --scores artifacts/p2/scores/scores_tbx11k_development_v1.parquet \
        --splits data/splits/tbx11k_splits_v1.csv

Reads the development C08 (all variant/head models), builds the candidate table
(sensitivity at 70% specificity, AUC, n, Wilson CI), applies the pre-registered rule, and
writes D08 (CLAHE), D09 (margin) and D11 (head). Decides on development only.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_cls import selection as S                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="P2-08/09: preprocessing and head selection")
    ap.add_argument("--scores", required=True, help="development C08 parquet (all models)")
    ap.add_argument("--splits", help="C01 split file with image_id, category (for labels)")
    ap.add_argument("--target-spec", type=float, default=S.TARGET_SPECIFICITY)
    a = ap.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq

    df = pq.read_table(a.scores).to_pandas()
    if a.splits:
        lab = pd.read_csv(a.splits)[["image_id", "category"]]
        cat = dict(zip(lab.image_id, lab.category))
        label_fn = lambda i: 0 if cat.get(i) in ("healthy", "sick_non_tb") else 1
    else:
        # fallback for the mock, whose ids encode the class
        label_fn = lambda i: 1 if str(i).startswith("tb") else 0

    cands = S.candidate_table(df, label_fn, target_spec=a.target_spec)
    print("candidate table (sensitivity at %.0f%% specificity):" % (100 * a.target_spec))
    print("  %-34s %-11s %7s  %-18s %6s" % ("model_id", "head", "sens", "95% CI", "AUC"))
    for c in cands:
        print("  %-34s %-11s %6.3f  [%.3f, %.3f]  %5.3f"
              % (c.model_id, c.head, c.sensitivity, c.sens_ci_low, c.sens_ci_high, c.auc))

    pre = S.select_preprocessing(cands)
    head = S.select_head(cands, pre["variant"])
    print("\nD08 CLAHE: %s | D09 margin: %s (variant %s)"
          % (pre["clahe"], pre["margin"], pre["variant"]))
    print("D11 head : primary=%s auxiliary=%s" % (head["primary_head"], head.get("auxiliary_head")))

    today = str(date.today())
    base = {"preregistered_at": today, "reviewed_by": "p4", "guide_ack": False,
            "locked": False, "data_used": ["train", "development"]}
    decs = {
        "p2_clahe.json": {
            "decision_id": "D08", "owner": "p2", "title": "CLAHE on/off (H3)",
            "criterion_preregistered": "Chosen on development by sensitivity at 70% specificity, AUC tie-break, among RAD-DINO binary variants.",
            "candidates": ["clahe0", "clahe1"],
            "value": {"clahe": pre["clahe"], "sensitivity": round(pre["sensitivity"], 4),
                      "sens_ci": [round(x, 4) for x in pre["sens_ci"]], "auc": round(pre["auc"], 4)},
            "evidence": [a.scores], **base},
        "p2_crop_margin.json": {
            "decision_id": "D09", "owner": "p2", "title": "Crop margin (H3)",
            "criterion_preregistered": "Chosen on development by sensitivity at 70% specificity, AUC tie-break, among RAD-DINO binary variants.",
            "candidates": [None, 0.05, 0.10, 0.15],
            "value": {"margin": pre["margin"], "variant": pre["variant"],
                      "sensitivity": round(pre["sensitivity"], 4)},
            "evidence": [a.scores], **base},
        "p2_head.json": {
            "decision_id": "D11", "owner": "p2", "title": "Head selection: binary vs three-class",
            "criterion_preregistered": "Binary vs three-class on development by sensitivity at 70% specificity with n; three-class kept as auxiliary if binary wins.",
            "candidates": ["binary", "three_class"],
            "value": head, "evidence": [a.scores], **base},
    }
    for fname, rec in decs.items():
        p = ROOT / "decisions" / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(p, "w", encoding="utf-8"), indent=1)
        print("wrote decisions/%s (%s, locked=false)" % (fname, rec["decision_id"]))


if __name__ == "__main__":
    main()
