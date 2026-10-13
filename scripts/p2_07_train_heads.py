# -*- coding: utf-8 -*-
"""P2-07 command line: train the classifier heads on frozen embeddings, export C08, write D10.

    py -3 scripts/p2_07_train_heads.py \
        --emb artifacts/p2/emb/radino_crop_m0.10_clahe0_v1.npz \
        --splits data/splits/tbx11k_splits_v1.csv \
        --head binary --imbalance logit_adjusted \
        --out artifacts/p2/scores/scores_tbx11k_development_v1.parquet

Trains five seeds, reports the development-AUC spread (D10), and exports the seed-averaged
TB probability as C08. Nothing is selected on calibration or test here; this produces the
development scores P2-08/P2-09 choose from.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_cls import contracts as K                                    # noqa: E402
from p2_cls import heads as H                                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43, 61, 79)


def load_emb(path):
    with np.load(path) as z:
        return z["image_id"].tolist(), z["emb"].astype(np.float32), str(z["variant"])


def main():
    ap = argparse.ArgumentParser(description="P2-07: train heads on frozen embeddings (C08)")
    ap.add_argument("--emb", required=True, help="C07 npz for the chosen variant")
    ap.add_argument("--splits", required=True, help="C01 split file (image_id, split, category)")
    ap.add_argument("--head", default="binary", choices=list(H.HEAD_TYPES))
    ap.add_argument("--imbalance", default="logit_adjusted", choices=list(H.APPROVED_IMBALANCE))
    ap.add_argument("--out", required=True, help="C08 parquet for the development split")
    ap.add_argument("--eval-split", default="development")
    ap.add_argument("--max-epochs", type=int, default=200)
    a = ap.parse_args()

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids, emb, variant = load_emb(a.emb)
    idx = {i: k for k, i in enumerate(ids)}
    splits = pd.read_csv(a.splits)
    need = {"image_id", "split", "category"}
    if not need.issubset(splits.columns):
        raise SystemExit("splits file needs columns %s" % need)
    splits = splits[splits.image_id.isin(idx)].copy()

    tr = splits[splits.split == "train"]
    dv = splits[splits.split == a.eval_split]
    if len(tr) == 0 or len(dv) == 0:
        raise SystemExit("need both 'train' and %r rows in the split file" % a.eval_split)

    def rows_to(df):
        e = emb[[idx[i] for i in df.image_id]]
        cats = df.category.tolist()
        y = (H.binary_labels(cats) if a.head == "binary" else H.three_class_labels(cats))
        return e, y, cats
    Xtr, ytr, _ = rows_to(tr)
    Xdv, ydv, _ = rows_to(dv)
    y_dev_bin = H.binary_labels(dv.category.tolist())

    cfg = H.TrainConfig(head_type=a.head, imbalance=a.imbalance, max_epochs=a.max_epochs)
    print("training %d seeds on %d train / %d %s embeddings (variant %s) ..."
          % (len(SEEDS), len(Xtr), len(Xdv), a.eval_split, variant))
    results, mean_auc, std_auc = H.train_seeds(Xtr, ytr, Xdv, ydv, cfg, seeds=SEEDS)
    for r in results:
        print("  seed %3d: dev AUC %.4f (best epoch %d)" % (r.config.seed, r.dev_auc, r.best_epoch))
    print("development AUC: %.4f +- %.4f  (seed spread, D10)" % (mean_auc, std_auc))

    p_tb = H.seed_averaged_p_tb(results, Xdv)
    # logit_tb is required for binary rows; derive it from the seed-averaged probability
    p_clip = np.clip(p_tb, 1e-6, 1 - 1e-6)
    logit_tb_avg = np.log(p_clip / (1 - p_clip))
    model_id = "radino_%s_%s" % (variant, "bin" if a.head == "binary" else "3c")
    recs = []
    for j, (_, row) in enumerate(dv.reset_index(drop=True).iterrows()):
        recs.append({
            "image_id": str(row.image_id), "dataset": "tbx11k", "split": a.eval_split,
            "model_id": model_id, "head": a.head, "variant": variant,
            "clahe": "clahe1" in variant, "margin": None,
            "logit_tb": float(logit_tb_avg[j]) if a.head == "binary" else None,
            "logits_3c": None,
            "p_tb_raw": float(p_tb[j]),
            "p3_healthy": None, "p3_sick_non_tb": None, "p3_tb": None,
            "is_primary": False,                    # D11 sets the primary later
            "model_sha": "seedavg_%s" % "_".join(str(s) for s in SEEDS),
            "seed": SEEDS[0],
        })
    table = pa.Table.from_pylist(recs, schema=K.C08_SCHEMA)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    bad = K.check_c08_invariants(pd.DataFrame(recs))
    if bad:
        raise SystemExit("C08 invariants violated: %s" % bad)
    print("wrote %s (%d rows), invariants OK" % (out, len(recs)))

    d10 = {
        "decision_id": "D10", "owner": "p2",
        "title": "Classifier head: imbalance method and training length",
        "criterion_preregistered": ("Imbalance handled by exactly one of weighted_sampler or "
                                    "logit_adjusted (no other method). Training length set by "
                                    "development early stopping. Seed spread reported."),
        "preregistered_at": str(date.today()),
        "candidates": list(H.APPROVED_IMBALANCE),
        "value": {"head": a.head, "imbalance": a.imbalance, "seeds": list(SEEDS),
                  "dev_auc_mean": round(mean_auc, 4), "dev_auc_std": round(std_auc, 4),
                  "best_epochs": [r.best_epoch for r in results], "variant": variant},
        "evidence": [str(out)],
        "data_used": ["train", a.eval_split],
        "reviewed_by": "p4", "guide_ack": False, "locked": False,
    }
    dec = ROOT / "decisions" / "p2_training_recipe.json"
    dec.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d10, open(dec, "w", encoding="utf-8"), indent=1)
    print("wrote %s (D10, locked=false pending P2-08/P2-09)" % dec)


if __name__ == "__main__":
    main()
