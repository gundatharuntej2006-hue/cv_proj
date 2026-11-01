# -*- coding: utf-8 -*-
"""P2-08/09: the preprocessing sweep that decides D08 (CLAHE) and D09 (crop margin).

Every variant gets the identical head, seeds, early stopping and criterion; only the
embeddings differ. So the comparison measures preprocessing and nothing else.

The criterion is the **amended** one (see `docs/p2_selection_criterion_issue.md`):
sensitivity at 95% specificity, AUC as tie-break, with sensitivity at 70% specificity
reported alongside because that is the clinical operating point. The original
"sensitivity at 70%" criterion is degenerate here — every variant scores 1.0000 — so
selecting on it would hand the decision silently to the tie-break.

Differences are judged by a **paired** bootstrap against the full image, because the
variants are scored on the same cases and a paired interval is far tighter than two
independent ones. A CI crossing zero means indistinguishable, which is the honest verdict
for most of these.

    py -3 scripts/p2_08_09_sweep_variants.py
    py -3 scripts/p2_08_09_sweep_variants.py --variants full_clahe0 crop_m0.10_clahe0

Writes artifacts/p2/dryrun/d08_d09_sweep.json and prints the table that goes in the report.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import heads as H                                       # noqa: E402
from p2_cls import registry as R                                    # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

EMB_DIR = ROOT / "artifacts/p2/emb"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT = ROOT / "artifacts/p2/dryrun/d08_d09_sweep.json"
REFERENCE = "full_clahe0"            # everything is compared against the plain full image
SEEDS = (17, 29, 43, 61, 79)
SPEC_SELECT = 0.95                   # the amended selection point
SPEC_REPORT = 0.70                   # the WHO screening point, reported not selected on
N_BOOT = 2000
ALL_VARIANTS = [
    "full_clahe0", "full_clahe1",
    "crop_m0.00_clahe0", "crop_m0.00_clahe1",
    "crop_m0.05_clahe0", "crop_m0.05_clahe1",
    "crop_m0.10_clahe0", "crop_m0.10_clahe1",
    "crop_m0.15_clahe0", "crop_m0.15_clahe1",
]


def emb_path(variant):
    """C04-built embeddings win if both exist; they are the ones Person 1's manifest made."""
    for suffix in ("_c04_v1.npz", "_v1.npz"):
        p = EMB_DIR / ("radino_tbx11k_%s%s" % (variant, suffix))
        if p.exists():
            return p
    return None


def load_variant(variant, want_ids=None):
    p = emb_path(variant)
    if p is None:
        return None
    with np.load(p) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    if want_ids is not None:
        idx = {i: k for k, i in enumerate(ids)}
        missing = [i for i in want_ids if i not in idx]
        if missing:
            raise SystemExit("%s is missing %d of the reference images (first: %s)"
                             % (variant, len(missing), missing[:3]))
        emb = emb[[idx[i] for i in want_ids]]
        ids = list(want_ids)
    return ids, emb


def splits_for(ids):
    import pandas as pd
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    out = {}
    for sp in ("train", "development"):
        g = c01[c01.split == sp]
        out[sp] = ([idx[i] for i in g.image_id], H.binary_labels(g.category.tolist()))
    return out


def evaluate(variant, emb, splits, log=print):
    (tr, ytr), (dv, ydv) = splits["train"], splits["development"]
    res, mean_auc, sd = H.train_seeds(
        emb[tr], ytr, emb[dv], ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    p = R.p_tb_from_logits(R.head_logits(emb[dv], W, b), 2)
    s95, thr95, n_pos, tp95 = SEL.sensitivity_at_specificity(ydv, p, SPEC_SELECT)
    s70 = SEL.sensitivity_at_specificity(ydv, p, SPEC_REPORT)[0]
    lo, hi = SEL.wilson_ci(tp95, n_pos)
    log("  %-20s AUC %.4f (sd %.4f) | sens@95 %.4f [%.4f, %.4f] | sens@70 %.4f"
        % (variant, mean_auc, sd, s95, lo, hi, s70))
    return {"variant": variant, "auc": round(float(mean_auc), 4),
            "seed_sd": round(float(sd), 4),
            "sens_at_95": round(float(s95), 4),
            "sens_at_95_ci": [round(lo, 4), round(hi, 4)],
            "sens_at_70": round(float(s70), 4),
            "threshold_at_95": round(float(thr95), 6), "_p": p, "_y": ydv}


def paired_bootstrap(y, p_a, p_b, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], p_a[k]) - SEL.auc(y[k], p_b[k]))
    d = np.asarray(d)
    return (float(SEL.auc(y, p_a) - SEL.auc(y, p_b)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    have = [v for v in args.variants if emb_path(v) is not None]
    missing = [v for v in args.variants if emb_path(v) is None]
    if REFERENCE not in have:
        raise SystemExit("the reference variant %s has no embeddings" % REFERENCE)
    if missing:
        print("not embedded yet, skipped: %s" % ", ".join(missing))

    # every variant is restricted to the reference's image set, so the paired bootstrap
    # really is paired and the AUCs are over identical cases
    ref_ids, ref_emb = load_variant(REFERENCE)
    splits = splits_for(ref_ids)
    print("%d images | train %d | development %d (%d TB)\n"
          % (len(ref_ids), len(splits["train"][0]), len(splits["development"][0]),
             int(splits["development"][1].sum())))
    print("variants (identical head, seeds %s, criterion sens@%.0f%% spec):"
          % (list(SEEDS), SPEC_SELECT * 100))

    t0 = time.perf_counter()
    rows = []
    for v in have:
        _ids, emb = load_variant(v, want_ids=ref_ids)
        rows.append(evaluate(v, emb, splits))

    ydv = rows[0]["_y"]
    ref = next(r for r in rows if r["variant"] == REFERENCE)
    print("\npaired bootstrap, dAUC vs %s (%d resamples):" % (REFERENCE, args.boot))
    for r in rows:
        if r["variant"] == REFERENCE:
            r["delta_auc"] = 0.0
            r["ci95"] = [0.0, 0.0]
            r["verdict"] = "reference"
            continue
        d, lo, hi = paired_bootstrap(ydv, r["_p"], ref["_p"], n_boot=args.boot)
        r["delta_auc"] = round(d, 4)
        r["ci95"] = [round(lo, 4), round(hi, 4)]
        r["verdict"] = "worse" if hi < 0 else "better" if lo > 0 else "no difference"
        print("  %-20s %+.4f [%+.4f, %+.4f]  %s"
              % (r["variant"], d, lo, hi, r["verdict"]))

    # the amended criterion, applied explicitly so the report can state what decided it
    ranked = sorted(rows, key=lambda r: (-r["sens_at_95"], -r["auc"]))
    winner = ranked[0]
    tied = [r["variant"] for r in ranked if r["sens_at_95"] == winner["sens_at_95"]]
    decided_by = "sens_at_95" if len(tied) == 1 else "auc tie-break among %s" % tied
    print("\nselected: %s (sens@95 %.4f, AUC %.4f) -- decided by %s"
          % (winner["variant"], winner["sens_at_95"], winner["auc"], decided_by))
    better = [r["variant"] for r in rows if r["verdict"] == "better"]
    print("variants significantly better than the full image: %s"
          % (", ".join(better) if better else "none"))

    out = {"created": time.strftime("%Y-%m-%d"), "reference": REFERENCE,
           "n_images": len(ref_ids), "n_dev": len(splits["development"][0]),
           "n_dev_tb": int(splits["development"][1].sum()),
           "seeds": list(SEEDS), "n_boot": args.boot,
           "criterion": {"select_at_specificity": SPEC_SELECT,
                         "report_at_specificity": SPEC_REPORT,
                         "tie_break": "auc",
                         "amended": "see docs/p2_selection_criterion_issue.md"},
           "variants": [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in rows],
           "selected": winner["variant"], "decided_by": decided_by,
           "significantly_better_than_reference": better,
           "skipped_not_embedded": missing,
           "elapsed_min": round((time.perf_counter() - t0) / 60, 1),
           "note": ("Provisional split (Person 1's C01 pending), development only; "
                    "internal_test untouched. D08/D09 are not locked until this is rerun "
                    "on the real C01.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s (%.1f min)" % (OUT, out["elapsed_min"]))


if __name__ == "__main__":
    sys.exit(main())
