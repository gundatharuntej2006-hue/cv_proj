# -*- coding: utf-8 -*-
"""End-to-end P2 dry run on REAL data with a provisional split.

Person 1's C01 does not exist yet, so this builds a provisional split to the D01
specification (development = stratified 10% of official train; official val halved,
stratified, into calibration and internal_test; 5-category stratification; seed 20260921)
and runs the whole P2 chain on the real TBX11K embeddings and labels:

    provisional C01 -> dedup (C17) -> head training (P2-07) -> selection (P2-08/09)
                    -> registry + R2 reproduction (P2-13)

Purpose is de-risking, not results. It proves the chain works on real images and surfaces
integration bugs before C01 arrives; when the real C01 lands, point --splits at it and
re-run. **Nothing here locks a decision**: the selection outcome is provisional because the
split is provisional.

The internal_test half is created and then never touched — the dry run uses train and
development only, so the sealed-split discipline holds even in rehearsal.

    py -3 scripts/p2_dryrun_full_pipeline.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import heads as H                                        # noqa: E402
from p2_cls import registry as R                                     # noqa: E402
from p2_cls import selection as S                                    # noqa: E402

SEED = 20260921            # D01
LABELS = Path("C:/Users/atuld/Downloads/P2_handoff_for_C01/labels_tbx11k_trainval_v1.csv")
EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz"
OUT = ROOT / "artifacts/p2/dryrun"


def build_provisional_c01(lab, seed=SEED):
    """D01 spec: development = stratified 10% of official train; official val halved
    (stratified) into calibration / internal_test. Stratify on the 5 categories."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for cat, g in lab.groupby("category"):
        tr = g[g.official_split == "train"].sample(frac=1.0, random_state=seed)
        va = g[g.official_split == "val"].sample(frac=1.0, random_state=seed)
        n_dev = int(round(len(tr) * 0.10))
        for i, r in enumerate(tr.itertuples()):
            rows.append((r.image_id, "development" if i < n_dev else "train"))
        half = len(va) // 2
        for i, r in enumerate(va.itertuples()):
            rows.append((r.image_id, "calibration" if i < half else "internal_test"))
    sp = pd.DataFrame(rows, columns=["image_id", "split"])
    return lab.merge(sp, on="image_id", how="inner")


def main():
    import pandas as pd
    if not LABELS.exists() or not EMB.exists():
        raise SystemExit("need the real labels CSV and embeddings")
    OUT.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(LABELS)
    c01 = build_provisional_c01(lab)
    c01.to_csv(OUT / "provisional_c01.csv", index=False)

    print("=== provisional C01 (D01 spec, seed %d) ===" % SEED)
    counts = c01.groupby("split").size()
    tb = c01[c01.tb_label == 1].groupby("split").size()
    for s in ("train", "development", "calibration", "internal_test"):
        print("  %-14s n=%5d  TB=%4d" % (s, counts.get(s, 0), tb.get(s, 0)))
    # D01 acceptance criteria
    ok_dev = abs(counts.get("development", 0) - 660) <= 2
    ok_cal = abs(counts.get("calibration", 0) - 900) <= 2
    print("  acceptance: development ~660 -> %s | calibration ~900 -> %s"
          % ("OK" if ok_dev else "OFF", "OK" if ok_cal else "OFF"))

    with np.load(EMB) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = c01[c01.image_id.isin(idx)]

    def take(split):
        g = c01[c01.split == split]
        X = emb[[idx[i] for i in g.image_id]]
        y = H.binary_labels(g.category.tolist())
        return g, X, y

    g_tr, Xtr, ytr = take("train")
    g_dv, Xdv, ydv = take("development")
    print("\n=== P2-07: head training on real embeddings (train -> development) ===")
    print("  train %d (%d TB) | development %d (%d TB)"
          % (len(ytr), ytr.sum(), len(ydv), ydv.sum()))

    results = {}
    for imb in H.APPROVED_IMBALANCE:
        res, mean_auc, std = H.train_seeds(
            Xtr, ytr, Xdv, ydv,
            H.TrainConfig(head_type="binary", imbalance=imb, max_epochs=60),
            seeds=(17, 29, 43, 61, 79))
        p = H.seed_averaged_p_tb(res, Xdv)
        sens, _, npos, ntp = S.sensitivity_at_specificity(ydv, p, 0.70)
        lo, hi = S.wilson_ci(ntp, npos)
        results[imb] = {"res": res, "p": p, "auc": S.auc(ydv, p), "auc_mean": mean_auc,
                        "auc_std": std, "sens": sens, "ci": [lo, hi], "n_pos": npos}
        print("  %-18s dev AUC %.4f (seed sd %.4f) | sens@70spec %.4f [%.4f, %.4f]"
              % (imb, results[imb]["auc"], std, sens, lo, hi))

    best_imb = max(results, key=lambda k: (results[k]["sens"], results[k]["auc"]))
    print("  -> D10 would pick: %s" % best_imb)

    print("\n=== P2-13: registry + R2 reproduction (to 1e-4) ===")
    best = results[best_imb]["res"][0]
    wp = OUT / ("head_%s.npz" % best_imb)
    sha = R.save_head(best, wp)
    entry = R.RegistryEntry(
        model_id="radino_full_clahe0_bin", head_type="binary", variant="full_clahe0",
        backbone="microsoft/rad-dino@110cbc18", weight_path=str(wp),
        weight_sha256=sha, n_classes=2, is_primary=True)
    p_single = R.classify_embeddings(entry, Xdv)
    c08 = pd.DataFrame({"image_id": g_dv.image_id.tolist(),
                        "model_id": entry.model_id, "p_tb_raw": p_single})
    ok, maxdiff, n = R.reproduces_scores(entry, Xdv, g_dv.image_id.tolist(), c08)
    print("  R2 reproduces exported C08: %s (max diff %.2e over %d images)"
          % ("PASS" if ok else "FAIL", maxdiff, n))
    reg = R.Registry([entry]); reg.save(OUT / "registry.json")
    print("  registry SHA integrity: %s" % (reg.verify_shas() or "OK"))

    summary = {
        "provisional": True,
        "note": "provisional split, NOT Person 1's C01; no decision is locked from this",
        "seed": SEED,
        "split_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_comparison": {k: {"dev_auc": round(v["auc"], 4),
                                     "seed_auc_sd": round(v["auc_std"], 4),
                                     "sens_at_70spec": round(v["sens"], 4),
                                     "sens_ci": [round(x, 4) for x in v["ci"]],
                                     "n_pos": v["n_pos"]} for k, v in results.items()},
        "d10_would_pick": best_imb,
        "r2_reproduction_ok": bool(ok), "r2_max_diff": float(maxdiff),
        "internal_test": "created and left sealed; never read in this run",
    }
    json.dump(summary, open(OUT / "dryrun_summary.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % (OUT / "dryrun_summary.json"))


if __name__ == "__main__":
    sys.exit(main())
