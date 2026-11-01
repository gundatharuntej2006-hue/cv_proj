# -*- coding: utf-8 -*-
"""The two Section 14 rows Person 2 owes that the variant sweep does not cover:

  **kNN k for the OOD gate (D13).** The methodology fixes k = 5 with a k = 1 ablation. The
  comparison that matters is not a single false-flag rate at an arbitrary threshold but
  whether k changes the gate's *ability* to separate real chest X-rays from aliens, which is
  threshold-free: AUROC against the lateral and non-medical pools. The in-distribution
  false-flag rate at a matched alien-detection level is reported alongside, since that is
  the cost side of the same choice.

  **Binary versus three-class head (D11).** A three-class head over (healthy, sick_non_tb,
  tb) gives the dashboard an auxiliary output and might regularise the TB decision by
  forcing the model to separate sick-but-not-TB. Or it might spend capacity on a distinction
  nobody asked for. Both are trained identically and compared on the TB decision alone,
  which is the only thing the triage uses, with a paired bootstrap.

    py -3 scripts/p2_14_ablations.py

Writes artifacts/p2/dryrun/section14_ablations.json.
"""
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
from p2_gate import ood as O                                        # noqa: E402

EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT = ROOT / "artifacts/p2/dryrun/section14_ablations.json"
OOD_SETS = {"non_medical": ROOT / "artifacts/p2/emb/radino_ood_nonmedical_v1.npz",
            "lateral_cxr": ROOT / "artifacts/p2/emb/radino_ood_lateral_v1.npz"}
K_VALUES = (1, 3, 5, 10, 20)
SEEDS = (17, 29, 43, 61, 79)
SPEC_SELECT = 0.95
SPEC_REPORT = 0.70
BANK_N = 4000
TARGET_ALIEN_CATCH = 1.0        # the gate must catch every known alien; cost is the variable


def load():
    import pandas as pd
    with np.load(EMB) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    return ids, emb, idx, c01


def ablate_k(emb, idx, c01, log=print):
    """Does k change what the gate can separate, or only where the threshold sits?"""
    tr = c01[c01.split == "train"]
    dv = c01[c01.split == "development"]
    rng = np.random.default_rng(0)
    Xtr = emb[[idx[i] for i in tr.image_id]]
    sel = rng.choice(len(Xtr), min(BANK_N, len(Xtr)), replace=False)
    bank_emb = O.l2_normalise(Xtr[sel])
    Xdv = emb[[idx[i] for i in dv.image_id]]
    cat = dv.category.to_numpy()
    sick = np.isin(cat, ["sick_non_tb"])
    is_tb = np.array(["tb" in c and "non" not in c for c in cat])

    aliens = {}
    for name, p in OOD_SETS.items():
        if p.exists():
            with np.load(p) as z:
                aliens[name] = z["emb"].astype(np.float32)
    if not aliens:
        log("no alien pools embedded; skipping the k ablation")
        return []

    log("kNN k ablation for the OOD gate (bank %d, development %d):" % (len(bank_emb), len(Xdv)))
    log("  %-4s %-12s %-12s %-10s %s" % ("k", "AUROC alien", "thr@100%", "sick FF", "TB FF"))
    rows = []
    for k in K_VALUES:
        bank = O.OodBank(bank_emb, k=k)
        s_in = bank.score(Xdv)
        aur = {}
        for name, X in aliens.items():
            s_out = bank.score(X)
            y = np.r_[np.zeros(len(s_in), int), np.ones(len(s_out), int)]
            aur[name] = round(float(SEL.auc(y, np.r_[s_in, s_out])), 4)
        # the largest threshold that still catches every alien in every pool
        thr = min(float(bank.score(X).min()) for X in aliens.values())
        thr = float(np.nextafter(thr, -np.inf))
        ff_all = float((s_in > thr).mean())
        ff_sick = float((s_in[sick] > thr).mean())
        ff_tb = float((s_in[is_tb] > thr).mean())
        mean_auroc = float(np.mean(list(aur.values())))
        rows.append({"k": k, "auroc_per_pool": aur, "mean_auroc": round(mean_auroc, 4),
                     "threshold_at_full_catch": round(thr, 4),
                     "false_flag_all": round(ff_all, 4),
                     "false_flag_sick_non_tb": round(ff_sick, 4),
                     "false_flag_tb": round(ff_tb, 4)})
        log("  %-4d %-12.4f %-12.4f %9s %8s"
            % (k, mean_auroc, thr, "%.2f%%" % (100 * ff_sick), "%.2f%%" % (100 * ff_tb)))
    best = max(rows, key=lambda r: (r["mean_auroc"], -r["false_flag_sick_non_tb"]))
    log("  -> highest alien separability at k=%d (mean AUROC %.4f)"
        % (best["k"], best["mean_auroc"]))
    spread = max(r["mean_auroc"] for r in rows) - min(r["mean_auroc"] for r in rows)
    log("  AUROC spread across all k: %.4f%s"
        % (spread, "  (k barely matters)" if spread < 0.01 else ""))
    return rows


def ablate_head(emb, idx, c01, log=print):
    """Binary versus three-class, judged only on the TB decision the triage actually uses."""
    tr = c01[c01.split == "train"]
    dv = c01[c01.split == "development"]
    Xtr, Xdv = emb[[idx[i] for i in tr.image_id]], emb[[idx[i] for i in dv.image_id]]
    ydv_bin = H.binary_labels(dv.category.tolist())

    log("\nhead ablation (identical embeddings, seeds and early stopping):")
    out, probs = [], {}
    for head_type in ("binary", "three_class"):
        lab = H.binary_labels if head_type == "binary" else H.three_class_labels
        ytr, ydv = lab(tr.category.tolist()), lab(dv.category.tolist())
        res, mean_auc, sd = H.train_seeds(
            Xtr, ytr, Xdv, ydv,
            H.TrainConfig(head_type=head_type, imbalance="logit_adjusted", max_epochs=60),
            seeds=SEEDS)
        W = res[0].state_dict["weight"].numpy().astype(np.float32)
        b = res[0].state_dict["bias"].numpy().astype(np.float32)
        n_cls = 2 if head_type == "binary" else 3
        p = R.p_tb_from_logits(R.head_logits(Xdv, W, b), n_cls)
        probs[head_type] = p
        # re-measured on the binary TB label for both, so the two are comparable
        auc_tb = float(SEL.auc(ydv_bin, p))
        s95 = SEL.sensitivity_at_specificity(ydv_bin, p, SPEC_SELECT)
        s70 = SEL.sensitivity_at_specificity(ydv_bin, p, SPEC_REPORT)[0]
        lo, hi = SEL.wilson_ci(s95[3], s95[2])
        out.append({"head_type": head_type, "n_classes": n_cls,
                    "auc_on_tb_label": round(auc_tb, 4),
                    "reported_training_auc": round(float(mean_auc), 4),
                    "seed_sd": round(float(sd), 4),
                    "sens_at_95": round(float(s95[0]), 4),
                    "sens_at_95_ci": [round(lo, 4), round(hi, 4)],
                    "sens_at_70": round(float(s70), 4)})
        log("  %-12s AUC(TB) %.4f (seed sd %.4f) | sens@95 %.4f [%.4f, %.4f] | sens@70 %.4f"
            % (head_type, auc_tb, sd, s95[0], lo, hi, s70))

    rng = np.random.default_rng(0)
    y = np.asarray(ydv_bin).astype(int)
    d = []
    for _ in range(2000):
        kk = rng.integers(0, len(y), len(y))
        if len(set(y[kk].tolist())) < 2:
            continue
        d.append(SEL.auc(y[kk], probs["three_class"][kk]) - SEL.auc(y[kk], probs["binary"][kk]))
    d = np.asarray(d)
    delta = float(SEL.auc(y, probs["three_class"]) - SEL.auc(y, probs["binary"]))
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    verdict = "worse" if hi < 0 else "better" if lo > 0 else "no difference"
    log("  paired bootstrap, three_class - binary: %+.4f [%+.4f, %+.4f]  %s"
        % (delta, lo, hi, verdict))
    return {"heads": out, "three_class_minus_binary": {
        "delta_auc": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "verdict": verdict}}


def main():
    ids, emb, idx, c01 = load()
    k_rows = ablate_k(emb, idx, c01)
    head = ablate_head(emb, idx, c01)
    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "knn_k_ablation": k_rows, "head_ablation": head,
           "note": ("Provisional split, development only; internal_test untouched. "
                    "'thr@100%' is the largest OOD threshold still catching every image "
                    "in every alien pool; 'FF' is the false-flag rate there.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
