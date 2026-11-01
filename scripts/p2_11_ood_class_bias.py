# -*- coding: utf-8 -*-
"""Does the OOD gate reject TB cases more often than healthy ones?

Noticed by accident: a real TBX11K TB image, served over the HTTP API, was rejected as
out-of-distribution (score 0.5781 against a threshold of 0.5746). At alpha = 0.05 some
in-distribution rejection is expected and correct — that is what the conformal guarantee
buys. The question this script answers is whether those rejections fall **evenly** across
classes.

If they do not, the gate is a safety problem rather than a safety feature: a gate that
preferentially rejects TB cases removes exactly the images the system exists to find, and
because a rejected case carries no probability, the miss is invisible in every accuracy
metric. Sensitivity would be measured only over the cases the gate let through.

    py -3 scripts/p2_11_ood_class_bias.py

Reports the flag rate per class with Wilson intervals, on development (where the threshold
was fitted) and on the untouched classes of the training split.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import selection as SEL                                 # noqa: E402
from p2_gate import ood as O                                        # noqa: E402

BUNDLE = ROOT / "artifacts/p2/bundle"
EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT = ROOT / "artifacts/p2/dryrun/ood_class_bias.json"


OOD_SETS = {
    "non_medical": ROOT / "artifacts/p2/emb/radino_ood_nonmedical_v1.npz",
    "lateral_cxr": ROOT / "artifacts/p2/emb/radino_ood_lateral_v1.npz",
}


def threshold_curve(bank, emb, idx, c01, fitted_thr):
    """What the gate is actually trading off.

    The gate exists to reject inputs that are not a frontal chest X-ray — a photograph, a
    lateral view. It is not supposed to reject *abnormal* chest X-rays, but the measurement
    above shows it does, because disease makes an image sit further from its neighbours.

    So the honest way to choose the threshold is a curve: at each candidate, how many
    genuine aliens are still caught, and how many sick frontals are wrongly thrown away?
    """
    sets = {}
    for name, p in OOD_SETS.items():
        if not p.exists():
            continue
        with np.load(p) as z:
            sets[name] = bank.score(z["emb"].astype(np.float32))
    dev = c01[c01.split == "development"]
    s_dev = bank.score(emb[[idx[i] for i in dev.image_id]])
    cat = dev.category.to_numpy()
    sick = s_dev[np.isin(cat, ["sick_non_tb"])]
    tb = s_dev[np.array(["tb" in c and "non" not in c for c in cat])]
    healthy = s_dev[cat == "healthy"]
    if not sets:
        print("\n(no OOD pools embedded; skipping the operating curve)")
        return None

    print("\noperating curve — the gate should reject aliens, not abnormal X-rays:")
    header = "  %-9s" % "threshold" + "".join("%14s" % k for k in sets) + \
             "%12s%12s%12s" % ("sick_non_tb", "TB", "healthy")
    print(header)
    rows = []
    cands = sorted(set([fitted_thr] + list(np.round(np.arange(0.55, 0.85, 0.025), 3))))
    for t in cands:
        r = {"threshold": round(float(t), 4),
             "caught": {k: round(float((v > t).mean()), 4) for k, v in sets.items()},
             "wrongly_rejected": {
                 "sick_non_tb": round(float((sick > t).mean()), 4),
                 "tb": round(float((tb > t).mean()), 4),
                 "healthy": round(float((healthy > t).mean()), 4)},
             "is_fitted": bool(abs(t - fitted_thr) < 1e-9)}
        rows.append(r)
        print("  %-9.4f%s%s%12.2f%%%12.2f%%%12.2f%%"
              % (t, "".join("%13.1f%%" % (100 * r["caught"][k]) for k in sets),
                 "  <- fitted" if r["is_fitted"] else "",
                 100 * r["wrongly_rejected"]["sick_non_tb"],
                 100 * r["wrongly_rejected"]["tb"],
                 100 * r["wrongly_rejected"]["healthy"]))
    return rows


def main():
    import pandas as pd
    meta = json.load(open(BUNDLE / "bundle.json", encoding="utf-8"))
    thr = meta.get("ood_threshold")
    if thr is None:
        raise SystemExit("this bundle's OOD threshold abstained; nothing to measure")
    with np.load(BUNDLE / "bank.npz") as z:
        bank = O.OodBank(z["bank"].astype(np.float32), k=meta.get("ood_k", 5))
    with np.load(EMB) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]

    print("OOD threshold %.4f (conformal, alpha=%s), bank n=%d, k=%d"
          % (thr, meta.get("ood_alpha"), len(bank.bank), bank.k))

    rows = []
    for split in ("train", "development", "calibration", "internal_test"):
        g = c01[c01.split == split]
        if g.empty:
            continue
        if split == "internal_test":
            # the held-out split is read once, under a seal; not here
            print("\n%s: skipped (sealed)" % split)
            continue
        print("\n%s (n=%d):" % (split, len(g)))
        scores = bank.score(emb[[idx[i] for i in g.image_id]])
        for cat in sorted(set(g.category)):
            m = (g.category == cat).to_numpy()
            flagged = int((scores[m] > thr).sum())
            n = int(m.sum())
            lo, hi = SEL.wilson_ci(flagged, n)
            rows.append({"split": split, "category": cat, "n": n, "flagged": flagged,
                         "rate": round(flagged / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
                         "median_score": round(float(np.median(scores[m])), 4)})
            print("  %-16s %4d images | flagged %3d (%5.2f%%) [%.2f%%, %.2f%%] | median score %.4f"
                  % (cat, n, flagged, 100 * flagged / n, 100 * lo, 100 * hi,
                     np.median(scores[m])))

    # the comparison that matters: TB versus everything else, on development
    dev = [r for r in rows if r["split"] == "development"]
    tb = [r for r in dev if "tb" in r["category"].lower() and "non" not in r["category"].lower()]
    other = [r for r in dev if r not in tb]
    verdict = None
    if tb and other:
        k_tb = sum(r["flagged"] for r in tb)
        n_tb = sum(r["n"] for r in tb)
        k_ot = sum(r["flagged"] for r in other)
        n_ot = sum(r["n"] for r in other)
        lo_tb, hi_tb = SEL.wilson_ci(k_tb, n_tb)
        lo_ot, hi_ot = SEL.wilson_ci(k_ot, n_ot)
        overlap = not (hi_tb < lo_ot or hi_ot < lo_tb)
        verdict = {"tb": {"n": n_tb, "flagged": k_tb, "rate": round(k_tb / n_tb, 4),
                          "ci95": [round(lo_tb, 4), round(hi_tb, 4)]},
                   "non_tb": {"n": n_ot, "flagged": k_ot, "rate": round(k_ot / n_ot, 4),
                              "ci95": [round(lo_ot, 4), round(hi_ot, 4)]},
                   "intervals_overlap": bool(overlap),
                   "ratio": round((k_tb / n_tb) / max(k_ot / n_ot, 1e-9), 2)}
        print("\nTB vs non-TB on development:")
        print("  TB      %d/%d = %.2f%% [%.2f%%, %.2f%%]"
              % (k_tb, n_tb, 100 * k_tb / n_tb, 100 * lo_tb, 100 * hi_tb))
        print("  non-TB  %d/%d = %.2f%% [%.2f%%, %.2f%%]"
              % (k_ot, n_ot, 100 * k_ot / n_ot, 100 * lo_ot, 100 * hi_ot))
        print("  -> %s (%.2fx)" % ("indistinguishable" if overlap
                                   else "DIFFERENT - the gate is class-biased",
                                   verdict["ratio"]))

    curve = threshold_curve(bank, emb, idx, c01, thr)

    out = {"created": time.strftime("%Y-%m-%d"), "threshold": thr,
           "operating_curve": curve,
           "alpha": meta.get("ood_alpha"), "k": bank.k, "bank_n": int(len(bank.bank)),
           "by_class": rows, "tb_vs_non_tb_development": verdict,
           "note": ("A rejected case carries no probability, so a class-biased gate hides "
                    "misses from every accuracy metric. internal_test not read.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
