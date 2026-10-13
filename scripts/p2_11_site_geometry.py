# -*- coding: utf-8 -*-
"""Why the kNN OOD gate does not transfer between sites: the embedding geometry.

Reproduces the analysis behind the site-transfer section of docs/p2_ood_gate_finding.md.
Four questions, answered on the real embeddings:

  1. Is the TBX11K-vs-newsite difference a low-rank shift? (mean-difference direction)
  2. Does projecting that direction out fix the gate? (no) and does it hurt TB? (no)
  3. How tight is each site? (self kNN distance) -> TBX11K is ~2x tighter
  4. Does whitening by TBX11K statistics fix the gate? (no)

Conclusion: the gate fails at a new site because TBX11K is pathologically homogeneous,
not because of a removable shift. The classifier is unaffected (TB signal preserved under
every transform). The fix is local conformal threshold re-fitting, not embedding surgery.

    py -3 scripts/p2_11_site_geometry.py     # writes artifacts/p2/gate/p2_11_site_geometry.json

Needs the real embeddings (TBX11K + the CheXpert frontal control + the OOD sets).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "artifacts/p2/emb"


def load(name):
    with np.load(EMB / name) as z:
        return z["image_id"].tolist(), z["emb"].astype(np.float32)


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def auroc(pos, neg):
    import pandas as pd
    s = np.concatenate([pos, neg])
    r = pd.Series(s).rank().to_numpy()
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def knn(q, bank, k=5, bl=512, exclude_self=False):
    o = np.empty(len(q), np.float32)
    for s in range(0, len(q), bl):
        sim = q[s:s + bl] @ bank.T
        if exclude_self:
            for r in range(sim.shape[0]):
                sim[r, s + r] = -2.0
        o[s:s + bl] = 1 - np.partition(sim, -k, axis=1)[:, -k:].mean(1)
    return o


def probe_auc(tr, ytr, te, yte):
    from numpy.linalg import lstsq
    w, _, _, _ = lstsq(np.c_[tr, np.ones(len(tr))], ytr.astype(float), rcond=None)
    s = np.c_[te, np.ones(len(te))] @ w
    return auroc(s[yte == 1], s[yte == 0])


def project_out(V, d):
    return unit(V - np.outer(V @ d, d))


def main():
    import pandas as pd
    need = ["radino_tbx11k_full_clahe0_v1.npz", "radino_control_frontal_v1.npz",
            "radino_ood_lateral_v1.npz", "radino_ood_nonmedical_v1.npz"]
    if not all((EMB / n).exists() for n in need):
        raise SystemExit("missing embeddings; need %s" % need)

    ids_tb, tb = load(need[0])
    _, ctl = load(need[1]); _, lat = load(need[2]); _, nm = load(need[3])
    sp = pd.read_csv(ROOT / "artifacts/p2/dedup/provisional_official_splits.csv")
    split = dict(zip(sp.image_id, sp.split))
    tr_m = np.array([split.get(i) == "train" for i in ids_tb])
    va_m = np.array([split.get(i) == "val" for i in ids_tb])
    y = np.array([1 if i.startswith("tb") else 0 for i in ids_tb])
    TBtr, TBva, CX, LAT, NM = unit(tb[tr_m]), unit(tb[va_m]), unit(ctl), unit(lat), unit(nm)
    ytr, yva = y[tr_m], y[va_m]

    out = {"note": "why the kNN OOD gate does not transfer between sites"}

    # 1. mean-difference (site) direction
    sd = CX.mean(0) - TBtr.mean(0); sd /= np.linalg.norm(sd)
    out["site_separability_1dir_auroc"] = auroc(CX @ sd, TBtr @ sd)

    # 2. project it out
    TBtr2, TBva2, CX2, LAT2, NM2 = [project_out(V, sd) for V in (TBtr, TBva, CX, LAT, NM)]

    def gate(bank, held, cx, lat, nm):
        t = float(np.quantile(knn(held, bank), 0.95))
        return {"chexpert_false_rejection": float((knn(cx, bank) > t).mean()),
                "lateral_caught": float((knn(lat, bank) > t).mean()),
                "non_medical_caught": float((knn(nm, bank) > t).mean())}

    out["gate_raw"] = gate(TBtr, TBva, CX, LAT, NM)
    out["gate_after_mean_removal"] = gate(TBtr2, TBva2, CX2, LAT2, NM2)
    out["tb_probe_auc_raw"] = probe_auc(TBtr, ytr, TBva, yva)
    out["tb_probe_auc_after_mean_removal"] = probe_auc(TBtr2, ytr, TBva2, yva)

    # 3. density: self kNN distance
    out["self_dist_tbx11k"] = float(np.median(knn(TBtr, TBtr, exclude_self=True)))
    out["self_dist_chexpert"] = float(np.median(knn(CX, CX, exclude_self=True)))

    # 4. whitening by TBX11K stats
    mu, sg = TBtr.mean(0), TBtr.std(0) + 1e-6
    wh = lambda V: unit((V - mu) / sg)
    TBtrW, TBvaW, CXW, LATW, NMW = [wh(V) for V in (TBtr, TBva, CX, LAT, NM)]
    out["gate_after_whitening"] = gate(TBtrW, TBvaW, CXW, LATW, NMW)
    out["tb_probe_auc_after_whitening"] = probe_auc(TBtrW, ytr, TBvaW, yva)

    p = ROOT / "artifacts/p2/gate/p2_11_site_geometry.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    print("\nwrote", p)


if __name__ == "__main__":
    sys.exit(main())
