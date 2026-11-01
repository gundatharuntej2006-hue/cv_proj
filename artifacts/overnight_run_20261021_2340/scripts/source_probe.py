# -*- coding: utf-8 -*-
"""Priority 2: diagnostic source-prediction probe, plus scanner/class association tests.

Leakage control, stated explicitly:
  * The probe is fitted ONLY on the Vietnam TUNE half (1,352 cases), whose documented role
    is development. The REPORT half is not read by this script at all.
  * The probe predicts rad_system, not tuberculosis. Nothing it produces is fed back into
    any feature, classifier or threshold.
  * Vietnam is one image per patient, so patient-grouped CV and ordinary stratified CV
    coincide; this is asserted rather than assumed.

Inputs are the globally pooled frozen embeddings (2,708 x 768) -- small enough to hold in
the roughly 0.6 GB of RAM available during this run.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SEED = 17


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from scipy.stats import chi2_contingency

    meta = list(csv.DictReader(io.open(
        ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv", encoding="utf-8-sig")))
    split = json.load(io.open(
        ROOT / "artifacts/p2/external/vietnam_tune_report_split.json", encoding="utf-8"))
    tune = set(str(x) for x in split["tune"])

    z = np.load(ROOT / "artifacts/p2/external/radino_vietnam_lungs_only_v1.npz")
    ids = [str(x) for x in z["image_id"]]
    emb = np.asarray(z["emb"], dtype=np.float32)
    assert emb.shape == (2708, 768), emb.shape

    by_pid = {str(r["pid"]): r for r in meta}
    assert set(ids) == set(by_pid), "embedding ids and Dryad pids do not match"
    assert len(set(ids)) == len(ids), "duplicate embedding ids"

    rows = [by_pid[i] for i in ids]
    scanner = np.array([int(r["rad_system"]) for r in rows])
    tb = np.array([int(r["xpert_result"]) for r in rows])
    in_tune = np.array([i in tune for i in ids])

    out = {"n_total": len(ids), "n_tune": int(in_tune.sum()),
           "report_half_read": False,
           "one_image_per_patient": bool(len(set(ids)) == len(ids))}

    # ---- association between scanner and TB status, on the full cohort (descriptive only)
    tab = np.array([[int(((scanner == s) & (tb == t)).sum()) for t in (0, 1)] for s in (1, 2, 3)])
    chi2, p, dof, _ = chi2_contingency(tab)
    n = tab.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
    out["scanner_by_tb"] = {
        "table_rows_scanner_1_2_3_cols_tb0_tb1": tab.tolist(),
        "chi2": round(float(chi2), 4), "dof": int(dof), "p_value": round(float(p), 4),
        "cramers_v": round(cramers_v, 4),
        "prevalence_by_scanner": {str(s): round(float(tb[scanner == s].mean()), 4)
                                  for s in (1, 2, 3)},
        "interpretation": "descriptive association only; it does not establish confounding "
                          "or its absence",
    }

    # ---- the probe itself, TUNE only
    X, y = emb[in_tune], scanner[in_tune]
    counts = {str(s): int((y == s).sum()) for s in (1, 2, 3)}
    majority = max(counts.values()) / len(y)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=2000,
                                           random_state=SEED))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    bal = balanced_accuracy_score(y, pred)
    mf1 = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred, labels=[1, 2, 3])

    # chance reference: the same CV on shuffled scanner labels
    rng = np.random.default_rng(SEED)
    chance = []
    for _ in range(10):
        ys = rng.permutation(y)
        pp = np.empty_like(ys)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, ys):
            clf.fit(X[tr], ys[tr])
            pp[te] = clf.predict(X[te])
        chance.append(balanced_accuracy_score(ys, pp))

    out["probe"] = {
        "fitted_on": "Vietnam TUNE half only",
        "n": int(len(y)), "class_counts": counts,
        "majority_class_accuracy": round(float(majority), 4),
        "balanced_accuracy": round(float(bal), 4),
        "macro_f1": round(float(mf1), 4),
        "balanced_accuracy_under_label_permutation_mean": round(float(np.mean(chance)), 4),
        "balanced_accuracy_under_label_permutation_max": round(float(np.max(chance)), 4),
        "confusion_rows_true_1_2_3_cols_pred_1_2_3": cm.tolist(),
        "classifier": "StandardScaler + multinomial LogisticRegression(C=0.1)",
        "cv": "5-fold stratified; one image per patient so this is patient-grouped",
        "seed": SEED,
    }

    (RUN / "source_probe_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
