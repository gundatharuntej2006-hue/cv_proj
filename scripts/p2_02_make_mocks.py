# -*- coding: utf-8 -*-
"""P2-02: mock generators for the five contracts Person 2 produces.

    C03  data/splits/ood_heldout_v1.csv
    C07  artifacts/p2/emb/radino_{dataset}_{variant}_v1.npz
    C08  artifacts/p2/scores/scores_{dataset}_{split}_v1.parquet
    C09  artifacts/p2/gate/gate_{dataset}_{split}_v1.parquet
    C17  artifacts/p2/dedup/near_dup_candidates_v1.csv

Everything lands under mock/ at the same relative path (contract rule 4.1) and is
deterministic given --seed.

Three things make these mocks worth more than random noise, because Persons 3 and 4
build against them for five weeks before real scores exist:

1. The scores have a known right answer. Latent ability s is drawn N(+1,1) for TB and
   N(-1,1) for the rest, so the true log-odds are exactly 2s + log(pi/(1-pi)). The
   emitted logit is that multiplied by T_TRUE = 1.7. Temperature scaling on the mock
   development split must therefore return about 1.7 and a near-flat reliability
   diagram, so P4-11 has an answer key instead of a vibe check.
2. The mocks are consistent with each other. C17 is not invented; it is a real
   all-pairs cosine search over the C07 mock embeddings, into which duplicate pairs
   were planted. C09's threshold is the real 95th percentile of the mock development
   scores, so the D13 acceptance criterion holds on the mock itself.
3. They contain the awkward cases on purpose: a train/internal_test duplicate pair (the
   H1 leak that dedup exists to catch), a pair sitting at exactly tau to force P1 to
   define >= against >, a full-image model row with a null margin, quality failures
   carrying reasons, and an OOD set that mostly fails the gate.

The OOD sources in the C03 mock are deliberately fake names under a fake licence, so a
mock row can never be mistaken for a real source or pasted into the committed file.
Real sources and their licences are chosen in P2-10 and recorded in D15 (W5).
"""
import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_cls import contracts as K                                    # noqa: E402
from p2_cls._paths import REPO_ROOT                                  # noqa: E402

T_TRUE = 1.7          # the overconfidence baked in; P4-11 should recover this
MU = 1.0              # latent separation; AUC by construction = Phi(2 MU / sqrt(2))
EMB_VARIANTS = ["crop_m0.10_clahe0", "full_clahe0"]
MODEL_REV = "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936"
SPLIT_ORDER = {"train": 0, "development": 1, "calibration": 2, "internal_test": 3}


def out_path(rel):
    p = REPO_ROOT / "mock" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- C01 (read or fake)
def splits_table(rng, n):
    """Use Person 1's mock C01 when it exists, so joins line up. Otherwise synthesise
    one with TBX11K's shape: about 45% healthy, 45% sick non-TB, 9% TB, the TB images
    split active / latent / both as the real training set is (473 / 104 / 23)."""
    c01 = REPO_ROOT / "mock" / "data" / "splits" / "tbx11k_splits_v1.csv"
    if c01.exists():
        df = pd.read_csv(c01)
        print("  using Person 1's mock C01: %d rows from %s" % (len(df), c01))
        return df, True

    frac = {"train": 0.55, "development": 0.10, "calibration": 0.175, "internal_test": 0.175}
    rows = []
    count = {"h": 0, "s": 0, "tb": 0}
    for split, f in frac.items():
        k = int(round(n * f))
        cats = rng.choice(["healthy", "sick_non_tb", "tb"], size=k, p=[0.455, 0.455, 0.09])
        for c in cats:
            if c == "tb":
                sub = rng.choice(["active_tb", "latent_tb", "active_latent_tb"],
                                 p=[0.788, 0.173, 0.039])
                count["tb"] += 1
                iid = "tb%04d" % count["tb"]
                tb_label = 1
                active = 0 if sub == "latent_tb" else 1
                cat = sub
                n_boxes = 0 if sub == "latent_tb" else int(rng.integers(1, 4))
                folder = "tb"
            elif c == "healthy":
                count["h"] += 1
                iid, tb_label, active, cat, n_boxes, folder = (
                    "h%04d" % count["h"], 0, 0, "healthy", 0, "health")
            else:
                count["s"] += 1
                iid, tb_label, active, cat, n_boxes, folder = (
                    "s%04d" % count["s"], 0, 0, "sick_non_tb", 0, "sick")
            rows.append({
                "image_id": iid,
                "rel_path": "imgs/%s/%s.png" % (folder, iid),
                "official_split": "train" if split in ("train", "development") else "val",
                "split": split,
                "category": cat,
                "tb_label": tb_label,
                "active_tb_label": (pd.NA if cat == "latent_tb" else active),
                "has_boxes": n_boxes > 0,
                "n_boxes": n_boxes,
                "dup_of": pd.NA,
                "split_version": "v1",
                "seed": 20260929,
            })
    df = pd.DataFrame(rows)
    print("  synthesised a stand-in C01: %d rows (%d TB, %.1f%%)"
          % (len(df), int(df.tb_label.sum()), 100 * df.tb_label.mean()))
    return df, False


# ---------------------------------------------------------------- C07
def make_embeddings(rng, df, n_dup_pairs=4):
    """Unit-norm 768-d vectors whose direction depends on the label, so both a kNN OOD
    gate and a cosine dedup search behave sensibly on them. Returns the planted pairs."""
    n = len(df)
    tb = df.tb_label.to_numpy() == 1
    axis = rng.normal(size=K.EMB_DIM)
    axis /= np.linalg.norm(axis)
    base = rng.normal(size=(n, K.EMB_DIM)) * 0.35
    base += np.where(tb, 0.55, -0.15)[:, None] * axis          # class-dependent direction
    base += axis * 0.9                                         # a shared "chest X-ray" direction

    ids = df.image_id.to_numpy()
    splits = df.split.to_numpy()
    planted = []
    train_idx = np.flatnonzero(splits == "train")
    targets = ["internal_test", "calibration", "development", "internal_test"]
    # pair 0 sits at tau itself, so P1 has to decide >= against >; the rest spread above it
    wanted = [K.TAU_DEFAULT, 0.9999, 0.995, 0.988]
    for i in range(n_dup_pairs):
        later = np.flatnonzero(splits == targets[i % len(targets)])
        if len(train_idx) == 0 or len(later) == 0:
            continue
        a, b = int(rng.choice(train_idx)), int(rng.choice(later))
        base[b] = _at_cosine(base[a], wanted[i % len(wanted)], rng)
        planted.append((ids[a], ids[b]))

    emb = base / np.linalg.norm(base, axis=1, keepdims=True)
    return emb.astype(np.float16), planted


def _at_cosine(a, target, rng):
    """Return a vector whose cosine to `a` is `target`, then nudge it up until float16
    storage cannot push it below the target. Exact construction beats random wobble:
    a boundary case is only a boundary case if it lands exactly on the boundary."""
    v = rng.normal(size=a.shape)
    v -= a * float(v @ a) / float(a @ a)                 # component orthogonal to a
    v /= np.linalg.norm(v)
    for bump in range(12):
        t = math.sqrt(max(1.0 / (target * (1 + 3e-5 * bump)) ** 2 - 1.0, 0.0))
        b = a + v * t * float(np.linalg.norm(a))
        a16 = (a / np.linalg.norm(a)).astype(np.float16).astype(np.float32)
        b16 = (b / np.linalg.norm(b)).astype(np.float16).astype(np.float32)
        if float(a16 @ b16 / (np.linalg.norm(a16) * np.linalg.norm(b16))) >= target:
            return b
    return b


def write_c07(emb, df, variant, dataset="tbx11k"):
    p = out_path("artifacts/p2/emb/radino_%s_%s_v1.npz" % (dataset, variant))
    np.savez_compressed(
        p,
        image_id=df.image_id.to_numpy().astype("<U32"),
        emb=emb,
        l2_normalised=np.array(True),
        variant=np.array(variant),
        model_id=np.array(MODEL_REV),
        preproc_version=np.array(K.SCHEMA_VERSION),
    )
    return p


# ---------------------------------------------------------------- C17
def write_c17(emb, df, tau, planted):
    """Not a second implementation: the mock calls the same search P2-04 runs for real
    (src/p2_gate/dedup.py), so the mock cannot drift from the production behaviour."""
    from p2_gate.dedup import find_near_duplicates

    out, within = find_near_duplicates(df.image_id.to_numpy(), df.split.to_numpy(),
                                       emb.astype(np.float32), tau, inclusive=True)
    p = out_path("artifacts/p2/dedup/near_dup_candidates_v1.csv")
    out.to_csv(p, index=False)
    found = set()
    for r in out.itertuples():
        found.add((r.image_id_a, r.image_id_b))
        found.add((r.image_id_b, r.image_id_a))
    missed = [q for q in planted if q not in found]
    return p, out, missed


# ---------------------------------------------------------------- C08
def fit_temperature(logits, y, lo=0.2, hi=5.0, iters=80):
    """Ternary search on the NLL. Only used to record what the mock actually contains;
    the real temperature scaling is P4's (P4-11, D22) and must not import this."""
    def nll(T):
        p = np.clip(1.0 / (1.0 + np.exp(-logits / T)), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    for _ in range(iters):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def auc(y, p):
    """Mann-Whitney U with ties averaged."""
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def write_c08(rng, df, splits=("development", "calibration", "internal_test")):
    written, aucs, priors, recovered = [], {}, {}, {}
    for split in splits:
        d = df[df.split == split].reset_index(drop=True)
        if len(d) == 0:
            continue
        y = d.tb_label.to_numpy()
        # each split's OWN prevalence: temperature scaling has no bias term, so if the
        # intercept is wrong for this split the fitted temperature silently absorbs the
        # error and the answer key stops being an answer key
        prior = float(y.mean())
        log_prior_odds = float(np.log(prior / (1 - prior)))
        priors[split] = round(log_prior_odds, 4)
        s_img = rng.normal(loc=np.where(y == 1, MU, -MU), scale=1.0)   # per-image ability
        recs = []
        for m in K.REGISTRY:
            if m["model_id"].startswith("effb3"):
                s = 0.80 * s_img + rng.normal(scale=0.50, size=len(d))  # a weaker baseline
                sep, var = 0.80 * MU, 0.80 ** 2 + 0.50 ** 2
            else:
                s = s_img + rng.normal(scale=0.35, size=len(d))
                sep, var = MU, 1.0 + 0.35 ** 2
            # For two equal-variance Gaussians at +-sep the exact posterior log-odds are
            # 2 sep s / var + log prior odds. The per-model noise widens var, so using the
            # naive 2 MU s here would make the answer key wrong: the recovered temperature
            # would come out near 1.96 rather than T_TRUE.
            true_logit = 2.0 * sep * s / var + log_prior_odds
            if m["head"] == "binary":
                logit = T_TRUE * true_logit                             # deliberately overconfident
                p = 1.0 / (1.0 + np.exp(-logit))
                logits_3c = [None] * len(d)
                p3 = (None, None, None)
            else:
                cat = d.category.to_numpy()
                heal = np.where(cat == "healthy", 0.9, -0.6) + rng.normal(scale=0.4, size=len(d))
                sick = np.where(cat == "sick_non_tb", 0.9, -0.6) + rng.normal(scale=0.4, size=len(d))
                lg = np.stack([heal, sick, T_TRUE * true_logit], axis=1).astype(np.float32)
                e = np.exp(lg - lg.max(axis=1, keepdims=True))
                sm = e / e.sum(axis=1, keepdims=True)
                logit = None
                p = sm[:, 2]
                logits_3c = [[float(v) for v in row] for row in lg]
                p3 = (sm[:, 0], sm[:, 1], sm[:, 2])
            for i in range(len(d)):
                recs.append({
                    "image_id": str(d.image_id[i]), "dataset": "tbx11k", "split": split,
                    "model_id": m["model_id"], "head": m["head"], "variant": m["variant"],
                    "clahe": m["clahe"], "margin": m["margin"],
                    "logit_tb": None if logit is None else float(logit[i]),
                    "logits_3c": logits_3c[i],
                    "p_tb_raw": float(p[i]),
                    "p3_healthy": None if p3[0] is None else float(p3[0][i]),
                    "p3_sick_non_tb": None if p3[1] is None else float(p3[1][i]),
                    "p3_tb": None if p3[2] is None else float(p3[2][i]),
                    "is_primary": m["is_primary"],
                    "model_sha": "mock%012x" % (abs(hash(m["model_id"])) % 10 ** 12),
                    "seed": 17,
                })
            if m["is_primary"]:
                aucs[split] = auc(y, p)
        for m in K.REGISTRY:                     # what the file actually contains
            if m["head"] != "binary":
                continue
            zz = np.array([r["logit_tb"] for r in recs if r["model_id"] == m["model_id"]])
            recovered.setdefault(split, {})[m["model_id"]] = round(
                fit_temperature(zz, y.astype(float)), 4)
        table = pa.Table.from_pylist(recs, schema=K.C08_SCHEMA)
        p_out = out_path("artifacts/p2/scores/scores_tbx11k_%s_v1.parquet" % split)
        pq.write_table(table, p_out)
        written.append((p_out, len(recs)))
    return written, aucs, priors, recovered


# ---------------------------------------------------------------- C09
def write_c09(rng, df, ood, splits=("development", "calibration")):
    """The threshold is the real 95th percentile of the mock development scores, so the
    D13 criterion (development pass rate 95.0 +- 0.2 %) holds on the mock itself."""
    dev = df[df.split == "development"]
    dev_k5 = rng.gamma(4.0, 0.038, size=len(dev))
    # D13 wants a development pass rate of 95.0 +- 0.2 %. An interpolated quantile does not
    # give that: take the k-th order statistic so exactly floor(0.95 n) images pass with <=.
    k = int(np.floor(0.95 * len(dev_k5)))
    thr = float(np.sort(dev_k5)[k - 1])
    written = []

    def block(ids, dataset, role, k5, quality_fail_rate):
        n = len(ids)
        k1 = np.clip(k5 * rng.uniform(0.80, 0.95, size=n), 0, None)
        qp = rng.random(n) >= quality_fail_rate
        reasons = np.where(qp, "", rng.choice(K.C09_QUALITY_REASONS, size=n))
        op = k5 <= thr
        return [{"image_id": str(ids[i]), "dataset": dataset, "split_or_role": role,
                 "quality_pass": bool(qp[i]), "quality_reasons": str(reasons[i]),
                 "ood_score_k5": float(k5[i]), "ood_score_k1": float(k1[i]),
                 "ood_threshold_k5": thr, "ood_pass": bool(op[i]),
                 "gate_pass": bool(qp[i] and op[i])} for i in range(n)]

    for split in splits:
        d = df[df.split == split]
        k5 = dev_k5 if split == "development" else rng.gamma(4.0, 0.038, size=len(d))
        recs = block(d.image_id.to_numpy(), "tbx11k", split, k5, 0.008)   # <= 1% quality fails
        p_out = out_path("artifacts/p2/gate/gate_tbx11k_%s_v1.parquet" % split)
        pq.write_table(pa.Table.from_pylist(recs, schema=K.C09_SCHEMA), p_out)
        written.append((p_out, len(recs), sum(r["gate_pass"] for r in recs) / len(recs)))
        if split == "development":
            dev_ood_rate = sum(r["ood_pass"] for r in recs) / len(recs)

    oe = ood[ood.role == "ood_eval"]
    k5 = rng.gamma(9.0, 0.050, size=len(oe))            # clearly further from the training bank
    recs = block(oe.image_id.to_numpy(), "ood", "ood_eval", k5, 0.02)
    p_out = out_path("artifacts/p2/gate/gate_ood_ood_eval_v1.parquet")
    pq.write_table(pa.Table.from_pylist(recs, schema=K.C09_SCHEMA), p_out)
    written.append((p_out, len(recs), sum(r["gate_pass"] for r in recs) / len(recs)))
    return written, thr, dev_ood_rate


# ---------------------------------------------------------------- C03
def write_c03(rng, n_eval=60, n_view=(80, 20, 20)):
    """Fake source names under a fake licence on purpose: see the module docstring."""
    rows, c = [], 0
    for ood_type in K.C03_OOD_TYPES:
        for _ in range(n_eval // 3):
            c += 1
            rows.append({"image_id": "ood_%s_%04d" % (ood_type[:3], c),
                         "rel_path": "mock_%s/%04d.png" % (ood_type, c),
                         "source": "mock_%s_source" % ood_type,
                         "ood_type": ood_type, "role": "ood_eval",
                         "licence": "MOCK-NOT-A-LICENCE"})
    for role, k in zip(["viewhead_train", "viewhead_dev", "viewhead_test"], n_view):
        for _ in range(k):
            c += 1
            rows.append({"image_id": "ood_lat_%04d" % c,
                         "rel_path": "mock_lateral/%04d.png" % c,
                         "source": "mock_lateral_source", "ood_type": "lateral_cxr",
                         "role": role, "licence": "MOCK-NOT-A-LICENCE"})
    df = pd.DataFrame(rows, columns=K.C03_COLUMNS)
    p = out_path("data/splits/ood_heldout_v1.csv")
    df.to_csv(p, index=False)
    return p, df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="P2-02 mock generators (C03, C07, C08, C09, C17)")
    ap.add_argument("--n", type=int, default=8400,
                    help="stand-in TBX11K images when Person 1's C01 mock is absent; the default matches the real train+development+calibration+internal_test total")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--tau", type=float, default=K.TAU_DEFAULT,
                    help="D06 near-duplicate cosine threshold")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    man = {"task": "P2-02", "generated": str(date.today()), "seed": a.seed,
           "schema_version": K.SCHEMA_VERSION, "files": {}}

    print("C01 (input)")
    df, from_p1 = splits_table(rng, a.n)
    man["c01_source"] = "p1_mock" if from_p1 else "synthesised_by_p2_until_p1_ships"

    print("C03  held-out OOD collection")
    p03, ood = write_c03(rng)
    print("  %s  (%d rows, %d roles)" % (p03, len(ood), ood.role.nunique()))
    man["files"]["C03"] = {"path": str(p03.relative_to(REPO_ROOT)), "rows": len(ood)}

    print("C07  RAD-DINO embeddings")
    emb, planted = make_embeddings(rng, df)
    for v in EMB_VARIANTS:
        if v == "full_clahe0":
            e = emb
        else:
            e = emb.astype(np.float32) + rng.normal(scale=0.02, size=emb.shape).astype(np.float32)
            e = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16)
        p = write_c07(e, df, v)
        print("  %s  [%d, %d] float16" % (p.name, len(df), K.EMB_DIM))
        man["files"].setdefault("C07", []).append(
            {"path": str(p.relative_to(REPO_ROOT)), "rows": len(df), "variant": v})

    print("C17  near-duplicate candidates (real cosine search over the C07 mock)")
    p17, c17, missed = write_c17(emb, df, a.tau, planted)
    print("  %s  (%d pairs >= tau %.2f; planted %d, recovered %d)"
          % (p17, len(c17), a.tau, len(planted), len(planted) - len(missed)))
    if len(c17):
        print("  splits involved: %s, cosine %.4f to %.4f"
              % (sorted(set(c17.split_a) | set(c17.split_b)), c17.cosine.min(), c17.cosine.max()))
    man["files"]["C17"] = {"path": str(p17.relative_to(REPO_ROOT)), "rows": len(c17),
                           "tau": a.tau, "planted": len(planted), "missed": len(missed)}

    print("C08  classifier scores")
    w08, aucs, priors, recovered = write_c08(rng, df)
    for p, n in w08:
        print("  %s  (%d rows = %d models)" % (p.name, n, len(K.REGISTRY)))
    man["files"]["C08"] = [{"path": str(p.relative_to(REPO_ROOT)), "rows": n} for p, n in w08]
    man["known_answers"] = {
        "true_temperature": T_TRUE,
        "recovered_temperature": recovered,
        "note": ("Scores are T_TRUE times the exact posterior log-odds for that split, so "
                 "temperature scaling should return roughly T_TRUE. It will not return it "
                 "exactly: each split is a finite draw, so the value that actually minimises "
                 "the NLL on these files drifts by up to about 0.2 (all models within one "
                 "split agree closely, which is the tell that the drift is the split's draw "
                 "and not the model). Assert against recovered_temperature, which is what "
                 "these files contain, and use true_temperature only to explain the design."),
        "log_prior_odds_by_split": priors,
        "primary_model_auc": dict((k, round(v, 4)) for k, v in aucs.items()),
        # AUC = Phi(separation / (sigma sqrt 2)); the per-model noise widens sigma, so the
        # naive Phi(2 MU / sqrt 2) would overstate it
        "auc_expected_radino": round(_norm_cdf(2 * MU / (math.sqrt(1 + 0.35 ** 2) * math.sqrt(2))), 4),
        "auc_expected_effb3": round(_norm_cdf(1.6 * MU / (math.sqrt(0.8 ** 2 + 0.5 ** 2) * math.sqrt(2))), 4),
    }

    print("C09  gate scores")
    w09, thr, dev_ood_rate = write_c09(rng, df, ood)
    for p, n, gp in w09:
        print("  %s  (%d rows, gate_pass %.1f%%)" % (p.name, n, 100 * gp))
    man["files"]["C09"] = [{"path": str(p.relative_to(REPO_ROOT)), "rows": n,
                            "gate_pass_rate": round(gp, 4)} for p, n, gp in w09]
    man["known_answers"]["ood_threshold_k5"] = round(thr, 6)
    man["known_answers"]["development_ood_pass_rate"] = round(dev_ood_rate, 4)

    mp = REPO_ROOT / "mock" / "mock_manifest_p2.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(mp, "w", encoding="utf-8"), indent=1)
    print("\nmanifest: %s" % mp)
    print("Read these with TB_MOCK=1; every path above is relative to mock/.")


if __name__ == "__main__":
    main()
