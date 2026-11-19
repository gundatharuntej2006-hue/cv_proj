# -*- coding: utf-8 -*-
"""P2-04 command line: near-duplicate candidates (C17), handshake H1.

C17 run on P1's C01 (the form P1-07 accepts):

    py -3 scripts/p2_04_dedup.py \
        --emb artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz \
        --splits data/splits/tbx11k_splits_v0.csv \
        --c01-sha256 <sha256 of the C01 file, as published by P1> \
        --restrict-to-c01 \
        --out artifacts/p2/dedup/near_dup_candidates_v1.csv

--restrict-to-c01 drops embeddings whose image is not in C01 (the extra TBX11K images C01
leaves out) and then requires the embedded ID set to EQUAL the C01 ID set: zero missing,
zero extra, no duplicate IDs. Anything else stops the run. Only the C01 `split` column is
read; `official_split` is ignored.

The threshold comes from a D06 decision file (--decision), read before the search runs, so
it cannot be chosen after seeing the output. The sidecar records which file, the value, the
comparison and the lock state, plus the candidate count at every swept threshold.
"""
import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_gate import dedup as D                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "decisions" / "p2_dedup_tau.json"
SWEEP_TAUS = (0.95, 0.96, 0.97, 0.98, 0.99)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_decision(path):
    d06 = json.load(open(path, encoding="utf-8"))
    v = d06.get("value", {})
    if "tau" not in v:
        raise SystemExit("%s has no value.tau" % path)
    comparison = v.get("comparison")
    if comparison not in (">=", ">"):
        raise SystemExit("%s does not state the comparison ('>=' or '>'); a threshold without "
                         "its operator is ambiguous at the boundary" % path)
    return {"d06_file": str(Path(path).as_posix()), "tau": float(v["tau"]),
            "comparison": comparison, "locked": bool(d06.get("locked")),
            "locked_at": d06.get("locked_at"), "guide_ack": d06.get("guide_ack")}


def restrict_and_check(ids, emb, c01_ids):
    """Keep embeddings for C01 images only, then require the two ID sets to be equal.

    Returns (kept_ids, kept_emb, coverage). Raises SystemExit on any mismatch.
    """
    c01_list = list(c01_ids)
    dup_c01 = len(c01_list) - len(set(c01_list))
    dup_emb = len(ids) - len(set(ids))
    if dup_c01 or dup_emb:
        raise SystemExit("duplicate image IDs: %d in C01, %d in the embedding file"
                         % (dup_c01, dup_emb))
    c01 = set(c01_list)
    keep = [k for k, i in enumerate(ids) if i in c01]
    kept_ids = [ids[k] for k in keep]
    kept = set(kept_ids)
    missing = sorted(c01 - kept)
    extra = sorted(kept - c01)
    coverage = {"n_c01_ids": len(c01), "n_embedded_before_filter": len(ids),
                "n_excluded_not_in_c01": len(ids) - len(keep), "n_embedded": len(kept_ids),
                "missing_embeddings": len(missing), "extra_embeddings": len(extra),
                "coverage_check": "PASS" if not missing and not extra else "FAIL"}
    if missing or extra:
        raise SystemExit("coverage check FAILED: %d C01 images have no embedding (first: %s), "
                         "%d embedded images are not in C01 (first: %s)"
                         % (len(missing), missing[:5], len(extra), extra[:5]))
    return kept_ids, emb[keep], coverage


def main():
    ap = argparse.ArgumentParser(description="P2-04: near-duplicate candidates (C17)")
    ap.add_argument("--emb", required=True, help="C07 npz, FULL-image variant")
    ap.add_argument("--splits", required=True, help="C01 split file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decision", default=str(DECISION), help="D06 file the threshold is read from")
    ap.add_argument("--c01-sha256", default=None,
                    help="expected SHA-256 of the C01 file; the run stops on a mismatch")
    ap.add_argument("--restrict-to-c01", action="store_true",
                    help="drop embeddings outside C01, then require equal ID sets")
    ap.add_argument("--tau", type=float, default=None,
                    help="override D06; recorded in the output as an override (not valid for C17)")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--official-splits", action="store_true",
                    help="splits are TBX11K's own train/val/test, not C01 names. For the "
                         "provisional pass before C01 exists; the output is NOT C17.")
    a = ap.parse_args()

    import pandas as pd

    if a.tau is None:
        if not Path(a.decision).exists():
            raise SystemExit("D06 file %s not found. The threshold is pre-registered before "
                             "the search runs, not chosen from its output." % a.decision)
        dec = read_decision(a.decision)
        tau, inclusive = dec["tau"], dec["comparison"] == ">="
        print("tau %.4f %s from %s (%s)" % (tau, dec["comparison"], dec["d06_file"],
                                             "locked" if dec["locked"] else "not locked"))
    else:
        dec = {"d06_file": None, "tau": a.tau, "comparison": ">=", "locked": False,
               "locked_at": None, "guide_ack": None}
        tau, inclusive = a.tau, True
        print("tau %.4f from --tau (override; not a valid C17)" % tau)

    c01_sha = sha256_of(a.splits)
    print("C01 %s sha256 %s" % (a.splits, c01_sha))
    if a.c01_sha256 and c01_sha != a.c01_sha256.lower():
        raise SystemExit("C01 SHA-256 mismatch: file is %s, expected %s" % (c01_sha, a.c01_sha256))

    order = D.OFFICIAL_ORDER if a.official_splits else D.SPLIT_ORDER
    ids, emb, variant = D.load_c07(a.emb)
    splits = pd.read_csv(a.splits)[["image_id", "split"]]     # C01 `split`, never `official_split`
    splits["image_id"] = splits.image_id.astype(str)

    if a.restrict_to_c01:
        ids, emb, coverage = restrict_and_check(ids, emb, splits.image_id.tolist())
        print("coverage: %d C01 IDs, %d embedded after dropping %d outside C01; "
              "missing %d, extra %d -> %s"
              % (coverage["n_c01_ids"], coverage["n_embedded"], coverage["n_excluded_not_in_c01"],
                 coverage["missing_embeddings"], coverage["extra_embeddings"],
                 coverage["coverage_check"]))
    else:
        no_split = set(ids) - set(splits.image_id)
        if no_split:
            raise SystemExit("C01 has no split for %d embedded images, first few: %s "
                             "(pass --restrict-to-c01 to drop them and check coverage both ways)"
                             % (len(no_split), sorted(no_split)[:5]))
        no_emb = set(splits.image_id) - set(ids)
        coverage = {"n_c01_ids": int(splits.image_id.nunique()), "n_embedded": len(ids),
                    "missing_embeddings": len(no_emb), "extra_embeddings": 0,
                    "coverage_check": "PASS" if not no_emb else "FAIL"}
        if no_emb:
            raise SystemExit("%d C01 images have no embedding, first few: %s"
                             % (len(no_emb), sorted(no_emb)[:5]))

    index_of = {i: k for k, i in enumerate(ids)}
    splits = splits[splits.image_id.isin(index_of)].copy()
    splits["k"] = [index_of[i] for i in splits.image_id]
    splits = splits.sort_values("k")
    emb = emb[splits.k.to_numpy()]
    ids_sorted = splits.image_id.tolist()
    spl = splits.split.tolist()
    print("variant %s, %d images, splits %s" % (variant, len(ids_sorted), sorted(set(spl))))

    sweep = D.threshold_sweep(ids_sorted, spl, emb, taus=SWEEP_TAUS, block=a.block, order=order)
    print("\nthreshold sweep (cross-split candidates):")
    print(sweep.to_string(index=False))
    print()

    df, within = D.find_near_duplicates(ids_sorted, spl, emb, tau,
                                        inclusive=inclusive, block=a.block, order=order)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("wrote %s: %d cross-split candidates" % (out, len(df)))
    if len(df):
        pairs = df.groupby(["split_a", "split_b"]).size().sort_values(ascending=False)
        for (sa, sb), c in pairs.items():
            flag = "  <-- leaks held-out data" if sb in ("internal_test", "challenge_test") else ""
            print("    %-14s -> %-14s %4d%s" % (sa, sb, c, flag))
        print("    cosine range %.4f to %.4f" % (df.cosine.min(), df.cosine.max()))
    print("%d within-split duplicate pairs (not in C17; they shrink the effective n and "
          "belong in the report, not the split freeze)" % within)

    meta = {"task": "P2-04", "contract": "C17", "date": str(date.today()),
            "c01_path": str(Path(a.splits).as_posix()), "c01_sha256": c01_sha,
            "split_column": "split",
            **coverage,
            "embedding_file": str(Path(a.emb).as_posix()), "embedding_sha256": sha256_of(a.emb),
            "variant": variant, "n_images": len(ids_sorted),
            "tau": tau, "comparison": ">=" if inclusive else ">",
            "tau_source": "D06" if a.tau is None else "cli_override",
            "d06_file": dec["d06_file"], "d06_locked": dec["locked"],
            "d06_locked_at": dec["locked_at"], "d06_guide_ack": dec["guide_ack"],
            "cross_split_candidates": int(len(df)), "within_split_pairs": int(within),
            "sweep_cross_split_candidates": {"%.2f" % t: int(c) for t, c in
                                             zip(sweep.tau, sweep.cross_split_candidates)},
            "scope": "cross-split pairs only, within the C01 image set; official TBX11K test "
                     "images are not in C01 and are not searched"}
    json.dump(meta, open(str(out) + ".json", "w", encoding="utf-8"), indent=1)
    print("sidecar %s.json" % out)
    print("\nNext: P1-07 reviews every row and records a verdict before the split freeze.")


if __name__ == "__main__":
    main()
