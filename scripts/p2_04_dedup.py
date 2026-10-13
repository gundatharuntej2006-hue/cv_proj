# -*- coding: utf-8 -*-
"""P2-04 command line: near-duplicate candidates (C17), handshake H1.

    py -3 scripts/p2_04_dedup.py \
        --emb artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz \
        --splits data/splits/tbx11k_splits_v1.csv \
        --out artifacts/p2/dedup/near_dup_candidates_v1.csv

Reads tau from decisions/p2_dedup_tau.json so the threshold cannot be chosen after
seeing the output. Pass --sweep to print the candidate count at several thresholds,
which is what D06 asks for before the value is locked.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_gate import dedup as D                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "decisions" / "p2_dedup_tau.json"


def main():
    ap = argparse.ArgumentParser(description="P2-04: near-duplicate candidates (C17)")
    ap.add_argument("--emb", required=True, help="C07 npz, FULL-image variant")
    ap.add_argument("--splits", required=True, help="C01 split file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=None,
                    help="override D06; recorded in the output as an override")
    ap.add_argument("--sweep", action="store_true", help="candidate counts per threshold")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--official-splits", action="store_true",
                    help="splits are TBX11K's own train/val/test, not C01 names. For the "
                         "provisional pass before C01 exists; the output is NOT C17.")
    a = ap.parse_args()

    import pandas as pd

    if a.tau is None:
        if not DECISION.exists():
            raise SystemExit(
                "D06 is not written yet (%s). The threshold is pre-registered before the "
                "search runs, not chosen from its output. Write it, or pass --tau to "
                "explore." % DECISION)
        d06 = json.load(open(DECISION, encoding="utf-8"))
        tau = float(d06["value"]["tau"])
        inclusive = d06["value"]["comparison"] == ">="
        print("tau %.4f from D06 (%s), comparison %s"
              % (tau, "locked" if d06.get("locked") else "not yet locked",
                 d06["value"]["comparison"]))
    else:
        tau, inclusive = a.tau, True
        print("tau %.4f from --tau (override; D06 still governs the committed file)" % tau)

    order = D.OFFICIAL_ORDER if a.official_splits else D.SPLIT_ORDER
    ids, emb, variant = D.load_c07(a.emb)
    splits = pd.read_csv(a.splits)[["image_id", "split"]]
    index_of = {i: k for k, i in enumerate(ids)}     # NOT `order`: that is the split precedence
    missing = set(ids) - set(splits.image_id)
    if missing:
        raise SystemExit("C01 has no split for %d embedded images, first few: %s"
                         % (len(missing), sorted(missing)[:5]))
    splits = splits[splits.image_id.isin(index_of)].copy()
    splits["k"] = [index_of[i] for i in splits.image_id]
    splits = splits.sort_values("k")
    emb = emb[splits.k.to_numpy()]
    ids_sorted = splits.image_id.tolist()
    spl = splits.split.tolist()
    print("variant %s, %d images, splits %s" % (variant, len(ids_sorted), sorted(set(spl))))

    if a.sweep:
        print("\nthreshold sweep (cross-split candidates):")
        print(D.threshold_sweep(ids_sorted, spl, emb, block=a.block,
                                order=order).to_string(index=False))
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
            "embedding_file": str(a.emb), "variant": variant, "n_images": len(ids_sorted),
            "tau": tau, "comparison": ">=" if inclusive else ">",
            "tau_source": "D06" if a.tau is None else "cli_override",
            "cross_split_candidates": int(len(df)), "within_split_pairs": int(within)}
    json.dump(meta, open(str(out) + ".json", "w", encoding="utf-8"), indent=1)
    print("sidecar %s.json" % out)
    print("\nNext: P1-07 reviews every row and records a verdict before the split freeze.")


if __name__ == "__main__":
    main()
