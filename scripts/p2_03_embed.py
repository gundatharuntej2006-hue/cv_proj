# -*- coding: utf-8 -*-
"""P2-03 command line: produce a C07 embedding file.

Normal use, once Person 1's C04 exists:

    py -3 scripts/p2_03_embed.py --manifest data/processed/v1/manifest.parquet \
        --variant full_clahe0 --out artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz

A crop variant additionally needs C05:

    ... --variant crop_m0.10_clahe0 --roi data/processed/v1/lung_roi.parquet

Before C04 exists, to unblock the Week 2 near-duplicate search only:

    ... --raw-dir data/raw/tbx11k --variant full_clahe0 --out <...>.npz

Add --verify to re-embed a random subsample and report the worst cosine against the
file just written, which is the task's acceptance criterion.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_cls import embed as E                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="P2-03: RAD-DINO embeddings (contract C07)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="C04 manifest (parquet or csv)")
    src.add_argument("--raw-dir", help="stand-in: a folder of images, full/clahe0 only")
    ap.add_argument("--variant", required=True, help="e.g. full_clahe0, crop_m0.10_clahe0")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--roi", help="C05 lung_roi.parquet, required for crop variants")
    ap.add_argument("--batch", type=int, default=16, help="16 fits in 0.67 GB; see P2-01")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N images, for a smoke test")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verify", action="store_true", help="re-run check on a subsample")
    ap.add_argument("--verify-n", type=int, default=64)
    a = ap.parse_args()

    variant = E.parse_variant(a.variant)
    df, preproc = E.load_manifest(a.manifest, variant, roi_path=a.roi, raw_dir=a.raw_dir)
    if a.limit:
        df = df.sort_values("image_id").head(a.limit)
    print("variant        : %s" % variant.name)
    print("images         : %d" % len(df))
    print("preproc_version: %s%s" % (preproc,
          "   <-- NOT the real C07; re-run against C04" if preproc != E.REAL_PREPROC else ""))

    t0 = time.perf_counter()
    out, n = E.run(df, variant, a.out, preproc, batch=a.batch, device=a.device,
                   resume=not a.no_resume)
    seconds = time.perf_counter() - t0

    worst = None
    if a.verify:
        worst, _ = E.verify(out, df, variant, sample=a.verify_n, batch=a.batch, device=a.device)

    p, meta = E.sidecar(out, variant, preproc, df, n, seconds, worst)
    print("\nwrote %s  (%d x %d float16)" % (out, n, meta["emb_dim"]))
    print("sidecar %s" % p)
    print("%.1f s for %d images -> %.1f min per 11,200 (criterion: 60)"
          % (seconds, n, meta["minutes_per_11200"]))
    if worst is not None:
        ok = worst >= 0.9999
        print("re-run worst cosine %.6f -> %s (criterion: >= 0.9999)"
              % (worst, "PASS" if ok else "FAIL"))
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
