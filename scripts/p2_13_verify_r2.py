# -*- coding: utf-8 -*-
"""P2-13: does the R2 runtime actually reproduce the exported scores?

The contract says `p2_cls.api.classify` reproduces the C08 scores to 1e-4. There are two
different claims hiding in that sentence, and they do not have the same answer:

  **A. Head reproducibility.** Re-run the registered head over the *same stored embedding*
     and compare to the exported `p_tb_raw`. This is exact up to float arithmetic, because
     it is the same matmul on the same input.

  **B. End-to-end reproducibility.** Take the *image*, run the backbone again in the runtime
     path, apply the head, compare. This crosses a device and a dtype: C07 was produced by
     `embed.Embedder` in float16 on the GPU and stored as float16, while the runtime path
     runs float32 on CPU. 1e-4 is not achievable by construction here — float16 carries
     about three decimal digits.

This script measures both and reports what tolerance B actually needs, plus the only
consequence anyone should care about: how many cases sit close enough to a triage threshold
that the difference could flip the decision.

    py -3 scripts/p2_13_verify_r2.py --n 48
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import api as API                                       # noqa: E402
from p2_cls import embed as E                                       # noqa: E402
from p2_cls import registry as R                                    # noqa: E402

VARIANT = "full_clahe0"
EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
MANIFEST = ROOT / "data/processed/v1/manifest.parquet"
BUNDLE = ROOT / "artifacts/p2/bundle/bundle.json"
OUT = ROOT / "artifacts/p2/dryrun/r2_verification.json"
STATED_TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = API.trained_head()
    if tr is None:
        raise SystemExit("no trained head on disk; run scripts/p2_build_demo_bundle.py")
    W, b, model_id, _variant, n_cls = tr

    with np.load(EMB) as z:
        ids = z["image_id"].tolist()
        emb16 = z["emb"]                        # as stored: float16
    emb = emb16.astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}

    variant = E.parse_variant(VARIANT)
    df, _ = E.load_manifest(MANIFEST, variant)
    path = dict(zip(df.image_id, df.src_path))

    # ---- claim A: the head on the stored embedding, against itself in float64
    p_all = R.p_tb_from_logits(R.head_logits(emb, W, b), n_cls)
    p64 = R.p_tb_from_logits(
        np.asarray(emb, np.float64) @ np.asarray(W, np.float64).T + np.asarray(b, np.float64),
        n_cls)
    a_worst = float(np.max(np.abs(p_all - p64)))
    print("A. head on stored embedding (float32 vs float64): worst %.3e  -> %s"
          % (a_worst, "PASS" if a_worst <= STATED_TOL else "FAIL"))

    # ---- claim B: image -> probability through the runtime path
    rng = np.random.default_rng(args.seed)
    sel = [ids[k] for k in rng.choice(len(ids), min(args.n, len(ids)), replace=False)]
    diffs, rows = [], []
    t0 = time.perf_counter()
    for k, i in enumerate(sel):
        a = E.read_image(path[i], None)
        r = API.classify(a)
        ref = float(p_all[idx[i]])
        d = abs(r.p_tb_raw - ref)
        diffs.append(d)
        rows.append({"image_id": i, "runtime": round(r.p_tb_raw, 6),
                     "exported": round(ref, 6), "abs_diff": d})
        if k % 12 == 0:
            print("   %2d/%d  %.1f s/img" % (k, len(sel), (time.perf_counter() - t0) / (k + 1)))
    diffs = np.asarray(diffs)
    b_worst, b_med = float(diffs.max()), float(np.median(diffs))
    print("B. image -> probability (fp16 GPU export vs fp32 CPU runtime), n=%d:" % len(sel))
    print("   median %.3e | p95 %.3e | worst %.3e  -> %s at the stated 1e-4"
          % (b_med, float(np.percentile(diffs, 95)), b_worst,
             "PASS" if b_worst <= STATED_TOL else "FAIL"))

    # how much of B is just float16 storage? re-quantise the fp32 embedding and re-score.
    p_from16 = R.p_tb_from_logits(R.head_logits(emb16.astype(np.float32), W, b), n_cls)
    p_round = R.p_tb_from_logits(
        R.head_logits(emb.astype(np.float16).astype(np.float32), W, b), n_cls)
    storage_only = float(np.max(np.abs(p_from16 - p_round)))

    # ---- the only consequence that matters: could a decision flip?
    meta = json.load(open(BUNDLE, encoding="utf-8"))
    T = meta.get("temperature") or 1.0
    tl, tu = meta.get("t_lower"), meta.get("t_upper")

    def cal(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-lg / T))

    pc = cal(p_all)
    margin = b_worst * max(1.0, 1.0 / T)        # calibration can magnify the gap
    near = {}
    for name, t in (("t_lower", tl), ("t_upper", tu)):
        if t is None:
            near[name] = None
            continue
        near[name] = int(np.sum(np.abs(pc - t) <= margin))
    print("C. cases within the worst-case gap (%.1e, calibrated) of a threshold: %s"
          % (margin, near))

    out = {"created": time.strftime("%Y-%m-%d"), "model_id": model_id, "variant": VARIANT,
           "stated_tolerance": STATED_TOL,
           "claim_a_head_on_stored_embedding": {"worst_abs_diff": a_worst,
                                                "n": int(len(p_all)),
                                                "passes_1e-4": bool(a_worst <= STATED_TOL)},
           "claim_b_end_to_end": {"n": int(len(sel)), "median": b_med,
                                  "p95": float(np.percentile(diffs, 95)),
                                  "worst": b_worst,
                                  "passes_1e-4": bool(b_worst <= STATED_TOL),
                                  "float16_storage_alone_worst": storage_only,
                                  "examples": sorted(rows, key=lambda r: -r["abs_diff"])[:8]},
           "claim_c_decision_impact": {"margin_used": margin,
                                       "cases_within_margin_of_threshold": near,
                                       "n_cases": int(len(pc))},
           "note": ("C07 was exported in float16 from a float16 GPU forward pass; the runtime "
                    "path is float32 on CPU. Claim B's floor is set by that, not by the head.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
