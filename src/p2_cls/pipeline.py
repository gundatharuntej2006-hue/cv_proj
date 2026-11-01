# -*- coding: utf-8 -*-
"""End-to-end inference: an image path in, a triage decision out.

    from p2_cls.pipeline import load_bundle, run_case
    bundle = load_bundle()                 # artifacts/p2/bundle/
    result = run_case("some_xray.png", bundle)

This is the path a dashboard or API calls. It runs the real stages in the methodology's
order and returns a plain dict, so a frontend can render it without importing torch:

    standardise -> quality gate -> embed -> OOD gate -> classify -> calibrate -> triage

Every stage can stop the case. A rejected image returns `analysis: None` with a reason,
rather than a probability nobody should trust.

Scope and ownership: the gate and classifier are Person 2's. Calibration and the triage
thresholds are Person 4's decisions (D22/D23) in the real system; the bundle carries
values computed by `p2_build_demo_bundle.py` so the system can run end to end today.
Swap in P4's fitted values when they exist — the bundle format does not change.

The returned dict mirrors the CaseResult contract's fields closely enough that P4's
schema can wrap it, but it is deliberately plain JSON-able types.
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT / "artifacts/p2/bundle"


def load_bundle(path=BUNDLE_DIR):
    """Load the inference bundle once and reuse it; the backbone load dominates cost."""
    from p2_cls import registry as R
    from p2_gate import ood as O
    p = Path(path)
    meta = json.load(open(p / "bundle.json", encoding="utf-8"))
    entry = R.RegistryEntry(**meta["model"])
    entry.weight_path = str(p / "head.npz")           # make portable
    with np.load(p / "bank.npz") as z:
        bank = z["bank"].astype(np.float32)
    return {"meta": meta, "entry": entry, "bank": O.OodBank(bank, k=meta.get("ood_k", 5)),
            "_embedder": None}


def _embedder(bundle):
    from p2_cls.embed import Embedder
    if bundle.get("_embedder") is None:
        bundle["_embedder"] = Embedder()
    return bundle["_embedder"]


def standardise(image_path):
    """Any readable image -> uint8 512x512 greyscale. Returns (img, error_or_None)."""
    try:
        from PIL import Image
        a = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    except Exception as e:
        return None, "unreadable: %s" % type(e).__name__
    if a.shape != (512, 512):
        try:
            import cv2
            a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            a = np.array(Image.fromarray(a).resize((512, 512), Image.Resampling.BILINEAR))
    return a, None


def triage(p_cal, t_lower, t_upper, n_lesions=0):
    """TB / REFER / NOT_TB plus the detector link rule.

    The link rule only ever moves NOT_TB up to REFER; it can never create or remove a TB
    call. With no detector wired in, n_lesions is 0 and the rule never fires.
    """
    if t_lower is None:
        return {"category": "REFER", "pre_link": "REFER", "link_fired": False,
                "reason": "no conformal threshold available; defaulting to human review"}
    if p_cal >= t_upper:
        cat = "TB"
    elif p_cal < t_lower:
        cat = "NOT_TB"
    else:
        cat = "REFER"
    pre = cat
    fired = False
    if cat == "NOT_TB" and n_lesions >= 1:
        cat, fired = "REFER", True
    return {"category": cat, "pre_link": pre, "link_fired": fired, "reason": ""}


def run_case(image_path, bundle, n_lesions=0, ood_policy=None):
    """Run one image through the full pipeline. Never raises on a bad image."""
    t0 = time.perf_counter()
    meta = bundle["meta"]
    img, err = standardise(image_path)
    if err:
        return {"image": str(image_path), "model_id": meta["model"]["model_id"],
                "variant": meta["variant"], "provisional": meta.get("provisional", True),
                "status": "REJECTED", "stage": "decode", "reasons": [err],
                "analysis": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return run_stages(img, bundle, n_lesions=n_lesions, image_label=str(image_path),
                      t0=t0, ood_policy=ood_policy)


def run_stages(img, bundle, n_lesions=0, image_label=None, t0=None, ood_policy=None):
    """The pipeline from a standardised uint8 512x512 array onwards.

    Split out from `run_case` so that a caller holding an image in memory — the HTTP service,
    which must not write uploads to disk — runs the *same* stages in the same order rather
    than a second copy of them that can drift.
    """
    from p2_cls import registry as R
    from p2_gate import quality as Q
    t0 = time.perf_counter() if t0 is None else t0
    meta = bundle["meta"]
    out = {"image": image_label, "model_id": meta["model"]["model_id"],
           "variant": meta["variant"], "provisional": meta.get("provisional", True)}

    # 1. quality gate
    qr = Q.assess(img)
    out["quality"] = {"pass": bool(qr.passed), "reasons": list(qr.reasons)}
    if not qr.passed:
        out.update(status="REJECTED", stage="quality", reasons=list(qr.reasons),
                   analysis=None, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return out

    # 2. embed
    emb = _embedder(bundle).embed([img]).astype(np.float32)

    # 3. OOD score — ADVISORY by default, never suppressive.
    #
    # This used to return here with analysis=None, which meant an out-of-distribution flag
    # silently cancelled a referral the classifier had not yet been allowed to make. The
    # operational audit priced that: on a screening cohort it deferred 21 tuberculosis
    # patients, 7 of whom the classifier would have correctly referred, and the deferral
    # rate varied threefold by scanner (20.1% vs 6.9%) while scanner carries no information
    # about TB status. See docs/p2_gate_operational_audit.md.
    #
    # The embedding is already computed above, so classifying anyway costs almost nothing.
    # The score is retained as a diagnostic and raises an *additional* review request.
    score = float(bundle["bank"].score(emb)[0])
    thr = meta.get("ood_threshold")
    is_ood = bool(thr is not None and score > thr)
    out["ood"] = {"score": round(score, 4), "threshold": thr, "is_ood": is_ood,
                  "abstained": thr is None}
    policy = ood_policy or meta.get("ood_policy") or "advisory"
    if policy not in ("advisory", "blocking"):
        raise ValueError("ood_policy must be 'advisory' or 'blocking', got %r" % policy)
    if is_ood and policy == "blocking":
        # Legacy behaviour, retained only so prior experiments can be reproduced. It is not
        # the default and should not be used for anything a clinician would read.
        out.update(status="REJECTED", stage="ood", reasons=["out_of_distribution"],
                   analysis=None, ood_policy="blocking",
                   latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return out

    # 4. classify + calibrate
    p_raw = float(R.classify_embeddings(bundle["entry"], emb)[0])
    T = meta["temperature"]
    lg = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
    p_cal = float(1 / (1 + np.exp(-lg / T)))

    # 5. triage. The OOD flag may ADD a review request; it may never remove a referral.
    tr = triage(p_cal, meta.get("t_lower"), meta.get("t_upper"), n_lesions)
    review_reasons = ["out_of_distribution"] if is_ood else []
    out.update(status="ANALYSED", stage="complete", reasons=[],
               ood_policy=policy,
               review_required=bool(review_reasons),
               review_reasons=review_reasons,
               analysis={"p_tb_raw": round(p_raw, 6), "p_tb_calibrated": round(p_cal, 6),
                         "temperature": T,
                         "t_lower": meta.get("t_lower"), "t_upper": meta.get("t_upper"),
                         "triage": tr["category"], "pre_link": tr["pre_link"],
                         "link_fired": tr["link_fired"],
                         "screen_positive": tr["category"] in ("TB", "REFER"),
                         # additive only: a flagged case still appears on the referral list
                         "requires_additional_review": bool(review_reasons)},
               latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return out


def run_batch(paths, bundle, n_lesions=0, ood_policy=None):
    """Several images; a failure on one never stops the rest."""
    results = []
    for p in paths:
        try:
            results.append(run_case(p, bundle, n_lesions, ood_policy=ood_policy))
        except Exception as e:                       # defensive: a worklist must complete
            results.append({"image": str(p), "status": "ERROR",
                            "reasons": ["%s: %s" % (type(e).__name__, e)], "analysis": None})
    return results


def worklist(results):
    """Sort a batch the way a clinician reads it: most likely TB first, unreadable last.

    An out-of-distribution flag does NOT demote a case. A flagged TB referral keeps its
    place at the top of the list and carries `requires_additional_review`; demoting it
    would reintroduce, through ordering, the suppression that was removed from the
    decision logic.
    """
    order = {"TB": 0, "REFER": 1, "NOT_TB": 2}

    def key(r):
        a = r.get("analysis")
        if not a:
            return (3, 0.0)          # only decode/quality failures land here
        return (order.get(a["triage"], 3), -a["p_tb_calibrated"])

    return sorted(results, key=key)
