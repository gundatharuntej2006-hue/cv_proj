# -*- coding: utf-8 -*-
"""Label-free per-scanner score calibration with a fail-closed guard.

What this is: the mean-centering studied in artifacts/unlabelled_scanner_calibration_20261101_1000
(R25). On the exposed Vietnam tuning cohort it recovered the supervised scanner-term gain from
25 unlabelled images per scanner, and it FAILED the pre-registered stability bound when the
calibration batch came from a different population than the one screened (previous-TB-only
batch: scanner sensitivity 0.677 -> 0.555). Verdict there: NOT USABLE AS IS.

What this module therefore does: it proposes per-scanner offsets only when the batch passes
the guard, and otherwise returns no offsets and the reasons. The fallback is always the frozen,
uncorrected model. The guard has NOT been validated prospectively; the status string says so
on every output. Software cannot prove a batch is consecutive - that needs auditable enrolment
records - so the guard checks what it can: batch size and questionnaire composition against a
site reference the operator declares.
"""
import numpy as np

STATUS = ("EXPERIMENTAL - label-free scanner calibration; guard not prospectively validated (R25); "
          "fallback is the uncorrected model")
MIN_N = 25
COMPOSITION_FIELDS = ("past_tb", "sex", "c", "f", "wl", "ns", "contact")
DEFAULT_TOL = 0.15          # absolute difference in a field's batch rate vs the site reference


def composition(clinical_rows):
    """Field rates in a batch of encoded clinical dicts (0/1 fields); None if a field is absent."""
    out = {}
    for f in COMPOSITION_FIELDS:
        vals = [r.get(f) for r in clinical_rows if r.get(f) in (0, 1)]
        out[f] = (round(float(np.mean(vals)), 4) if vals else None)
    return out


def propose(batch_scores, batch_clinical=None, site_reference=None, min_n=MIN_N, tol=DEFAULT_TOL):
    """Per-scanner mean-centering offsets, or a refusal.

    batch_scores    {scanner_id: [image logits of unlabelled local images]}
    batch_clinical  {scanner_id: [encoded clinical dicts]} - optional; without it only the
                    size check runs and the composition check is reported as 'not checked'
    site_reference  {field: expected rate} for the site's screening population, declared by
                    the operator; without it the composition check cannot run and the
                    proposal is refused (fail closed)
    Returns {"status", "applied": bool, "offsets": {scanner: float} or {}, "reasons": [...],
             "per_scanner": {...}}
    """
    reasons, per_scanner, offsets = [], {}, {}
    if not batch_scores:
        return {"status": STATUS, "applied": False, "offsets": {}, "reasons": ["no batch"], "per_scanner": {}}
    if site_reference is None:
        reasons.append("no site reference declared: composition cannot be checked (fail closed)")
    for sc, scores in batch_scores.items():
        s = np.asarray(scores, dtype=float)
        rec = {"n": int(len(s)), "ok": True, "reasons": []}
        if len(s) < min_n:
            rec["ok"] = False
            rec["reasons"].append("batch of %d < %d" % (len(s), min_n))
        if batch_clinical is not None and sc in batch_clinical and site_reference is not None:
            comp = composition(batch_clinical[sc])
            rec["composition"] = comp
            for f, ref in site_reference.items():
                if comp.get(f) is None:
                    rec["ok"] = False
                    rec["reasons"].append("field %s missing from batch" % f)
                elif abs(comp[f] - ref) > tol:
                    rec["ok"] = False
                    rec["reasons"].append("field %s: batch rate %.2f vs site %.2f (tol %.2f)" % (f, comp[f], ref, tol))
        elif site_reference is not None:
            rec["ok"] = False
            rec["reasons"].append("no clinical rows for this scanner: composition not checked")
        if rec["ok"] and site_reference is not None:
            rec["offset"] = round(float(-s.mean()), 4)
            rec["batch_sd"] = round(float(s.std()), 4)
        per_scanner[sc] = rec
    all_ok = site_reference is not None and all(r["ok"] for r in per_scanner.values())
    if all_ok:
        offsets = {sc: r["offset"] for sc, r in per_scanner.items()}
    else:
        reasons += ["%s: %s" % (sc, "; ".join(r["reasons"])) for sc, r in per_scanner.items() if not r["ok"]]
        reasons.append("calibration refused; frozen uncorrected model stays in use; flag for human review")
    return {"status": STATUS, "applied": all_ok, "offsets": offsets, "reasons": reasons, "per_scanner": per_scanner}


def apply(score, scanner, offsets):
    """Corrected score for one image, or the raw score when no offset exists for its scanner."""
    if not offsets or scanner not in offsets:
        return float(score), False
    return float(score + offsets[scanner]), True
