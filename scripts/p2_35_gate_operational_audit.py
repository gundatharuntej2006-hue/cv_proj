# -*- coding: utf-8 -*-
"""What does the OOD gate actually cost in a screening workflow?

Every previous report of this gate quoted *rejection rates*: "100% of lateral films and
non-medical images rejected at a 0.2% in-distribution false-flag cost". That is a statement
about adversarial inputs, and it has never been translated into the thing that matters -
**how many real tuberculosis patients the gate turns away before the classifier ever sees
them**, and what the resulting human-review workload is.

Three modes on the same eligible cases:

  A  classifier alone, no gate
  B  classifier + gate, where a rejected case is **deferred for human review** and is
     never relabelled negative
  C  operational accounting: referrals = classifier-positive among accepted, plus every
     deferred case, since all of them must be looked at by somebody

Definitions, written here so the numbers cannot be misread:

  coverage        accepted / total
  deferral rate   rejected / total (the gate's workload cost)
  sensitivity     among ALL TB cases, not among accepted ones, unless stated
  conditional     a metric computed only on accepted cases; always labelled as such
  referral        a case sent for further assessment; NOT a diagnosis

A deferred tuberculosis case is not a caught case. It is a case the system declined to
assess, and whether it is eventually found depends on a human pathway this project has no
outcome data for. Nothing here estimates end-to-end clinical sensitivity.

    py -3 scripts/p2_35_gate_operational_audit.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p2_cls import selection as SEL                                  # noqa: E402
from p2_gate import ood as O                                         # noqa: E402

OUT = ROOT / "artifacts/p2/external/gate_operational_audit.json"
MD = ROOT / "docs/p2_gate_operational_audit.md"
TB = ("active_tb", "latent_tb", "active_latent_tb")


def wilson(k, n):
    return SEL.wilson_ci(int(k), int(n)) if n else (float("nan"), float("nan"))


def main():
    import pandas as pd
    import p2_32_mil_sweep as S

    meta = json.load(open(ROOT / "artifacts/p2/bundle/bundle.json", encoding="utf-8"))
    thr = meta["ood_threshold"]
    with np.load(ROOT / "artifacts/p2/bundle/bank.npz") as z:
        bank = O.OodBank(z["bank"].astype(np.float32), k=meta.get("ood_k", 5))
    print("gate as built: k=%d bank=%d threshold=%.4f (rule: %s)"
          % (bank.k, len(bank.bank), thr, meta.get("ood_threshold_rule")))
    print("  provenance: threshold chosen as the largest value still rejecting 100%% of two")
    print("  alien pools (lateral films, non-medical images). Those pools are easy and")
    print("  unlike anything a screening clinic sends, so gate metrics below may be")
    print("  optimistic about real-world unfamiliar inputs.\n")

    # the classifier: current best configuration
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    with np.load(ROOT / "artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    ti = {i: k for k, i in enumerate(tid)}
    c = c01[c01.image_id.isin(ti)]
    tr = c[(c.split == "train") & (c.category.isin(TB + ("healthy",)))]
    dv = c[c.split == "development"]
    Ttr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, 12)
    Tdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, 12)
    ytr = np.array([1 if x in TB else 0 for x in tr.category])
    ydv = np.array([1 if x in TB else 0 for x in dv.category])
    del TX

    heads = [S.train(Ttr, ytr, Tdv, ydv, "max", sd)[:2] for sd in (17, 29, 43, 61, 79)]
    p_dv = np.mean([S.score(Tdv, W, b, "max") for W, b in heads], axis=0)

    # operating threshold chosen on DEVELOPMENT at 70% specificity, then frozen
    op = float(np.quantile(p_dv[ydv == 0], 0.70))
    print("operating threshold from development at 70%% specificity: %.4f (frozen)\n" % op)

    # the screening cohort
    csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    with np.load(ROOT / "artifacts/p2/regional/regional_vietnam_lungs_g12_v1.npz") as z:
        vid, VX = z["image_id"].tolist(), z["reg"]
    V = S.repool(VX, 12, 12)
    del VX
    pid = [int(i) for i in vid]
    y = csv.loc[pid, "xpert_result"].to_numpy()
    p = np.mean([S.score(V, W, b, "max") for W, b in heads], axis=0)

    split = json.load(open(ROOT / "artifacts/p2/external/vietnam_tune_report_split.json",
                           encoding="utf-8"))
    tune = set(split["tune"])
    is_tune = np.array([q in tune for q in pid])
    op = float(np.quantile(p[is_tune & (y == 0)], 0.70))
    print("operating threshold refitted on the Vietnam TUNE half at 70%% specificity:"
          " %.4f" % op)
    print("  (a threshold carried over from TBX11K development gives 89% referrals,")
    print("   because the MIL aggregator changes the score scale)")
    print("")

    # the gate scores the GLOBAL embedding, which is what its bank is built from
    with np.load(ROOT / "artifacts/p2/external/radino_vietnam_lungs_only_v1.npz") as z:
        gid, GX = z["image_id"].tolist(), z["emb"].astype(np.float32)
    gmap = {int(i): k for k, i in enumerate(gid)}
    s_ood = bank.score(GX[[gmap[q] for q in pid]])

    accepted = s_ood <= thr
    rejected = ~accepted
    n = len(y)

    rows = {}
    print("=" * 74)
    print("Vietnam screening cohort: %d participants, %d tuberculosis (%.1f%%)"
          % (n, y.sum(), 100 * y.mean()))
    print()

    # ---- Mode A: classifier alone
    predA = p >= op
    sensA = predA[y == 1].mean()
    specA = (~predA[y == 0]).mean()
    print("MODE A - classifier alone, no gate")
    print("  sensitivity %.4f  specificity %.4f  referrals %d" % (sensA, specA, predA.sum()))

    # ---- Mode B: gate first
    cov = accepted.mean()
    rej_tb = rejected[y == 1].sum()
    rej_non = rejected[y == 0].sum()
    lo_tb, hi_tb = wilson(rej_tb, (y == 1).sum())
    print("\nMODE B - gate first, rejected cases DEFERRED (never called negative)")
    print("  coverage %.4f | deferral rate %.4f" % (cov, 1 - cov))
    print("  TB cases deferred     %3d / %3d = %.2f%% [%.2f%%, %.2f%%]"
          % (rej_tb, (y == 1).sum(), 100 * rej_tb / max((y == 1).sum(), 1),
             100 * lo_tb, 100 * hi_tb))
    print("  non-TB cases deferred %3d / %3d = %.2f%%"
          % (rej_non, (y == 0).sum(), 100 * rej_non / max((y == 0).sum(), 1)))
    if accepted.sum() and len(set(y[accepted].tolist())) > 1:
        sB = p[accepted] >= op
        print("  CONDITIONAL on acceptance: sensitivity %.4f  specificity %.4f"
              % (sB[y[accepted] == 1].mean(), (~sB[y[accepted] == 0]).mean()))
        print("  (conditional metrics exclude deferred cases and are NOT comparable to Mode A)")

    # ---- Mode C: operational accounting per 1000
    referred = (accepted & predA).sum()
    deferred = rejected.sum()
    tb_caught = (accepted & predA & (y == 1)).sum()
    tb_deferred = rej_tb
    tb_missed = (accepted & ~predA & (y == 1)).sum()
    k = 1000.0 / n
    print("\nMODE C - operational accounting, per 1,000 screened")
    print("  classifier referrals        %6.0f" % (referred * k))
    print("  deferred for human review   %6.0f" % (deferred * k))
    print("  TOTAL needing human time    %6.0f" % ((referred + deferred) * k))
    print("  --")
    print("  TB flagged by classifier    %6.0f" % (tb_caught * k))
    print("  TB deferred to review       %6.0f   <- found only if the human pathway works"
          % (tb_deferred * k))
    print("  TB MISSED outright          %6.0f   <- called not-TB, no further review"
          % (tb_missed * k))

    # ---- what the gate buys
    print("\nWHAT THE GATE BUYS")
    fp_A = (predA & (y == 0)).sum()
    fp_B = (accepted & predA & (y == 0)).sum()
    print("  false-positive referrals, no gate %d -> with gate %d  (%+d)"
          % (fp_A, fp_B, fp_B - fp_A))
    print("  but it defers %d cases, of which %d have tuberculosis." % (deferred, tb_deferred))
    if deferred:
        print("  Net human workload change: %+d cases." % ((referred + deferred) - predA.sum()))
    if tb_deferred == 0 and deferred == 0:
        verdict = ("The gate is inert on this cohort: it accepts everything, so it neither "
                   "helps nor harms here. Its value is confined to inputs unlike this one.")
    elif tb_deferred > 0:
        verdict = ("The gate defers %d tuberculosis cases. Those are not caught by the "
                   "system; they depend on a human pathway whose outcomes we cannot "
                   "measure." % tb_deferred)
    else:
        verdict = "The gate defers only non-TB cases on this cohort."
    print("  -> " + verdict)

    rows = {"created": time.strftime("%Y-%m-%d"), "cohort": "Vietnam screening",
            "n": int(n), "n_tb": int(y.sum()),
            "gate": {"k": bank.k, "threshold": thr, "bank_n": int(len(bank.bank)),
                     "provenance": meta.get("ood_threshold_rule"),
                     "caveat": "threshold set against lateral/non-medical pools only"},
            "operating_threshold": {"value": round(op, 6),
                                    "chosen_on": "TBX11K development at 70% specificity"},
            "mode_a": {"sensitivity": round(float(sensA), 4),
                       "specificity": round(float(specA), 4),
                       "referrals": int(predA.sum())},
            "mode_b": {"coverage": round(float(cov), 4),
                       "deferral_rate": round(float(1 - cov), 4),
                       "tb_deferred": int(rej_tb), "non_tb_deferred": int(rej_non),
                       "tb_deferred_ci95": [round(lo_tb, 4), round(hi_tb, 4)]},
            "mode_c_per_1000": {"classifier_referrals": round(referred * k),
                                "deferred_for_review": round(deferred * k),
                                "total_human_review": round((referred + deferred) * k),
                                "tb_flagged": round(tb_caught * k),
                                "tb_deferred": round(tb_deferred * k),
                                "tb_missed_outright": round(tb_missed * k)},
            "verdict": verdict,
            "note": ("A deferred case is not a negative and not a caught case. No "
                     "end-to-end clinical sensitivity is estimated, because downstream "
                     "human-review outcomes are unavailable.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
