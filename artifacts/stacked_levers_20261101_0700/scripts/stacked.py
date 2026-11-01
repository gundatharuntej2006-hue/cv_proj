# -*- coding: utf-8 -*-
"""The three external levers stacked, exactly as PROTOCOL.md.

    py -3 stacked.py          # dry: counts, guards
    py -3 stacked.py --run    # embed NLM with MAIRA-2 (resumable), fit five arms, statistics
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
CACHE = RUN / "cache"
R20 = ROOT / "artifacts/multiencoder_ensemble_20261101_0600"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(R20 / "scripts"))
import p2_32_mil_sweep as S                                         # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402
import ensemble as E20                                              # noqa: E402  (R20: Embedder, trainer, stats)

REG = ROOT / "artifacts/p2/regional"
CALEMB = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/regional_tbx11k_calibration_lungs_g12_v1.npz"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
VSPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VCSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
NLM_IMG = ROOT / "data/processed/v1/nlm_img512_lungs"
SEEDS = E20.SEEDS
N_BOOT, BOOT_SEED, MIN_EFFECT, G = E20.N_BOOT, E20.BOOT_SEED, E20.MIN_EFFECT, E20.G
TB, ACTIVE = E20.TB, E20.ACTIVE
REPRO = {"A": 0.9498, "MB1": 0.9497}
SINGLE_LEVER_TUNE_SUM = 0.0203 + 0.0077 + 0.0188


class Blocker(SystemExit):
    pass


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    p = TAB / "protocol_hash.txt"
    if p.is_file() and p.read_text(encoding="utf-8").split()[0] != h:
        raise Blocker("PROTOCOL.md changed after hashing")
    TAB.mkdir(exist_ok=True)
    p.write_text(h + "  PROTOCOL.md\n", encoding="utf-8")
    return h


def nlm_label(image_id):
    v = image_id.rsplit("_", 1)[1]
    if v not in ("0", "1"):
        raise Blocker("unexpected NLM id %s" % image_id)
    return int(v)


def verdict(delta, lo, hi):
    if (hi - lo) / 2 > 0.05:
        return "INCONCLUSIVE"
    return "ACCEPT" if (delta >= MIN_EFFECT and lo > 0) else "REJECT"


def thr_at_spec(y, s, spec):
    neg = np.sort(s[y == 0])
    return float(neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    t0 = time.perf_counter()
    proto = check_protocol()
    logf = open(RUN / "run.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    c01 = pd.read_csv(C01)
    forbidden = set(c01[c01.split == "internal_test"].image_id)
    tr = c01[(c01.split == "train") & c01.category.isin(TB + ("healthy",))]
    dv = c01[c01.split == "development"]
    cal = c01[c01.split == "calibration"]
    assert (len(tr), len(dv), len(cal)) == (3241, 659, 899)
    sp = json.loads(VSPLIT.read_text(encoding="utf-8"))
    tune, report = [str(i) for i in sp["tune"]], {str(i) for i in sp["report"]}
    with np.load(REG / "regional_nlm_lungs_g12_v1.npz") as z:
        nid = z["image_id"].tolist()
    y_nlm = np.array([nlm_label(i) for i in nid])
    assert len(nid) == 800
    for key in ("D2",):
        for n in ("tbx_train", "tbx_dev", "tbx_cal", "vie_tune"):
            if not (R20 / "cache" / ("regional_%s_%s_lungs_g12.npz" % (key, n))).is_file():
                raise Blocker("R20 cache missing: %s %s" % (key, n))
    log("train %d dev %d cal %d tune %d nlm %d (tb %d)" % (len(tr), len(dv), len(cal), len(tune), len(nid), y_nlm.sum()))
    if not a.run:
        log("DRY. Pass --run.")
        return 0

    # --- MAIRA-2 features for NLM (same PNGs as the RAD-DINO NLM cache), resumable
    nlm_cache = CACHE / "regional_D2_nlm_lungs_g12.npz"
    if not nlm_cache.is_file():
        emb = E20.Embedder("D2")
        paths = [NLM_IMG / (i + ".png") for i in nid]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise Blocker("%d NLM images missing" % len(missing))
        E20.CACHE = CACHE                      # write into this run's cache
        E20.embed_set(emb, "D2", "nlm", paths, log)
        del emb
    Xnlm_D2 = E20.load_cache(nlm_cache, nid)

    def d2(n, ids):
        return E20.load_cache(R20 / "cache" / ("regional_D2_%s_lungs_g12.npz" % n), ids)

    F = {"D": {"tr": S.repool(E20.load_cache(REG / "regional_tbx11k_lungs_g12_v1.npz", tr.image_id), 12, G),
               "dv": S.repool(E20.load_cache(REG / "regional_tbx11k_lungs_g12_v1.npz", dv.image_id), 12, G),
               "cal": S.repool(E20.load_cache(CALEMB, cal.image_id), 12, G),
               "tune": S.repool(E20.load_cache(REG / "regional_vietnam_lungs_g12_v1.npz", tune), 12, G),
               "nlm": S.repool(E20.load_cache(REG / "regional_nlm_lungs_g12_v1.npz", nid), 12, G)},
         "D2": {"tr": d2("tbx_train", tr.image_id), "dv": d2("tbx_dev", dv.image_id),
                "cal": d2("tbx_cal", cal.image_id), "tune": d2("vie_tune", tune), "nlm": Xnlm_D2}}
    vdf = pd.read_csv(VCSV).set_index("pid")
    y_tune = vdf.loc[[int(i) for i in tune], "xpert_result"].to_numpy().astype(int)
    y_cal = np.array([1 if c in ACTIVE else 0 for c in cal.category])
    y_dv_tb = np.array([1 if c in TB else 0 for c in dv.category])

    ARMS = {"A": ("D", TB, False), "MB1": ("D", ACTIVE, True), "X": ("D2", ACTIVE, True),
            "X_noNLM": ("D2", ACTIVE, False), "X_tb": ("D2", TB, True)}
    out, per_seed = {}, []
    for arm, (enc, pos, with_nlm) in ARMS.items():
        ytr = np.array([1 if c in pos else 0 for c in tr.category])
        ydv = np.array([1 if c in pos else 0 for c in dv.category])
        Xtr, ytr_ = ((np.concatenate([F[enc]["tr"], F[enc]["nlm"]]), np.concatenate([ytr, y_nlm]))
                     if with_nlm else (F[enc]["tr"], ytr))
        log("\narm %-8s enc=%s target=%s rows=%d (pos %d)" % (arm, enc, "tb" if pos == TB else "active", len(ytr_), ytr_.sum()))
        Ws = []
        for sd in SEEDS:
            W, b, best = E20.train(Xtr, ytr_, F[enc]["dv"], ydv, sd)
            Ws.append((W, b))
            per_seed.append({"arm": arm, "seed": sd, "early_stop_dev_auc": round(best, 4),
                             "cal_auroc_vs_active": round(SEL.auc(y_cal, E20.image_logits(F[enc]["cal"], W, b)), 4)})
            log("   seed %2d dev %.4f" % (sd, best))
        out[arm] = {s: np.mean([E20.image_logits(F[enc][s], W, b) for W, b in Ws], 0) for s in ("dv", "cal", "tune")}
        if arm == "X":
            np.savez(CACHE / "X_head.npz", W=np.stack([w for w, _ in Ws]), b=np.stack([b_ for _, b_ in Ws]))
    E20.write_csv(TAB / "per_seed.csv", per_seed)

    gate, ok = [], True
    for arm, want in REPRO.items():
        here = SEL.auc(y_cal, out[arm]["cal"])
        p = abs(here - want) <= 5e-4
        ok &= p
        gate.append({"arm": arm, "expected": want, "here": round(here, 4), "pass": p})
        log("gate %-4s expected %.4f here %.4f %s" % (arm, want, here, "pass" if p else "FAIL"))
    E20.write_csv(TAB / "reproduction_gate.csv", gate)
    if not ok:
        raise Blocker("reproduction gate failed")

    metrics = []
    for arm in ARMS:
        ac, lc, hc = E20.boot_auc(y_cal, out[arm]["cal"], np.random.default_rng(BOOT_SEED))
        at, lt, ht = E20.boot_auc(y_tune, out[arm]["tune"], np.random.default_rng(BOOT_SEED))
        row = {"arm": arm, "dev_auroc": round(SEL.auc(y_dv_tb, out[arm]["dv"]), 4),
               "cal_auroc_vs_active": round(ac, 4), "cal_ci": "[%.4f, %.4f]" % (lc, hc),
               "cal_auprc": round(E20.auprc(y_cal, out[arm]["cal"]), 4),
               "tune_auroc_EXPLORATORY": round(at, 4), "tune_ci": "[%.4f, %.4f]" % (lt, ht),
               "tune_auprc": round(E20.auprc(y_tune, out[arm]["tune"]), 4)}
        for spec in (0.70, 0.90):
            thr = thr_at_spec(y_dv_tb, out[arm]["dv"], spec)
            for s, y in (("cal", y_cal), ("tune", y_tune)):
                sc = out[arm][s]
                row["%s_sens_at_dev_spec%.2f" % (s, spec)] = round(float((sc[y == 1] > thr).mean()), 4)
                row["%s_spec_at_dev_spec%.2f" % (s, spec)] = round(float((sc[y == 0] <= thr).mean()), 4)
        metrics.append(row)
        log("%-8s dev %.4f cal %.4f [%.4f, %.4f] tune %.4f [%.4f, %.4f]" % (arm, row["dev_auroc"], ac, lc, hc, at, lt, ht))
    E20.write_csv(TAB / "metrics_by_arm.csv", metrics)

    diffs = []
    for a_, b_, primary in (("A", "X", True), ("MB1", "X", False), ("A", "X_noNLM", False),
                            ("A", "X_tb", False), ("X_noNLM", "X", False), ("X_tb", "X", False), ("A", "MB1", False)):
        dc, lc, hc = E20.paired(y_cal, out[a_]["cal"], out[b_]["cal"], np.random.default_rng(BOOT_SEED))
        dt, lt, ht = E20.paired(y_tune, out[a_]["tune"], out[b_]["tune"], np.random.default_rng(BOOT_SEED))
        v = verdict(dc, lc, hc) if primary else ""
        diffs.append({"contrast": "%s - %s" % (b_, a_), "primary": primary, "cal_delta": round(dc, 4),
                      "cal_ci": "[%+.4f, %+.4f]" % (lc, hc), "tune_delta_EXPLORATORY": round(dt, 4),
                      "tune_ci": "[%+.4f, %+.4f]" % (lt, ht), "verdict": v})
        log("%s - %s: cal %+.4f [%+.4f, %+.4f] | tune %+.4f [%+.4f, %+.4f] %s" % (b_, a_, dc, lc, hc, dt, lt, ht, v))
    E20.write_csv(TAB / "paired_differences.csv", diffs)
    x_a_tune = [d for d in diffs if d["contrast"] == "X - A"][0]["tune_delta_EXPLORATORY"]

    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "encoder_X": {"model": E20.ENCODERS["D2"][0], "revision": E20.ENCODERS["D2"][1]},
        "rad_dino_revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
        "seeds": SEEDS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "calibration_n": int(len(y_cal)), "calibration_pos": int(y_cal.sum()),
        "tune_n": int(len(y_tune)), "tune_pos": int(y_tune.sum()),
        "additivity": {"sum_of_single_lever_tune_deltas": round(SINGLE_LEVER_TUNE_SUM, 4),
                       "observed_X_minus_A_tune": x_a_tune},
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=1),
        encoding="utf-8")
    log("\ndone in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
