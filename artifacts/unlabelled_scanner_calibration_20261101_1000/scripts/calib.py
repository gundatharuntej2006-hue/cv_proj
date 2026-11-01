# -*- coding: utf-8 -*-
"""Label-free per-scanner calibration of the v3 candidate score, exactly as PROTOCOL.md.

    py -3 calib.py --run
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
R16 = ROOT / "artifacts/clinical_fusion_20261101_0330"
R20 = ROOT / "artifacts/multiencoder_ensemble_20261101_0600"
R22 = ROOT / "artifacts/fusion_on_stacked_head_20261101_0800"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(R16 / "scripts"))
sys.path.insert(0, str(R20 / "scripts"))
import fusion_cv as F16                                              # noqa: E402  load() with fingerprint guard
import ensemble as E20                                               # noqa: E402

SIZES = (25, 50, 100, 200)
DRAWS = 50
SPLIT_SEED = 0
REFER_FRAC = 0.30
N_BOOT, BOOT_SEED = 2000, 0
CONDITIONS = ("natural", "prevalence 0", "prevalence 0.30", "previous TB only", "no previous TB")
SHIFT_SCANNER = 2
RULE = {"min_fraction_of_B_gain": 0.5, "max_sens_move": 0.10, "max_burden_move": 0.10}


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


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def auc(y, p):
    return F16.auc(y, p)


def split_groups(y, scanner, rng):
    """Per scanner, stratified 50/50 into calibration pool and evaluation group."""
    pool, evl = [], []
    for sc in np.unique(scanner):
        for lab in (0, 1):
            idx = np.where((scanner == sc) & (y == lab))[0]
            idx = idx[rng.permutation(len(idx))]
            h = len(idx) // 2
            pool.extend(idx[:h].tolist())
            evl.extend(idx[h:].tolist())
    return np.array(sorted(pool)), np.array(sorted(evl))


def draw_batch(pool_idx, n, rng, y=None, past=None, condition="natural"):
    """A calibration batch from one scanner's pool. Labels are used ONLY to construct the
    composition-shift conditions (simulating a different population), never by the method."""
    idx = pool_idx
    if condition == "prevalence 0":
        idx = idx[y[idx] == 0]
    elif condition == "prevalence 0.30":
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        npos = min(len(pos), int(round(0.30 * n)))
        return np.concatenate([pos[rng.permutation(len(pos))[:npos]], neg[rng.permutation(len(neg))[:n - npos]]])
    elif condition == "previous TB only":
        idx = idx[past[idx] == 1]
    elif condition == "no previous TB":
        idx = idx[past[idx] == 0]
    n = min(n, len(idx))
    return idx[rng.permutation(len(idx))[:n]]


def correct(arm, s, scanner, batches, y=None):
    """Return corrected scores for all patients (evaluation uses its slice) and the pooled
    corrected calibration scores (for the label-free referral threshold)."""
    out = s.copy()
    cal_scores = []
    if arm == "A":
        for sc, bi in batches.items():
            cal_scores.append(s[bi])
    elif arm == "B":
        from sklearn.linear_model import LogisticRegression
        bi = np.concatenate(list(batches.values()))
        scs = sorted(batches)
        D = np.column_stack([s] + [(scanner == sc).astype(float) for sc in scs[1:]])
        if len(np.unique(y[bi])) < 2:
            return None, None
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(D[bi], y[bi])
        out = lr.decision_function(D)
        cal_scores.append(out[bi])
    else:
        for sc, bi in batches.items():
            m = (scanner == sc)
            mu = s[bi].mean()
            sd = s[bi].std() + 1e-12 if arm == "D" else 1.0
            out[m] = (s[m] - mu) / sd
            cal_scores.append(out[bi])
    return out, np.concatenate(cal_scores)


def evaluate(out, cal_scores, y, scanner, past, evl):
    thr = np.quantile(cal_scores, 1 - REFER_FRAC)
    e = evl
    ref = out[e] >= thr
    r = {"pooled_auroc": auc(y[e], out[e]),
         "prev_tb_auroc": auc(y[e][past[e] == 1], out[e][past[e] == 1]),
         "sens": float(ref[y[e] == 1].mean()), "burden": float(ref.mean())}
    for sc in np.unique(scanner):
        m = scanner[e] == sc
        r["within_auroc_s%d" % sc] = auc(y[e][m], out[e][m])
        r["sens_s%d" % sc] = float(ref[m & (y[e] == 1)].mean())
        r["burden_s%d" % sc] = float(ref[m].mean())
    return r


def summarise(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = round(float(v.mean()), 4)
        out[k + "_lo"] = round(float(np.percentile(v, 2.5)), 4)
        out[k + "_hi"] = round(float(np.percentile(v, 97.5)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    proto = check_protocol()
    import pandas as pd
    pids, y, X, _, fp, past = F16.load()
    m = pd.read_csv(F16.META)
    m["pid"] = m.pid.astype(str)
    scanner = m.set_index("pid").loc[pids, "rad_system"].to_numpy(int)
    with np.load(R22 / "tables/X_noNLM_head.npz") as z:
        Ws = list(zip(z["W"], z["b"]))
    s = np.mean([E20.image_logits(E20.load_cache(R20 / "cache/regional_D2_vie_tune_lungs_g12.npz", pids), W, b) for W, b in Ws], 0)
    if abs(auc(y, s) - 0.7585) > 5e-4:
        raise Blocker("candidate score gate failed")
    pool, evl = split_groups(y, scanner, np.random.default_rng(SPLIT_SEED))
    assert not set(pool) & set(evl) and len(pool) + len(evl) == len(y)
    scs = sorted(np.unique(scanner).tolist())
    pools = {sc: pool[scanner[pool] == sc] for sc in scs}
    print("TUNE %d | evaluation %d (pos %d) | pools %s" % (len(y), len(evl), y[evl].sum(), {sc: len(p) for sc, p in pools.items()}))
    (TAB / "groups.json").write_text(json.dumps({
        "split_seed": SPLIT_SEED, "evaluation_n": int(len(evl)), "evaluation_pos": int(y[evl].sum()),
        "pools": {str(sc): {"n": int(len(p)), "pos": int(y[p].sum())} for sc, p in pools.items()},
        "raw_within_scanner_auroc_eval": {str(sc): round(auc(y[evl][scanner[evl] == sc], s[evl][scanner[evl] == sc]), 4) for sc in scs},
        "raw_pooled_auroc_eval": round(auc(y[evl], s[evl]), 4)}, indent=1), encoding="utf-8")
    if not a.run:
        print("DRY. Pass --run.")
        return 0

    # ---- arms x sizes
    table, drawavg = [], {}
    for n in SIZES:
        for arm in ("A", "B", "C", "D"):
            rows, outs = [], []
            for d in range(DRAWS):
                rng = np.random.default_rng(1000 * n + d)
                batches = {sc: draw_batch(pools[sc], n, rng) for sc in scs}
                out, cal = correct(arm, s, scanner, batches, y)
                if out is None:
                    continue
                rows.append(evaluate(out, cal, y, scanner, past, evl))
                outs.append(out)
            summ = summarise(rows)
            summ.update(arm=arm, n_per_scanner=n, draws=len(rows))
            table.append(summ)
            drawavg[(arm, n)] = np.mean(outs, 0)
            print("n=%3d %s pooled %.4f [%.4f, %.4f] prevTB %.4f sens %.3f burden %.3f | within %s | burden by scanner %s" % (
                n, arm, summ["pooled_auroc"], summ["pooled_auroc_lo"], summ["pooled_auroc_hi"], summ["prev_tb_auroc"],
                summ["sens"], summ["burden"], [summ["within_auroc_s%d" % sc] for sc in scs], [summ["burden_s%d" % sc] for sc in scs]))
    write_csv(TAB / "arms_by_size.csv", table)

    # ---- paired bootstrap at n = 100, draw-averaged scores vs A, evaluation group
    rng = np.random.default_rng(BOOT_SEED)
    idxs = [rng.integers(0, len(evl), len(evl)) for _ in range(N_BOOT)]

    def paired(pa, pb, mask=None):
        mk = np.ones(len(evl), bool) if mask is None else mask
        d = []
        for k in idxs:
            kk = evl[k][mk[k]]
            if len(np.unique(y[kk])) < 2:
                continue
            d.append(auc(y[kk], pb[kk]) - auc(y[kk], pa[kk]))
        e = evl[mk]
        return round(auc(y[e], pb[e]) - auc(y[e], pa[e]), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)

    contrasts = []
    for arm in ("B", "C", "D"):
        d, lo, hi = paired(s, drawavg[(arm, 100)])
        dp, lop, hip = paired(s, drawavg[(arm, 100)], past[evl] == 1)
        contrasts.append({"contrast": "%s - A (n=100, draw-averaged)" % arm, "delta": d, "ci": "[%+.4f, %+.4f]" % (lo, hi),
                          "prev_tb_delta": dp, "prev_tb_ci": "[%+.4f, %+.4f]" % (lop, hip)})
        print("%s - A: %+.4f [%+.4f, %+.4f] | prevTB %+.4f [%+.4f, %+.4f]" % (arm, d, lo, hi, dp, lop, hip))
    write_csv(TAB / "paired_vs_raw_n100.csv", contrasts)

    # ---- composition shift on scanner 2's batch, n = 100
    shift = []
    for cond in CONDITIONS:
        for arm in ("A", "B", "C", "D"):
            rows = []
            for d in range(DRAWS):
                rng = np.random.default_rng(7000 + d)
                batches = {sc: draw_batch(pools[sc], 100, rng, y, past, cond if sc == SHIFT_SCANNER else "natural") for sc in scs}
                out, cal = correct(arm, s, scanner, batches, y)
                if out is None:
                    continue
                rows.append(evaluate(out, cal, y, scanner, past, evl))
            summ = summarise(rows)
            summ.update(condition=cond, arm=arm, draws=len(rows))
            shift.append(summ)
            print("%-16s %s pooled %.4f prevTB %.4f | scanner-2 sens %.3f burden %.3f | pooled sens %.3f" % (
                cond, arm, summ["pooled_auroc"], summ["prev_tb_auroc"], summ["sens_s2"], summ["burden_s2"], summ["sens"]))
    write_csv(TAB / "composition_shift_n100.csv", shift)

    # ---- reading rule
    T = {(r["arm"], r["n_per_scanner"]): r for r in table}
    gainB = T[("B", 100)]["pooled_auroc"] - T[("A", 100)]["pooled_auroc"]
    Sh = {(r["condition"], r["arm"]): r for r in shift}
    verdict = {}
    for arm in ("C", "D"):
        gain = T[(arm, 100)]["pooled_auroc"] - T[("A", 100)]["pooled_auroc"]
        nat = Sh[("natural", arm)]
        worst_sens = max(abs(Sh[(c, arm)]["sens_s2"] - nat["sens_s2"]) for c in CONDITIONS[1:])
        worst_burden = max(abs(Sh[(c, arm)]["burden_s2"] - nat["burden_s2"]) for c in CONDITIONS[1:])
        usable = gain >= RULE["min_fraction_of_B_gain"] * gainB and worst_sens <= RULE["max_sens_move"] and worst_burden <= RULE["max_burden_move"]
        verdict[arm] = {"gain_vs_A_n100": round(gain, 4), "B_gain_vs_A_n100": round(gainB, 4),
                        "worst_scanner2_sens_move": round(worst_sens, 4), "worst_scanner2_burden_move": round(worst_burden, 4),
                        "usable": bool(usable)}
        print("verdict %s: gain %+.4f (B %+.4f) | worst s2 sens move %.3f burden move %.3f -> %s" % (
            arm, gain, gainB, worst_sens, worst_burden, "USABLE" if usable else "NOT USABLE AS IS"))

    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "cohort_fingerprint": fp, "n": int(len(y)), "positives": int(y.sum()),
        "sizes": SIZES, "draws": DRAWS, "refer_fraction": REFER_FRAC, "shift_scanner": SHIFT_SCANNER,
        "rule": RULE, "verdict": verdict,
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=1), encoding="utf-8")
    print("\ndone in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
