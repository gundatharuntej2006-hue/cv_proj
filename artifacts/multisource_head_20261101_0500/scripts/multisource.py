# -*- coding: utf-8 -*-
"""Multi-source regional head: four arms, one fitting path, exactly as PROTOCOL.md.

    py -3 multisource.py           # dry: row counts, guards
    py -3 multisource.py --run
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import p2_32_mil_sweep as S                                         # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

REG = ROOT / "artifacts/p2/regional"
CALEMB = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/regional_tbx11k_calibration_lungs_g12_v1.npz"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
VSPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VCSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 4000
BOOT_SEED = 0
MIN_EFFECT = 0.02
G = 12
TB = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE = ("active_tb", "active_latent_tb")
REPRO = {"A": 0.9498, "B1": 0.9460}      # R12 / R17 calibration AUROC vs active_tb_label


class Blocker(SystemExit):
    pass


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    p = TAB / "protocol_hash.txt"
    if p.is_file() and p.read_text(encoding="utf-8").split()[0] != h:
        raise Blocker("PROTOCOL.md changed after hashing")
    p.write_text(h + "  PROTOCOL.md\n", encoding="utf-8")
    return h


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def nlm_label(image_id):
    """'shz__CHNCXR_0001_0' -> 0 ; 'mcu__MCUCXR_0100_1' -> 1."""
    v = image_id.rsplit("_", 1)[1]
    if v not in ("0", "1"):
        raise Blocker("unexpected NLM id %s" % image_id)
    return int(v)


def train(Xtr, ytr, Xdv, ydv, seed, epochs=120, lr=3e-3, bs=256, wd=1e-4):
    """Identical to the R17 trainer with lambda = 0 (which reproduced production)."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.tensor(Xdv, device=dev)
    lin = torch.nn.Linear(xt.shape[2], 1).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    prior = float(yt.mean())
    adj = float(np.log(max(prior, 1e-6) / max(1 - prior, 1e-6)))
    n = len(yt)
    best, best_state, patience = -1.0, None, 0
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        lin.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            z = lin(xt[k].float()).squeeze(-1)
            loss = bce(z.max(1).values - adj, yt[k])
            loss.backward()
            opt.step()
        lin.eval()
        with torch.no_grad():
            sv = torch.cat([lin(xv[i:i + 512].float()).squeeze(-1)
                            for i in range(0, len(xv), 512)]).cpu().numpy()
        a = SEL.auc(ydv, sv.max(1))
        if a > best + 1e-5:
            best, patience = a, 0
            best_state = {k2: v.detach().clone() for k2, v in lin.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break
    lin.load_state_dict(best_state)
    return lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(), best


def region_logits(X, W, b, chunk=512):
    return np.concatenate([(X[i:i + chunk].astype(np.float32) @ W.T + b)[:, :, 0]
                           for i in range(0, len(X), chunk)])


def boot_auc(y, p, rng, n=N_BOOT):
    v = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        v.append(SEL.auc(y[k], p[k]))
    return float(SEL.auc(y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired(y, pa, pb, rng, n=N_BOOT):
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], pb[k]) - SEL.auc(y[k], pa[k]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(SEL.auc(y, pb) - SEL.auc(y, pa)), float(lo), float(hi)


def auprc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def thr_at_spec(y, s, spec):
    neg = np.sort(s[y == 0])
    return float(neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)])


def verdict(delta, lo, hi):
    if (hi - lo) / 2 > 0.05:
        return "INCONCLUSIVE"
    return "ACCEPT" if (delta >= MIN_EFFECT and lo > 0) else "REJECT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    t0 = time.perf_counter()
    proto = check_protocol()

    c01 = pd.read_csv(C01)
    forbidden = set(c01[c01.split == "internal_test"].image_id)
    with np.load(REG / "regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    with np.load(CALEMB) as z:
        cid, CX = z["image_id"].tolist(), z["reg"]
    with np.load(REG / "regional_nlm_lungs_g12_v1.npz") as z:
        nid, NX = z["image_id"].tolist(), z["reg"]
    if forbidden & (set(tid) | set(cid)):
        raise Blocker("internal_test present in a feature file")
    ti, ci = {i: k for k, i in enumerate(tid)}, {i: k for k, i in enumerate(cid)}
    tr = c01[(c01.split == "train") & c01.image_id.isin(ti) & c01.category.isin(TB + ("healthy",))]
    dv = c01[(c01.split == "development") & c01.image_id.isin(ti)]
    cal = c01[(c01.split == "calibration") & c01.image_id.isin(ci)]
    assert (len(tr), len(dv), len(cal)) == (3241, 659, 899)
    y_nlm = np.array([nlm_label(i) for i in nid])
    site = np.array([i.split("__")[0] for i in nid])
    assert len(nid) == 800 and set(site) == {"shz", "mcu"}
    counts = {s: {"n": int((site == s).sum()), "tb": int(y_nlm[site == s].sum())} for s in ("shz", "mcu")}
    print("TBX11K train %d | dev %d | cal %d | NLM %s" % (len(tr), len(dv), len(cal), counts))
    (TAB / "training_sets.json").write_text(json.dumps({
        "tbx11k_train": int(len(tr)), "tbx11k_train_tb": int(tr.category.isin(TB).sum()),
        "tbx11k_train_active": int(tr.category.isin(ACTIVE).sum()),
        "nlm": counts, "development": int(len(dv)), "calibration": int(len(cal)),
        "nlm_in_development": False}, indent=1), encoding="utf-8")
    if not a.run:
        print("\nDRY. Pass --run.")
        return 0

    Xtr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, G)
    Xnlm = S.repool(NX, 12, G)
    Xdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, G)
    Xcal = S.repool(CX[[ci[i] for i in cal.image_id]], 12, G)
    cat_dv, cat_cal = dv.category.tolist(), cal.category.tolist()
    y_cal = np.array([1 if c in ACTIVE else 0 for c in cat_cal])

    with np.load(REG / "regional_vietnam_lungs_g12_v1.npz") as z:
        vid, VX = z["image_id"].tolist(), z["reg"]
    sp = json.loads(VSPLIT.read_text(encoding="utf-8"))
    tune, report = set(sp["tune"]), set(sp["report"])
    vsel = [k for k, i in enumerate(vid) if int(i) in tune]
    if {int(vid[k]) for k in vsel} & report:
        raise Blocker("REPORT leaked")
    vdf = pd.read_csv(VCSV).set_index("pid")
    y_tune = vdf.loc[[int(vid[k]) for k in vsel], "xpert_result"].to_numpy().astype(int)
    Xtune = S.repool(VX[vsel], 12, G)

    ARMS = {"A": (TB, False), "M": (TB, True), "B1": (ACTIVE, False), "MB1": (ACTIVE, True)}
    out, per_seed = {}, []
    for arm, (pos, with_nlm) in ARMS.items():
        ytr = np.array([1 if c in pos else 0 for c in tr.category])
        ydv = np.array([1 if c in pos else 0 for c in cat_dv])
        X_, y_ = (np.concatenate([Xtr, Xnlm]), np.concatenate([ytr, y_nlm])) if with_nlm else (Xtr, ytr)
        Ws = []
        print("\narm %-3s target=%s rows=%d (pos %d)" % (arm, "tb" if pos == TB else "active", len(y_), y_.sum()))
        for sd in SEEDS:
            W, b, best = train(X_, y_, Xdv, ydv, sd)
            Ws.append((W, b))
            per_seed.append({"arm": arm, "seed": sd, "early_stop_dev_auc": round(best, 4),
                             "cal_auroc_vs_active": round(SEL.auc(y_cal, region_logits(Xcal, W, b).max(1)), 4)})
            print("   seed %2d dev %.4f" % (sd, best))
        out[arm] = {"cal": np.mean([region_logits(Xcal, W, b).max(1) for W, b in Ws], 0),
                    "dv": np.mean([region_logits(Xdv, W, b).max(1) for W, b in Ws], 0),
                    "tune": np.mean([region_logits(Xtune, W, b).max(1) for W, b in Ws], 0)}
    write_csv(TAB / "per_seed.csv", per_seed)

    gate, ok = [], True
    for arm, want in REPRO.items():
        here = SEL.auc(y_cal, out[arm]["cal"])
        p = abs(here - want) <= 5e-4
        ok &= p
        gate.append({"arm": arm, "expected": want, "here": round(here, 4), "pass": p})
        print("gate %-3s expected %.4f here %.4f %s" % (arm, want, here, "pass" if p else "FAIL"))
    write_csv(TAB / "reproduction_gate.csv", gate)
    if not ok:
        raise Blocker("reproduction gate failed")

    y_dv_tb = np.array([1 if c in TB else 0 for c in cat_dv])
    metrics = []
    for arm in ARMS:
        ac, lc, hc = boot_auc(y_cal, out[arm]["cal"], np.random.default_rng(BOOT_SEED))
        at, lt, ht = boot_auc(y_tune, out[arm]["tune"], np.random.default_rng(BOOT_SEED))
        thr = thr_at_spec(y_dv_tb, out[arm]["dv"], 0.90)
        sc, yc = out[arm]["cal"], y_cal
        metrics.append({"arm": arm, "cal_auroc_vs_active": round(ac, 4), "cal_ci": "[%.4f, %.4f]" % (lc, hc),
                        "cal_auprc": round(auprc(yc, sc), 4),
                        "cal_sens_at_dev_spec0.90": round(float((sc[yc == 1] > thr).mean()), 4),
                        "cal_spec_at_dev_spec0.90": round(float((sc[yc == 0] <= thr).mean()), 4),
                        "tune_auroc_EXPLORATORY": round(at, 4), "tune_ci": "[%.4f, %.4f]" % (lt, ht),
                        "tune_sens_at_dev_spec0.90": round(float((out[arm]["tune"][y_tune == 1] > thr).mean()), 4),
                        "tune_spec_at_dev_spec0.90": round(float((out[arm]["tune"][y_tune == 0] <= thr).mean()), 4)})
    write_csv(TAB / "metrics_by_arm.csv", metrics)

    diffs = []
    for a_, b_, primary in (("A", "M", True), ("B1", "MB1", False), ("A", "MB1", False), ("A", "B1", False)):
        dc, lc, hc = paired(y_cal, out[a_]["cal"], out[b_]["cal"], np.random.default_rng(BOOT_SEED))
        dt, lt, ht = paired(y_tune, out[a_]["tune"], out[b_]["tune"], np.random.default_rng(BOOT_SEED))
        v = verdict(dc, lc, hc) if primary else ""
        diffs.append({"contrast": "%s - %s" % (b_, a_), "primary": primary, "cal_delta": round(dc, 4),
                      "cal_ci": "[%+.4f, %+.4f]" % (lc, hc), "tune_delta_EXPLORATORY": round(dt, 4),
                      "tune_ci": "[%+.4f, %+.4f]" % (lt, ht), "verdict": v})
        print("%s - %s: cal %+.4f [%+.4f, %+.4f] | tune %+.4f [%+.4f, %+.4f] %s"
              % (b_, a_, dc, lc, hc, dt, lt, ht, v))
    write_csv(TAB / "paired_differences.csv", diffs)

    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "seeds": SEEDS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "calibration_n": int(len(y_cal)), "calibration_pos": int(y_cal.sum()),
        "tune_n": int(len(y_tune)), "tune_pos": int(y_tune.sum()),
        "partitions": {"vietnam_report_used": False, "tbx11k_internal_test_used": False},
        "production_changed": False, "elapsed_s": round(time.perf_counter() - t0, 1)}, indent=1),
        encoding="utf-8")
    print("\ndone in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
