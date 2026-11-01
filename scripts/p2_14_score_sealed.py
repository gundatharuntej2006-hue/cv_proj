# -*- coding: utf-8 -*-
"""P2-14 driver: seal a model, then score the held-out internal test split exactly once.

Three subcommands, run at three different times by two different people.

    # 1. Person 2, after D08/D09/D11 are locked and before the test split is touched:
    py -3 scripts/p2_14_score_sealed.py seal --seal-id primary_v1

    # 2. Person 4, after reviewing the seal (needs the signing key, which P2 does not have):
    py -3 scripts/p2_14_score_sealed.py issue --seal-id primary_v1 --ttl 24

    # 3. Person 2, once, with the token:
    py -3 scripts/p2_14_score_sealed.py score --seal-id primary_v1 \
        --token artifacts/p2/sealed/gate_primary_v1.json

`score` refuses unless the token is valid for this exact configuration and the seal has not
been scored before. It reads the internal_test embeddings only after every check passes.

The key is read from TB_SEAL_KEY (or --key-file). It must not live in the repository: the
whole arrangement depends on Person 2 not holding it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import registry as R                                   # noqa: E402
from p2_cls import sealed as SL                                     # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

SEAL_DIR = ROOT / "artifacts/p2/sealed"
BUNDLE = ROOT / "artifacts/p2/bundle"
DEFAULT_SPLIT = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
SPEC_REPORT = 0.70          # WHO screening operating point — the headline number
SPEC_SELECT = 0.95          # the specificity the criterion was amended to select at


def get_key(args):
    if args.key_file:
        return Path(args.key_file).read_text(encoding="utf-8").strip()
    k = os.environ.get("TB_SEAL_KEY", "").strip()
    if not k:
        raise SystemExit("no signing key: set TB_SEAL_KEY or pass --key-file. Person 2 is "
                         "not supposed to have it; ask the key holder to run `issue`.")
    return k


def load_bundle_entry():
    meta = json.load(open(BUNDLE / "bundle.json", encoding="utf-8"))
    entry = R.RegistryEntry(**meta["model"])
    entry.weight_path = str(BUNDLE / "head.npz")
    entry.weight_sha256 = SL.file_sha256(entry.weight_path)
    thr = {k: meta.get(k) for k in
           ("temperature", "t_lower", "t_upper", "ood_threshold", "alpha_fnr", "ood_alpha")}
    return meta, entry, thr


# ------------------------------------------------------------------------- subcommands
def cmd_seal(args):
    meta, entry, thr = load_bundle_entry()
    split = Path(args.split or DEFAULT_SPLIT)
    if not split.exists():
        raise SystemExit("split file not found: %s" % split)
    if meta.get("provisional", True) and not args.allow_provisional:
        raise SystemExit("this bundle is marked provisional (not Person 1's C01, and the "
                         "thresholds are not Person 4's fitted values). Sealing it would "
                         "spend the one-shot test read on a model that is not final. "
                         "Pass --allow-provisional only for a rehearsal.")
    s = SL.seal(args.seal_id, entry, thr, split, SEAL_DIR, note=args.note)
    print("sealed %s" % s.seal_id)
    print("  model      %s (%s)" % (s.model_id, s.variant))
    print("  weights    %s" % s.weight_sha256[:16])
    print("  split      %s" % s.split_sha256[:16])
    print("  thresholds %s" % json.dumps(s.thresholds))
    print("  config_sha %s" % s.config_sha)
    print("\nSend the config_sha to the key holder. Nothing is scored until they issue a "
          "token for it.")


def cmd_issue(args):
    s = SL.load_seal(SEAL_DIR / ("seal_%s.json" % args.seal_id))
    if SL.already_scored(s.seal_id, SEAL_DIR):
        raise SystemExit("seal %r was already scored; issuing another token would "
                         "authorise a second read of the held-out split." % s.seal_id)
    t = SL.issue_token(s, get_key(args), ttl_hours=args.ttl, issuer=args.issuer)
    p = SL.save_token(t, SEAL_DIR / ("gate_%s.json" % args.seal_id))
    print("token issued for %s (config %s), valid %g h" % (s.seal_id, s.config_sha[:12], args.ttl))
    print("  %s" % p)


def cmd_score(args):
    meta, entry, _ = load_bundle_entry()
    s = SL.load_seal(SEAL_DIR / ("seal_%s.json" % args.seal_id))
    token = SL.load_token(args.token) if args.token else None

    def score_fn():
        """Only ever called after the gate passes. This is the single test-split read."""
        import pandas as pd
        emb_path = ROOT / ("artifacts/p2/emb/radino_tbx11k_%s_c04_v1.npz" % s.variant)
        if not emb_path.exists():
            emb_path = ROOT / ("artifacts/p2/emb/radino_tbx11k_%s_v1.npz" % s.variant)
        with np.load(emb_path) as z:
            ids, emb = z["image_id"].tolist(), z["emb"].astype(np.float32)
        idx = {i: k for k, i in enumerate(ids)}
        c01 = pd.read_csv(args.split or DEFAULT_SPLIT)
        g = c01[(c01.split == "internal_test") & (c01.image_id.isin(idx))]
        if g.empty:
            raise SystemExit("no internal_test rows found in %s" % (args.split or DEFAULT_SPLIT))
        from p2_cls import heads as H
        X = emb[[idx[i] for i in g.image_id]]
        y = H.binary_labels(g.category.tolist())

        p_raw = R.classify_embeddings(entry, X)
        T = s.thresholds.get("temperature") or 1.0
        lg = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
        p = 1 / (1 + np.exp(-lg / T))

        tl, tu = s.thresholds.get("t_lower"), s.thresholds.get("t_upper")
        screen_pos = p >= (tl if tl is not None else 0.5)
        sens = float(screen_pos[y == 1].mean())
        spec = float((~screen_pos)[y == 0].mean())
        lo, hi = SEL.wilson_ci(int(screen_pos[y == 1].sum()), int((y == 1).sum()))

        def at(spec):
            sens, thr, n_pos, tp = SEL.sensitivity_at_specificity(y, p, spec)
            cl, ch = SEL.wilson_ci(tp, n_pos)
            return {"sensitivity": round(float(sens), 4),
                    "ci95": [round(cl, 4), round(ch, 4)],
                    "threshold": round(float(thr), 6)}

        return {
            "n": int(len(y)), "n_tb": int((y == 1).sum()),
            "auc": round(float(SEL.auc(y, p)), 4),
            "sens_at_spec_%.2f" % SPEC_REPORT: at(SPEC_REPORT),
            "sens_at_spec_%.2f" % SPEC_SELECT: at(SPEC_SELECT),
            "operating_point": {"t_lower": tl, "t_upper": tu,
                                "sensitivity": round(sens, 4),
                                "sensitivity_ci95": [round(lo, 4), round(hi, 4)],
                                "specificity": round(spec, 4)},
            "triage_counts": {
                "TB": int((p >= (tu if tu is not None else 1.01)).sum()),
                "NOT_TB": int((p < (tl if tl is not None else 0.0)).sum()),
            },
        }

    try:
        res = SL.score_sealed(s, token, get_key(args), score_fn, SEAL_DIR,
                              weight_path=entry.weight_path)
    except SL.SealError as e:
        print("REFUSED: %s" % e)
        return 2
    print("scored %s -> %s" % (s.seal_id, res["path"]))
    print(json.dumps(res["metrics"], indent=1))
    print("\nLedger row written. This seal cannot be scored again.")


def cmd_status(args):
    seals = sorted(SEAL_DIR.glob("seal_*.json"))
    rows = {r["seal_id"]: r for r in SL.ledger_entries(SEAL_DIR)}
    if not seals:
        print("no seals yet (%s)" % SEAL_DIR)
        return
    for p in seals:
        s = SL.load_seal(p)
        r = rows.get(s.seal_id)
        print("%-14s %-18s %s" % (s.seal_id, s.variant,
                                  ("SCORED %s" % r["scored_utc"]) if r else "sealed, unscored"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("seal", help="freeze a model + thresholds + split (Person 2)")
    a.add_argument("--seal-id", required=True)
    a.add_argument("--split")
    a.add_argument("--note", default="")
    a.add_argument("--allow-provisional", action="store_true",
                   help="rehearsal only: seal a bundle that is not final")
    a.set_defaults(fn=cmd_seal)

    b = sub.add_parser("issue", help="sign a one-shot gate token (key holder)")
    b.add_argument("--seal-id", required=True)
    b.add_argument("--ttl", type=float, default=24.0)
    b.add_argument("--issuer", default="person4")
    b.add_argument("--key-file")
    b.set_defaults(fn=cmd_issue)

    c = sub.add_parser("score", help="read internal_test once, with a token (Person 2)")
    c.add_argument("--seal-id", required=True)
    c.add_argument("--token")
    c.add_argument("--split")
    c.add_argument("--key-file")
    c.set_defaults(fn=cmd_score)

    d = sub.add_parser("status", help="which seals exist and which have been spent")
    d.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
