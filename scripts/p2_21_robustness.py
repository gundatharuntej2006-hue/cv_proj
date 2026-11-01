# -*- coding: utf-8 -*-
"""How much does the classifier degrade under the things that actually happen to X-rays?

Every number in the report is measured on clean, uniformly preprocessed TBX11K images. A
deployed screening tool meets rotated films, noisy sensors, JPEG recompression, different
exposure, and occasionally an image flipped by whoever loaded it. None of that has been
measured.

Design: the head is trained **once** on the clean training split and then frozen, and only
the *input* varies. That is the deployment situation — the model is fixed, the world is not.
Training a fresh head per perturbation would instead measure how learnable each perturbation
is, which is a different and less useful question.

Perturbations, all applied after the standard 512px preprocessing:

  rotation      +-5, +-10 degrees        films are not always straight
  noise         Gaussian sigma 5/10/20  sensor and dose variation
  jpeg          quality 90/70/50        recompression in transit
  gamma         0.8 / 1.25              exposure and display differences
  blur          sigma 1.0 / 2.0         focus and motion
  hflip         left-right mirror       a real data-entry error, and a laterality probe

`hflip` is the interesting one. A chest X-ray is anatomically asymmetric - the heart sits
left, the right lung has three lobes. If accuracy survives a mirror, the model is largely
ignoring laterality, which is informative either way.

    py -3 scripts/p2_21_robustness.py

Development split only (659 images), so each perturbation costs about a minute to embed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import embed as E                                       # noqa: E402
from p2_cls import heads as H                                       # noqa: E402
from p2_cls import registry as R                                    # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

MANIFEST = ROOT / "data/processed/v1/manifest.parquet"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
REF_EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
OUT = ROOT / "artifacts/p2/dryrun/robustness.json"
SEEDS = (17, 29, 43, 61, 79)
SPEC = 0.95
N_BOOT = 2000


def perturbations():
    """name -> function(uint8 HxW) -> uint8 HxW. Ordered roughly by expected severity."""
    import cv2

    def rot(deg):
        def f(a):
            h, w = a.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
            return cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        return f

    def noise(sigma):
        def f(a):
            rng = np.random.default_rng(abs(hash(a.tobytes())) % (2 ** 32))
            return np.clip(a.astype(np.float32) + rng.normal(0, sigma, a.shape),
                           0, 255).astype(np.uint8)
        return f

    def jpeg(q):
        def f(a):
            ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return a
            return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return f

    def gamma(g):
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
        return lambda a: lut[a]

    def blur(sigma):
        return lambda a: cv2.GaussianBlur(a, (0, 0), sigma)

    return {
        "clean": lambda a: a,
        "rot_+5": rot(5), "rot_-5": rot(-5), "rot_+10": rot(10), "rot_-10": rot(-10),
        "noise_5": noise(5), "noise_10": noise(10), "noise_20": noise(20),
        "jpeg_90": jpeg(90), "jpeg_70": jpeg(70), "jpeg_50": jpeg(50),
        "gamma_0.8": gamma(0.8), "gamma_1.25": gamma(1.25),
        "blur_1.0": blur(1.0), "blur_2.0": blur(2.0),
        "hflip": lambda a: a[:, ::-1].copy(),
    }


def main():
    import pandas as pd
    with np.load(REF_EMB) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]

    # 1. the frozen model: trained once, on clean data
    tr = c01[c01.split == "train"]
    dv = c01[c01.split == "development"]
    Xtr, ytr = emb[[idx[i] for i in tr.image_id]], H.binary_labels(tr.category.tolist())
    Xdv, ydv = emb[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist())
    res, mean_auc, sd = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    print("frozen head: clean dev AUC %.4f (seed sd %.4f)\n" % (mean_auc, sd))

    # the operating threshold is fixed on CLEAN development, as a deployed one would be
    p_clean_ref = R.p_tb_from_logits(R.head_logits(Xdv, W, b), 2)
    thr = float(np.quantile(p_clean_ref[ydv == 0], SPEC))

    # 2. re-embed the development images under each perturbation
    variant = E.parse_variant("full_clahe0")
    df, _ = E.load_manifest(MANIFEST, variant)
    path = dict(zip(df.image_id, df.src_path))
    dev_ids = [i for i in dv.image_id if i in path]
    y = H.binary_labels(dv.set_index("image_id").loc[dev_ids].category.tolist())
    print("development %d images (%d TB) | fixed threshold %.4f at %.0f%% clean specificity"
          % (len(dev_ids), int(y.sum()), thr, 100 * SPEC))

    embedder = E.Embedder(batch=16)
    perts = perturbations()
    raw = [E.read_image(path[i], None) for i in dev_ids]

    rows, probs = [], {}
    print("\n  %-12s %8s %9s %9s %12s" % ("perturbation", "AUC", "dAUC", "sens", "spec"))
    t0 = time.perf_counter()
    for name, fn in perts.items():
        imgs = [fn(a) for a in raw]
        vecs = []
        for s in range(0, len(imgs), 16):
            vecs.append(embedder.embed(imgs[s:s + 16]).astype(np.float32))
        X = np.concatenate(vecs)
        p = R.p_tb_from_logits(R.head_logits(X, W, b), 2)
        probs[name] = p
        auc = float(SEL.auc(y, p))
        sens = float((p[y == 1] >= thr).mean())
        spec = float((p[y == 0] < thr).mean())
        rows.append({"perturbation": name, "auc": round(auc, 4),
                     "sensitivity_at_fixed_threshold": round(sens, 4),
                     "specificity_at_fixed_threshold": round(spec, 4)})
        print("  %-12s %8.4f %9s %9.4f %12.4f"
              % (name, auc, "-" if name == "clean" else "%+.4f" % (auc - rows[0]["auc"]),
                 sens, spec))
    print("  (%.1f min)" % ((time.perf_counter() - t0) / 60))

    # 3. paired bootstrap against clean
    print("\npaired bootstrap, dAUC vs clean (%d resamples):" % N_BOOT)
    rng = np.random.default_rng(0)
    for r in rows[1:]:
        p_a, p_b = probs[r["perturbation"]], probs["clean"]
        d = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(y), len(y))
            if len(set(y[k].tolist())) < 2:
                continue
            d.append(SEL.auc(y[k], p_a[k]) - SEL.auc(y[k], p_b[k]))
        d = np.asarray(d)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        r["delta_auc"] = round(r["auc"] - rows[0]["auc"], 4)
        r["ci95"] = [round(lo, 4), round(hi, 4)]
        r["verdict"] = "worse" if hi < 0 else "better" if lo > 0 else "no difference"
        print("  %-12s %+.4f [%+.4f, %+.4f]  %s"
              % (r["perturbation"], r["delta_auc"], lo, hi, r["verdict"]))

    worst = min(rows[1:], key=lambda r: r["auc"])
    sens_drop = min(rows[1:], key=lambda r: r["sensitivity_at_fixed_threshold"])
    print("\nworst AUC: %s (%.4f, %+.4f)" % (worst["perturbation"], worst["auc"],
                                             worst["delta_auc"]))
    print("worst sensitivity at the fixed threshold: %s (%.4f vs %.4f clean)"
          % (sens_drop["perturbation"], sens_drop["sensitivity_at_fixed_threshold"],
             rows[0]["sensitivity_at_fixed_threshold"]))
    significant = [r["perturbation"] for r in rows[1:] if r["verdict"] == "worse"]
    print("significantly worse than clean: %s"
          % (", ".join(significant) if significant else "none"))

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "n_dev": len(dev_ids), "n_dev_tb": int(y.sum()),
           "clean_dev_auc": round(float(mean_auc), 4),
           "fixed_threshold": round(thr, 6), "threshold_specificity": SPEC,
           "results": rows, "significantly_worse": significant,
           "note": ("Head trained once on clean data and frozen; only the input varies, "
                    "as in deployment. Threshold fixed on clean development. Provisional "
                    "split, development only; internal_test untouched.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
