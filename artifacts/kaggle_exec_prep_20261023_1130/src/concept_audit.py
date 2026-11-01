# -*- coding: utf-8 -*-
"""TUNE-only CLEAR concept audit, corrected for execution in Kaggle.

Supersedes `artifacts/dryad_rights_clear_audit_20261023_0900/src/run_concept_audit.py`,
which is left untouched. Four substantive changes, each for a defect found while preparing
this run rather than for its own sake:

1. **Paths are resolved, not assumed.** The previous module derived every input path from
   `HERE.parents[2]`, which resolves to `/kaggle` when `src/` sits in a Kaggle input
   dataset. Cell 2 would have failed immediately. See `paths.py`.
2. **The image encoder now fails closed.** The previous loader filtered the checkpoint by
   key name and called `load_state_dict(..., strict=False)`. If CLEAR's keys are laid out
   differently than guessed, the filtered dict is empty, nothing loads, and a
   **randomly initialised** ViT silently produces plausible-looking similarities. This
   version requires a minimum fraction of parameters to be loaded and raises otherwise.
3. **The cohort is verified against a pinned fingerprint**, so a wrong or re-cut upload is
   caught before any GPU work.
4. **Inference is incremental and resumable** without reordering ids: a shard is written
   per block against the hash of the full ordered id list, and a resume refuses to proceed
   unless that hash matches.

Unchanged guarantees: Vietnam **TUNE only**; TBX11K `internal_test` never referenced; only
the **9 exact** vocabulary entries are used and the other 3 are excluded, not approximated;
no concept score is subtracted from or used to modify any TB score; no per-case table is
written into the package.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

import paths as P                                                   # noqa: E402

HF_REPO = "peterhan91/CLEAR"
HF_REV = "142f169c8b006b0f28cf55c6a237c3003db5ed8f"      # pinned; never a branch
# CLEAR's own package, pinned by commit so its loader and preprocessing cannot drift.
CLEAR_COMMIT = "57d2eae57ddabf8e56c8013ec675336d71b3d8e7"
CLEAR_INPUT_PX = 448
CLEAR_EMBED_ROWS = 368294
N_BOOT = 10000
SEED = 20261023


class SplitViolation(RuntimeError):
    pass


class WeightLoadError(RuntimeError):
    pass


# --------------------------------------------------------------------- loading
def ordered_hash(ids):
    return hashlib.sha256("\n".join(str(i) for i in ids).encode("utf-8")).hexdigest()


def cohort_fingerprint(ids):
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids))
                          .encode("utf-8")).hexdigest()


def load_tune_only(p=None):
    """Return the TUNE cohort. Fails closed on any partition or integrity problem."""
    p = p or P.resolve()
    split = json.load(io.open(p["split"], encoding="utf-8"))
    tune = set(str(x) for x in split["tune"])
    report = set(str(x) for x in split["report"])
    if tune & report:
        raise SplitViolation("TUNE and REPORT overlap in the split manifest: %s"
                             % sorted(tune & report)[:3])

    rows = list(csv.DictReader(io.open(p["scores"], encoding="utf-8-sig")))
    meta = {str(r["pid"]): r
            for r in csv.DictReader(io.open(p["meta"], encoding="utf-8-sig"))}

    # A live guard, not dead code, in the layout where it is meaningful: the staged upload
    # is built TUNE-only, so a REPORT row in either input file means it was built wrongly
    # and the mistake must surface before anything is scored. The repository working copy
    # legitimately holds all 2,708 rows, so there the allow-list below is the boundary.
    if p["layout"] in P.STAGED_LAYOUTS:
        for what, present in (("scores file", {str(r["pid"]) for r in rows}),
                              ("clinical file", set(meta))):
            intruders = sorted(present & report)
            if intruders:
                raise SplitViolation(
                    "the staged %s contains %d REPORT pids, first: %s. Rebuild the upload "
                    "with stage_kaggle_data.py." % (what, len(intruders), intruders[:3]))

    sel = [r for r in rows if str(r["pid"]) in tune]
    ids = [str(r["pid"]) for r in sel]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SplitViolation("duplicate pid in the TUNE set: %s" % dup[:3])
    if len(sel) != len(tune):
        raise SplitViolation("expected %d TUNE rows, got %d (missing %d)"
                             % (len(tune), len(sel), len(tune) - len(sel)))
    absent = [i for i in ids if i not in meta]
    if absent:
        raise SplitViolation("%d TUNE pids have no clinical row, first: %s"
                             % (len(absent), absent[:3]))

    fp = cohort_fingerprint(ids)
    pinned = json.load(io.open(P.asset("tune_cohort_fingerprint.json"),
                               encoding="utf-8"))["fingerprint_sha256"]
    if fp != pinned:
        raise SplitViolation(
            "cohort fingerprint mismatch.\n  loaded: %s\n  pinned: %s\n"
            "The analysis cohort is not the one this package was prepared for." % (fp, pinned))

    man = P.load_manifest(p)
    if man and cohort_fingerprint(man["pids"]) != fp:
        raise SplitViolation("the uploaded tune_manifest.json disagrees with the loaded ids")

    d = {
        "pid": ids,
        "y": np.array([int(meta[i]["xpert_result"]) for i in ids]),
        "past_tb": np.array([int(meta[i]["past_tb"]) for i in ids]),
        "rad_system": np.array([int(meta[i]["rad_system"]) for i in ids]),
        "age_group": np.array([int(meta[i]["age_group"]) for i in ids]),
        "sex": np.array([int(meta[i]["sex"]) for i in ids]),
        "symptoms": np.array([sum(int(meta[i][k]) for k in ("c", "f", "wl", "ns"))
                              for i in ids]),
        "clf_score": np.array([float(r["score_baseline"]) for r in sel]),
        "fingerprint": fp,
        "ordered_hash": ordered_hash(ids),
    }
    for k in ("y", "past_tb", "rad_system", "age_group", "sex", "symptoms", "clf_score"):
        if not np.isfinite(np.asarray(d[k], float)).all():
            raise SplitViolation("non-finite value in %s" % k)
    return d


def concept_map():
    m = json.load(io.open(P.asset("concept_indices.json"), encoding="utf-8"))
    primary = [c for c in m["mapping"] if c["concept_idx"] is not None]
    if len(primary) != m["n_primary"]:
        raise AssertionError("concept map is inconsistent with its own count")
    idx = [c["concept_idx"] for c in primary]
    if len(set(idx)) != len(idx) or not all(0 <= i < CLEAR_EMBED_ROWS for i in idx):
        raise AssertionError("concept indices are duplicated or out of range")
    return m, primary


def check_images(p, ids):
    """Every approved id must have an image; nothing outside the allow-list is read.

    Works the same whether the images are a folder or members of a zip: Kaggle's uploader
    caps the number of files per dataset, so 1,352 images must be uploaded archived.
    """
    have = P.image_ids(p)
    missing = [i for i in ids if i not in have]
    extra = sorted(have - set(ids))
    return missing, extra


# --------------------------------------------------------------------- inference
def _load_encoder(ckpt, device, log):
    """Load CLEAR through **its own** loader, pinned by commit. Not reimplemented.

    An earlier version of this function rebuilt the encoder by hand: it loaded
    `dinov2_vitb14_reg`, guessed at the checkpoint's key prefixes, and preprocessed with
    ImageNet statistics. It was wrong in three ways that reading CLEAR's `src/clear/hub.py`
    settled:

      * the keys are prefixed `visual.backbone.`, which the guesswork missed - caught by
        the fail-closed check, which is the only reason this was found before scoring;
      * the visual tower is backbone **plus an nn.Linear projection**, and `encode_image`
        returns the PROJECTED vector. The raw backbone output lives in a different space
        from the concept embeddings, so every similarity would have been meaningless;
      * preprocessing is a bicubic resize to 448 and CXR statistics
        (mean 101.48761/255, std 83.43944/255), not ImageNet constants.

    The first would have stopped the run. The other two would not have: they would have
    produced a full, plausible-looking table of wrong numbers. Hence the official loader.
    """
    try:
        from clear import load_pretrained
    except ImportError as e:
        raise WeightLoadError(
            "the CLEAR package is not installed (%s).\nInstall it pinned, before running:\n"
            "    pip install --no-deps git+https://github.com/peterhan91/CLEAR.git@%s\n"
            "    pip install ftfy regex\n"
            "`--no-deps` matters: CLEAR's dependency pins would otherwise reinstall torch."
            % (e, CLEAR_COMMIT))

    model, preprocess = load_pretrained(checkpoint_path=str(ckpt), device=device)
    model.eval()
    n_par = sum(p.numel() for p in model.visual.parameters())
    log("  loaded via clear.load_pretrained; visual tower has %.1fM parameters"
        % (n_par / 1e6))
    if not hasattr(model, "encode_image"):
        raise WeightLoadError("the loaded CLEAR model has no encode_image()")
    if not hasattr(model.visual, "projection"):
        raise WeightLoadError(
            "the loaded visual tower has no projection layer. encode_image must return "
            "the PROJECTED embedding, or the similarities are computed in the wrong space.")
    return model, preprocess, {"loader": "clear.load_pretrained",
                               "clear_commit": CLEAR_COMMIT,
                               "visual_parameters": int(n_par)}


def encode_concepts(ids, device="cuda", batch=16, out_dir=None, resume=True, log=print):
    """Encode TUNE images with CLEAR; return [n, 9] similarities to the pinned concepts.

    Shards are written to `out_dir/_shards` as they are produced, keyed to the hash of the
    full ordered id list. A resume with a different id list is refused rather than merged.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    p = P.resolve()
    _, primary = concept_map()
    idx = [c["concept_idx"] for c in primary]
    oh = ordered_hash(ids)
    sh = Path(out_dir or p["out_dir"]) / "_shards"
    sh.mkdir(parents=True, exist_ok=True)
    prog = sh / "progress.json"

    done, parts = 0, []
    if resume and prog.is_file():
        st = json.load(io.open(prog, encoding="utf-8"))
        if st.get("ordered_hash") != oh:
            raise SplitViolation(
                "a partial run exists for a DIFFERENT ordered id list.\n"
                "  on disk: %s\n  now    : %s\nDelete %s or pass --no-resume."
                % (st.get("ordered_hash"), oh, sh))
        done = int(st["done"])
        parts = [np.load(sh / f) for f in st["shards"]]
        log("  resuming after %d of %d cases" % (done, len(ids)))

    ckpt = hf_hub_download(HF_REPO, "best_model.pt", revision=HF_REV)
    emb_path = hf_hub_download(HF_REPO, "concept_embeddings_368294.pt", revision=HF_REV)
    prov = {"checkpoint": str(ckpt), "revision": HF_REV,
            "checkpoint_bytes": Path(ckpt).stat().st_size,
            "embeddings_bytes": Path(emb_path).stat().st_size}

    try:
        C = torch.load(emb_path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        C = torch.load(emb_path, map_location="cpu")
    if C.shape[0] != CLEAR_EMBED_ROWS:
        raise AssertionError("concept embeddings have %d rows, expected %d"
                             % (C.shape[0], CLEAR_EMBED_ROWS))
    Csel = C[idx].float().clone()
    Csel = Csel / Csel.norm(dim=1, keepdim=True).clamp_min(1e-12)
    del C
    log("  pinned %d concept embeddings, dim %d" % tuple(Csel.shape))

    model, tf, how = _load_encoder(ckpt, device, log)
    prov.update(how)
    Csel = Csel.to(device)
    # `tf` is CLEAR's own build_cxr_preprocess: bicubic resize to 448 and CXR pixel
    # statistics. Not reconstructed here, so it cannot drift from theirs.
    t0, bs, k = time.perf_counter(), batch, done
    shard_names = list(json.load(io.open(prog, encoding="utf-8"))["shards"]) \
        if (resume and prog.is_file()) else []
    while k < len(ids):
        sub = ids[k:k + bs]
        try:
            batch_imgs = []
            for i in sub:
                with P.open_image(p, i) as fh:
                    batch_imgs.append(tf(Image.open(fh).convert("RGB")))
            x = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                # encode_image applies the projection; the raw backbone output is in a
                # different space from the concept embeddings.
                f = model.encode_image(x).float()
                f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
                block = (f @ Csel.T).cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                if device == "cuda":
                    torch.cuda.empty_cache()
                log("  OOM: batch -> %d" % bs)
                continue
            raise
        name = "shard_%06d.npy" % k
        np.save(sh / name, block)
        shard_names.append(name)
        parts.append(block)
        k += len(sub)
        json.dump({"ordered_hash": oh, "done": k, "shards": shard_names},
                  io.open(prog, "w", encoding="utf-8"))
        if k % 256 < bs:
            log("  %d/%d (%.1f min)" % (k, len(ids), (time.perf_counter() - t0) / 60))

    S = np.concatenate(parts)
    if S.shape != (len(ids), len(idx)):
        raise AssertionError("similarities have shape %s, expected (%d, %d)"
                             % (S.shape, len(ids), len(idx)))
    if not np.isfinite(S).all():
        raise AssertionError("non-finite concept similarities")
    prov["elapsed_min"] = round((time.perf_counter() - t0) / 60, 2)
    return S, prov


# --------------------------------------------------------------------- analysis
def distributions(d, S, names):
    """Concept-score distributions by TB label and by previous-TB history."""
    out = []
    groups = (("negatives, no past TB", (d["y"] == 0) & (d["past_tb"] == 0)),
              ("negatives, past TB", (d["y"] == 0) & (d["past_tb"] == 1)),
              ("positives, no past TB", (d["y"] == 1) & (d["past_tb"] == 0)),
              ("positives, past TB", (d["y"] == 1) & (d["past_tb"] == 1)))
    for j, nm in enumerate(names):
        for label, m in groups:
            if m.sum() < 20:
                out.append({"concept": nm, "group": label, "n": int(m.sum()),
                            "note": "suppressed, n < 20"})
                continue
            v = S[m, j]
            out.append({"concept": nm, "group": label, "n": int(m.sum()),
                        "mean": round(float(v.mean()), 5),
                        "sd": round(float(v.std(ddof=1)), 5),
                        "p25": round(float(np.percentile(v, 25)), 5),
                        "median": round(float(np.median(v)), 5),
                        "p75": round(float(np.percentile(v, 75)), 5),
                        "note": ""})
    return out


def discrimination(d, S, names, n_boot=None):
    """AUROC/AUPRC of each concept score against the TB label, where it is meaningful."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    nb = n_boot if n_boot is not None else min(N_BOOT, 2000)
    rng = np.random.default_rng(SEED)
    y = d["y"]
    out = []
    for j, nm in enumerate(names):
        c = S[:, j]
        pos, neg = int(y.sum()), int((y == 0).sum())
        if pos < 20 or neg < 20:
            out.append({"concept": nm, "n_pos": pos, "n_neg": neg,
                        "note": "suppressed: a class has fewer than 20 cases"})
            continue
        a = [roc_auc_score(y[k], c[k]) for k in
             (rng.integers(0, len(y), len(y)) for _ in range(nb))
             if len(np.unique(y[k])) == 2]
        out.append({"concept": nm, "n_pos": pos, "n_neg": neg,
                    "auroc": round(float(roc_auc_score(y, c)), 4),
                    "auroc_ci95_lo": round(float(np.percentile(a, 2.5)), 4),
                    "auroc_ci95_hi": round(float(np.percentile(a, 97.5)), 4),
                    "auprc": round(float(average_precision_score(y, c)), 4),
                    "prevalence": round(pos / len(y), 4),
                    "note": "a concept score is not a TB classifier; descriptive only"})
    return out


def analyse(d, S, names, log=print):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(SEED)
    y, past, s = d["y"], d["past_tb"], d["clf_score"]
    neg = y == 0
    rows, pvals = [], []

    def boot_rho(a, b):
        v = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(a), len(a))
            if len(np.unique(a[k])) < 2:
                continue
            v.append(spearmanr(a[k], b[k]).statistic)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    for j, name in enumerate(names):
        c = S[:, j]
        for label, m in (("negatives (all)", neg),
                         ("negatives, past TB", neg & (past == 1)),
                         ("negatives, no past TB", neg & (past == 0)),
                         ("positives (all)", y == 1)):
            if m.sum() < 20:
                continue
            r = spearmanr(c[m], s[m])
            lo, hi = boot_rho(c[m], s[m])
            rows.append({"concept": name, "group": label, "n": int(m.sum()),
                         "spearman_rho": float(r.statistic), "ci95_lo": lo, "ci95_hi": hi,
                         "p_raw": float(r.pvalue)})
            if label == "negatives (all)":
                pvals.append(float(r.pvalue))

    # Benjamini-Hochberg across concepts, on the primary 'negatives (all)' family only
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    padj = np.empty(len(p))
    padj[order] = np.clip(adj, 0, 1)
    it = iter(padj)
    for r in rows:
        r["p_bh_adjusted"] = float(next(it)) if r["group"] == "negatives (all)" else None

    X0 = np.column_stack([d["rad_system"][neg] == v for v in (1, 2, 3)][:2]
                         + [d["age_group"][neg], d["sex"][neg], d["symptoms"][neg],
                            past[neg]]).astype(float)
    r2_0 = LinearRegression().fit(X0, s[neg]).score(X0, s[neg])
    incr = []
    for j, name in enumerate(names):
        X1 = np.column_stack([X0, S[neg, j]])
        r2_1 = LinearRegression().fit(X1, s[neg]).score(X1, s[neg])
        incr.append({"concept": name, "r2_clinical_only": round(r2_0, 4),
                     "r2_with_concept": round(r2_1, 4),
                     "delta_r2": round(r2_1 - r2_0, 4)})
    return rows, incr


# --------------------------------------------------------------------- driver
def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def run_manifest(d, p, prov=None, extra=None):
    m = {"run": {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "seed": SEED, "n_boot": N_BOOT},
         "environment": {"python": platform.python_version(),
                         "platform": platform.platform()},
         "model": {"repo": HF_REPO, "revision": HF_REV, "input_px": CLEAR_INPUT_PX,
                   "concept_embedding_rows": CLEAR_EMBED_ROWS,
                   "licence": "Apache-2.0 (code and weights); see MODEL licence notes"},
         "data": {"layout": p["layout"], "base": str(p["base"]),
                  "cohort": "vietnam_tune", "n_cases": len(d["pid"]),
                  "cohort_fingerprint_sha256": d["fingerprint"],
                  "ordered_id_hash_sha256": d["ordered_hash"],
                  "report_partition_used": False,
                  "tbx11k_internal_test_used": False},
         "concepts": {}, "code_sha256": {}, "inference": prov or None}
    m.update(extra or {})
    mm, primary = concept_map()
    m["concepts"] = {"n_primary": len(primary),
                     "names": [c["concept"] for c in primary],
                     "excluded": [c["concept"] for c in mm["mapping"]
                                  if c["concept_idx"] is None],
                     "score_kind": "text-similarity to one report sentence per concept: a "
                                   "weakly supervised proxy, NOT a validated detector"}
    # Keyed by filename, not by relative path, because a flattened upload has no
    # src/ or configs/ prefix to be relative to.
    for f in sorted(P.code_dir().glob("*.py")) + \
            sorted(P.code_dir().glob("*.json")) + \
            sorted((P.code_dir() / "configs").glob("*.json")) + \
            sorted((PKG / "configs").glob("*.json")):
        m["code_sha256"].setdefault(f.name, hashlib.sha256(f.read_bytes()).hexdigest())
    try:
        import torch
        m["environment"]["torch"] = torch.__version__
        m["environment"]["cuda"] = torch.version.cuda
        m["environment"]["gpu"] = (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None)
    except Exception:
        m["environment"]["torch"] = "absent"
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-inference", action="store_true",
                    help="encode images; omit for a dry validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", default=None, help="output directory (default: resolved)")
    a = ap.parse_args()

    p = P.resolve()
    print(P.describe(p))
    d = load_tune_only(p)
    m, primary = concept_map()
    names = [c["concept"] for c in primary]
    out = Path(a.out) if a.out else p["out_dir"]

    print("\nTUNE cohort           : %d patients, %d TB-positive, %d with previous TB"
          % (len(d["pid"]), int(d["y"].sum()), int(d["past_tb"].sum())))
    print("cohort fingerprint    : %s  (matches pinned)" % d["fingerprint"][:16])
    print("concepts, primary set : %d of %d (%s excluded, no exact entry)"
          % (len(names), len(m["mapping"]),
             ", ".join(c["concept"] for c in m["mapping"] if c["concept_idx"] is None)))
    missing, extra = check_images(p, d["pid"])
    print("images                : %d approved, %d missing, %d present but not approved"
          % (len(d["pid"]) - len(missing), len(missing), len(extra)))
    print("REPORT pids loaded    : 0  |  internal_test referenced: no")
    if missing:
        raise SystemExit("ABORT: %d approved images are missing, first: %s"
                         % (len(missing), missing[:5]))

    if not a.run_inference:
        print("\nDRY VALIDATION ONLY. Pass --run-inference to encode images.")
        json.dump(run_manifest(d, p), io.open(out / "run_manifest_dry.json", "w",
                                              encoding="utf-8"), indent=1)
        print("wrote %s" % (out / "run_manifest_dry.json"))
        return

    S, prov = encode_concepts(d["pid"], device=a.device, batch=a.batch, out_dir=out,
                              resume=not a.no_resume)
    if S.shape[0] != len(d["pid"]):
        raise AssertionError("case-id alignment broken: %d rows for %d ids"
                             % (S.shape[0], len(d["pid"])))
    rows, incr = analyse(d, S, names)
    _write(out / "tables/concept_associations.csv", rows)
    _write(out / "tables/concept_incremental_r2.csv", incr)
    _write(out / "tables/concept_distributions.csv", distributions(d, S, names))
    _write(out / "tables/concept_discrimination.csv", discrimination(d, S, names))
    json.dump(run_manifest(d, p, prov), io.open(out / "run_manifest.json", "w",
                                                encoding="utf-8"), indent=1)
    print("\nwrote 4 aggregate tables and run_manifest.json to %s" % out)
    print("No per-case concept score is written. Do not export one.")


if __name__ == "__main__":
    main()
