# -*- coding: utf-8 -*-
"""P2-03: RAD-DINO embedding extraction. Produces C07.

Acceptance criteria from the work plan:
  "Re-run cosine >= 0.9999 per image; 11,200 images <= 60 min on laptop; output validates"

Three properties the criteria are really asking for, and how each is obtained:

**Deterministic.** Images are processed in sorted `image_id` order, in fixed-size shards
cut by index, and batches are fixed slices inside a shard. So the same image always lands
in the same batch with the same neighbours. That matters more than it sounds: in fp16 a
different batch size can select a different kernel and shift the last decimal, so
"deterministic" and "resumable" are the same requirement. TF32 and cuDNN autotuning are
off for the same reason.

**Resumable.** Each shard is written to `<out>.partial/shard_XXXXX.npz` the moment it is
finished, written to a temporary name and renamed, so a crash mid-shard cannot leave a
half-file. A resumed run skips completed shards and, because shard boundaries are a
function of the index rather than of what is left to do, reproduces identical batches.

**Honest about its input.** The real pipeline reads Person 1's C04 image store. Before
C04 exists, `--raw-dir` lets P2 resize raw images itself so the Week 2 dedup is not
blocked. That path stamps `preproc_version="v0-p2-standin"` into the output, because P2's
resize is not guaranteed to be P1's and the embeddings are therefore not the real C07.
Re-run against C04 when it lands; the stamp is how anyone notices it has not been.
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p2_cls import contracts as K

MODEL_ID = "microsoft/rad-dino"
REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"      # D15
SHARD = 512
REAL_PREPROC = "v1"
STANDIN_PREPROC = "v0-p2-standin"


# ----------------------------------------------------------------- variant parsing
@dataclass(frozen=True)
class Variant:
    kind: str            # "full" | "crop"
    margin: float | None
    clahe: bool

    @property
    def name(self):
        head = "full" if self.kind == "full" else "crop_m%.2f" % self.margin
        return "%s_clahe%d" % (head, int(self.clahe))


def parse_variant(s):
    """'full_clahe0', 'crop_m0.10_clahe1' -> Variant. Rejects anything else loudly: a
    typo here silently produces a file that looks right and contains the wrong pixels."""
    m = re.fullmatch(r"(full|crop_m(\d\.\d{2}))_clahe([01])", s)
    if not m:
        raise ValueError(
            "bad variant %r. Expected 'full_clahe{0,1}' or 'crop_m<margin>_clahe{0,1}', "
            "e.g. crop_m0.10_clahe0" % s)
    kind = "full" if m.group(1) == "full" else "crop"
    return Variant(kind, None if kind == "full" else float(m.group(2)), m.group(3) == "1")


# ----------------------------------------------------------------- input resolution
def load_manifest(manifest_path, variant, roi_path=None, raw_dir=None):
    """Returns (DataFrame[image_id, src_path, crop_box], preproc_version).

    `crop_box` is None for full-image variants, otherwise (x0, y0, x1, y1) in 512
    coordinates taken from C05 for the requested margin.
    """
    import pandas as pd

    if raw_dir is not None:
        if variant.clahe:
            raise SystemExit(
                "--raw-dir cannot produce a CLAHE variant: the CLAHE image store is "
                "Person 1's (C04 path_img512_clahe). Use clahe0, or wait for C04.")
        if variant.kind == "crop":
            raise SystemExit(
                "--raw-dir cannot produce a crop variant: crop boxes come from C05 "
                "(Person 1). Use the full variant, which is what the W2 dedup needs.")
        root = Path(raw_dir)
        rows = [{"image_id": p.stem, "src_path": str(p), "crop_box": None}
                for p in sorted(root.rglob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not rows:
            raise SystemExit("no images found under %s" % root)
        return pd.DataFrame(rows), STANDIN_PREPROC

    mp = Path(manifest_path)
    man = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_csv(mp)
    col = "path_img512_clahe" if variant.clahe else "path_img512"
    if col not in man.columns:
        raise SystemExit("manifest %s has no column %r; is this a C04 manifest?" % (mp, col))
    base = mp.parent
    df = man[["image_id", col]].rename(columns={col: "src_path"}).copy()
    df["src_path"] = [str(p) if os.path.isabs(str(p)) else str(base / str(p))
                      for p in df["src_path"]]
    df["crop_box"] = None

    if variant.kind == "crop":
        if roi_path is None:
            raise SystemExit("a crop variant needs --roi pointing at C05 lung_roi.parquet")
        roi = pd.read_parquet(roi_path)
        sel = roi[np.isclose(roi["margin"].to_numpy(float), variant.margin)]
        if sel.empty:
            raise SystemExit("C05 has no rows for margin %.2f; available: %s"
                             % (variant.margin, sorted(set(roi["margin"]))))
        boxes = {r.image_id: (int(r.crop_x0), int(r.crop_y0), int(r.crop_x1), int(r.crop_y1))
                 for r in sel.itertuples()}
        missing = set(df.image_id) - set(boxes)
        if missing:
            raise SystemExit("C05 is missing crop boxes for %d images, first few: %s"
                             % (len(missing), sorted(missing)[:5]))
        df["crop_box"] = [boxes[i] for i in df.image_id]

    return df, REAL_PREPROC


def read_image(path, crop_box):
    """uint8 (H, W) grayscale, cropped if asked. The crop is applied in 512 coordinates,
    which is what C05 records, so the image must already be 512 wide when cropping."""
    from PIL import Image
    a = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        if a.shape[0] != 512 or a.shape[1] != 512:
            raise ValueError("crop boxes are in 512 coordinates but %s is %dx%d"
                             % (path, a.shape[1], a.shape[0]))
        a = a[max(y0, 0):min(y1, 512), max(x0, 0):min(x1, 512)]
        if a.size == 0:
            raise ValueError("empty crop %s for %s" % (crop_box, path))
    return a


# ----------------------------------------------------------------- the model
class Embedder:
    def __init__(self, device=None, batch=16):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        # autotuning and TF32 pick kernels by timing, which makes runs differ
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION)
        self.half = self.device == "cuda"
        self.model = (model.half() if self.half else model.float()).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(model.config.hidden_size)

    def embed(self, images):
        """list of uint8 (H, W) -> float16 [n, dim], L2-normalised."""
        torch = self.torch
        rgb = [np.stack([im] * 3, axis=-1) for im in images]
        x = self.proc(images=rgb, return_tensors="pt")["pixel_values"]
        x = x.to(self.device, dtype=torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            out = self.model(pixel_values=x)
            feat = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
            feat = feat.float()
            feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return feat.cpu().numpy().astype(np.float16)


# ----------------------------------------------------------------- shards
def shard_path(out, idx):
    return Path(str(out) + ".partial") / ("shard_%05d.npz" % idx)


def write_shard(out, idx, ids, emb):
    p = shard_path(out, idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    # savez appends ".npz" when handed a path whose name does not end in it, which would
    # leave the rename below with nothing to rename. A file object is taken verbatim.
    with open(tmp, "wb") as fh:
        np.savez(fh, image_id=np.asarray(ids, dtype="<U32"), emb=emb)
    os.replace(tmp, p)                      # atomic: a crash cannot leave half a shard


def run(df, variant, out, preproc_version, batch=16, device=None, resume=True, log=print):
    """Embeds every row of df, in sorted image_id order, and writes the C07 npz."""
    df = df.sort_values("image_id").reset_index(drop=True)
    n = len(df)
    n_shards = (n + SHARD - 1) // SHARD
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    todo = [k for k in range(n_shards) if not (resume and shard_path(out, k).exists())]
    done = n_shards - len(todo)
    if done:
        log("resuming: %d/%d shards already on disk" % (done, n_shards))
    if todo:
        emb_model = Embedder(device=device, batch=batch)
        if emb_model.dim != K.EMB_DIM:
            raise SystemExit("model hidden size %d but contract C07 says %d"
                             % (emb_model.dim, K.EMB_DIM))
        t0, imgs_done = time.perf_counter(), 0
        for k in todo:
            lo, hi = k * SHARD, min((k + 1) * SHARD, n)
            part = df.iloc[lo:hi]
            vecs = []
            for b in range(0, len(part), batch):        # fixed slices: resume-stable
                chunk = part.iloc[b:b + batch]
                images = [read_image(r.src_path, r.crop_box) for r in chunk.itertuples()]
                vecs.append(emb_model.embed(images))
            write_shard(out, k, part.image_id.tolist(), np.concatenate(vecs, axis=0))
            imgs_done += len(part)
            rate = imgs_done / (time.perf_counter() - t0)
            log("  shard %d/%d  %d images  %.1f img/s  eta %.1f min"
                % (k + 1, n_shards, hi, rate, (n - hi) / max(rate, 1e-9) / 60))

    ids, embs = [], []
    for k in range(n_shards):
        # a context manager, because an open NpzFile holds a Windows file handle and the
        # unlink below would fail with PermissionError
        with np.load(shard_path(out, k)) as z:
            ids.append(z["image_id"])
            embs.append(z["emb"])
    ids = np.concatenate(ids)
    emb = np.concatenate(embs, axis=0)
    assert len(ids) == n, "shards hold %d rows, manifest has %d" % (len(ids), n)
    assert len(set(ids.tolist())) == len(ids), "duplicate image_id in the output"

    with open(out, "wb") as fh:          # again verbatim, so `out` is the path we return
        np.savez_compressed(
            fh,
            image_id=ids.astype("<U32"),
            emb=emb,
            l2_normalised=np.array(True),
            variant=np.array(variant.name),
            model_id=np.array("%s@%s" % (MODEL_ID, REVISION)),
            preproc_version=np.array(preproc_version),
        )
    for k in range(n_shards):                           # partials only after the merge
        shard_path(out, k).unlink()
    try:
        shard_path(out, 0).parent.rmdir()
    except OSError:
        pass
    return out, len(ids)


# ----------------------------------------------------------------- verification
def verify(out, df, variant, sample=64, batch=16, device=None, seed=0, log=print):
    """The acceptance criterion, as a command: re-embed a random subsample and compare.
    Returns (min_cosine, n_checked)."""
    z = np.load(out)
    saved = {i: v for i, v in zip(z["image_id"].tolist(), z["emb"])}
    df = df.sort_values("image_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pick = df.iloc[sorted(rng.choice(len(df), size=min(sample, len(df)), replace=False))]
    emb_model = Embedder(device=device, batch=batch)
    worst, checked = 1.0, 0
    for b in range(0, len(pick), batch):
        chunk = pick.iloc[b:b + batch]
        fresh = emb_model.embed([read_image(r.src_path, r.crop_box) for r in chunk.itertuples()])
        for row, v in zip(chunk.itertuples(), fresh):
            a = saved[row.image_id].astype(np.float32)
            b2 = v.astype(np.float32)
            cos = float(a @ b2 / (np.linalg.norm(a) * np.linalg.norm(b2)))
            worst = min(worst, cos)
            checked += 1
    log("re-run check: %d images, worst cosine %.6f" % (checked, worst))
    return worst, checked


def input_fingerprint(df):
    """A hash of what went in, so a C07 file can be traced to its manifest."""
    h = hashlib.sha256()
    for r in df.sort_values("image_id").itertuples():
        h.update(str(r.image_id).encode())
        h.update(str(r.crop_box).encode())
    return h.hexdigest()[:16]


def sidecar(out, variant, preproc_version, df, n, seconds, worst_cos=None):
    meta = {
        "task": "P2-03", "contract": "C07", "variant": variant.name,
        "model_id": "%s@%s" % (MODEL_ID, REVISION),
        "preproc_version": preproc_version, "n_images": n,
        "emb_dim": K.EMB_DIM, "dtype": "float16", "l2_normalised": True,
        "input_fingerprint": input_fingerprint(df),
        "seconds": round(seconds, 1),
        "minutes_per_11200": round(seconds / max(n, 1) * 11200 / 60, 1),
        "rerun_worst_cosine": worst_cos,
    }
    p = Path(str(out) + ".json")
    json.dump(meta, open(p, "w", encoding="utf-8"), indent=1)
    return p, meta
