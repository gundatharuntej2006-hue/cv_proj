"""P3-04: dump random detector samples with boxes drawn, for a visual check.

    python notebooks/p3/p3_04_viz_check.py                 # real P1 data (or TB_MOCK=1 for P1 mocks)
    python notebooks/p3/p3_04_viz_check.py --fake          # FAKE P1 data (until P1's files exist)

Writes artifacts/p3/det/viz_check/{NN}_{image_id}.png (+ index.csv). Each panel shows the
augmented detector input (left) and the un-augmented one (right), both with boxes, plus the
original 512 image with the crop (yellow) and the boxes mapped back through det_to_full512 (green).
Only train/development images are ever used. These PNGs are derived from images: do not commit
them for real data (README 4.6), only look at them.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_det.data import DataPaths, DetDataConfig, TBDetectionDataset, det_to_full512  # noqa: E402
from p3_det.data.fake_p1 import make_fake_p1_tree  # noqa: E402

PANEL = 320


def _rgb(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img).convert("RGB")


def _draw(im: Image.Image, boxes, colour, scale: float = 1.0, width: int = 2) -> None:
    d = ImageDraw.Draw(im)
    for b in np.asarray(boxes).reshape(-1, 4):
        d.rectangle([float(v) * scale for v in b], outline=colour, width=width)


def render(ds: TBDetectionDataset, index: int) -> tuple[Image.Image, dict]:
    S = ds.cfg.input_size
    aug_img, aug_boxes, info = ds.load(index, augment_=True)
    plain_img, plain_boxes, _ = ds.load(index, augment_=False)
    s = ds.samples[index]
    full = np.asarray(Image.open(s.img_path).convert("L"))

    panels = []
    for img, boxes in ((aug_img, aug_boxes), (plain_img, plain_boxes)):
        p = _rgb(img).resize((PANEL, PANEL))
        _draw(p, boxes, (0, 255, 0), PANEL / S)
        panels.append(p)
    p = _rgb(full).resize((PANEL, PANEL))
    k = PANEL / 512
    _draw(p, [s.crop_box], (255, 220, 0), k)
    _draw(p, s.boxes_full512, (255, 0, 0), k, width=1)                      # annotation (red)
    _draw(p, det_to_full512(plain_boxes, s.crop_box, S), (0, 255, 0), k)   # mapped back (green)
    panels.append(p)

    canvas = Image.new("RGB", (3 * PANEL + 20, PANEL + 34), "white")
    for j, p in enumerate(panels):
        canvas.paste(p, (j * (PANEL + 10), 34))
    a = info["aug"] or {}
    title = (f"{s.image_id}  {'TB' if s.tb_label else 'non-TB'}  boxes {len(plain_boxes)}  crop {s.crop_box}  "
             f"| aug: scale {a.get('scale', 1):.3f} t=({a.get('tx', 0):.0f},{a.get('ty', 0):.0f}) "
             f"flip {a.get('hflip', False)} c {a.get('contrast', 1):.2f} b {a.get('brightness', 0):.0f}")
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), title, fill=(0, 0, 0))
    d.text((4, 16), "augmented input | plain input | full 512: crop (yellow), annotated (red), mapped back (green)",
           fill=(80, 80, 80))
    return canvas, {"image_id": s.image_id, "tb_label": s.tb_label, "n_boxes": len(plain_boxes),
                    "crop_box": s.crop_box, **{f"aug_{k}": v for k, v in a.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--split", default="train", choices=["train", "development"])
    ap.add_argument("--input-size", type=int, default=512, choices=[512, 640])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="use FAKE P1 data in a temp dir")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "p3" / "det" / "viz_check"))
    a = ap.parse_args(argv)

    if a.fake:
        paths = DataPaths(make_fake_p1_tree(Path(tempfile.mkdtemp(prefix="p3_fake_p1_")), n_per_split=80))
    else:
        paths = DataPaths.default()
        if not paths.splits.exists():
            print(f"no C01 at {paths.splits}; P1 data missing. Re-run with --fake (or TB_MOCK=1).")
            return 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = TBDetectionDataset(DetDataConfig(a.split, input_size=a.input_size, seed=a.seed, augment=True), paths)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "FAKE_" if a.fake else ""
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    rows = []
    for k, i in enumerate(idx):
        ds.set_epoch(int(rng.integers(0, 100)))
        img, row = render(ds, int(i))
        img.save(out / f"{prefix}{k:02d}_{row['image_id']}.png")
        rows.append({"file": f"{prefix}{k:02d}_{row['image_id']}.png", **row})
    with open(out / f"{prefix}index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} samples -> {out}  {ds.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
