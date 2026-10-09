"""P3-12: select 40 development cases for the Grad-CAM review, render overlays, build the review sheet.

The selection rule is pre-registered in docs/p3_gradcam_review.md section 1; this module implements it
literally. Selection uses only C12 + C01 (labels and decisions); heatmaps and boxes are produced only
afterwards, so they cannot influence which cases are shown.

    python -m p3_xai.review select    # -> artifacts/p3/review/selection.csv
    python -m p3_xai.review render    # -> artifacts/p3/review/R01_<id>.png ... + review_sheet.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from p3_det import decisions as dec
from p3_det.data.config import DataPaths, resolve_preprocessing

SEED = 20261009
PER_CELL = 10
CELLS = (("TP", 1, True), ("TN", 0, True), ("FN", 1, False), ("FP", 0, False))   # (name, tb_label, correct)
SPLIT = "development"
OUT_DIR = dec.REPO / "artifacts" / "p3" / "review"
C12_PATH = dec.REPO / "artifacts" / "p4" / "triage" / "triage_tbx11k_development_v1.parquet"
C10_PATH = dec.REPO / "artifacts" / "p3" / "det" / "detections_tbx11k_development_v1.json"
CAVEAT = ("Heatmaps are attention evidence, not a finding. Agreement with radiologist boxes is overlap with "
          "incomplete annotations, not proof of clinical correctness.")
SEL_FIELDS = ["review_no", "image_id", "cell", "correct", "tb_label", "true_category", "category_final",
              "category_pre_link", "link_fired", "p_tb_cal", "screen_positive", "filled_from"]


# --- selection -------------------------------------------------------------------------------------

def eligible(c12_rows: list[dict], c01: dict[str, dict]) -> list[dict]:
    out = []
    for r in c12_rows:
        if r.get("split") != SPLIT:
            raise ValueError(f"C12 row {r['image_id']} is from split {r.get('split')!r}; development only")
        lab = c01.get(r["image_id"])
        if not r["gate_pass"] or lab is None or int(lab["tb_label"]) not in (0, 1):
            continue
        tb = int(lab["tb_label"])
        out.append({**r, "tb_label": tb, "true_category": lab["category"],
                    "correct": bool(r["screen_positive"]) == (tb == 1)})
    return out


def select(rows: list[dict], seed: int = SEED, per_cell: int = PER_CELL) -> list[dict]:
    """The pre-registered rule: 4 cells x 10, sorted ids, one generator, same-group fill, shuffled order."""
    rng = np.random.default_rng(seed)
    pools = {name: sorted(r["image_id"] for r in rows if r["tb_label"] == tb and r["correct"] == ok)
             for name, tb, ok in CELLS}
    by_id = {r["image_id"]: r for r in rows}
    chosen: dict[str, list[str]] = {}
    for name, _, _ in CELLS:
        pool = pools[name]
        chosen[name] = [str(x) for x in rng.choice(pool, size=min(per_cell, len(pool)), replace=False)] if pool else []
    picks = []
    for name, _, ok in CELLS:
        picks += [(i, name, "") for i in chosen[name]]
    for name, _, ok in CELLS:                                  # shortfall: fill from the other cell of the group
        short = per_cell - len(chosen[name])
        if short > 0:
            other = next(n for n, _, o in CELLS if o == ok and n != name)
            left = [i for i in pools[other] if i not in chosen[other] and i not in {p[0] for p in picks}]
            extra = [str(x) for x in rng.choice(left, size=min(short, len(left)), replace=False)] if left else []
            picks += [(i, other, f"filled for {name}") for i in extra]
    order = rng.permutation(len(picks))
    out = []
    for k, j in enumerate(order, start=1):
        iid, cell, filled = picks[j]
        r = by_id[iid]
        out.append({"review_no": f"R{k:02d}", "image_id": iid, "cell": cell, "correct": r["correct"],
                    "tb_label": r["tb_label"], "true_category": r["true_category"],
                    "category_final": r["category_final"], "category_pre_link": r["category_pre_link"],
                    "link_fired": bool(r["link_fired"]), "p_tb_cal": float(r["p_tb_cal"]),
                    "screen_positive": bool(r["screen_positive"]), "filled_from": filled})
    return out


def load_c12(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: P4's development C12 (or its mock via TB_MOCK=1) is needed")
    return pq.read_table(path).to_pylist()


def load_c01(paths: DataPaths) -> dict[str, dict]:
    with open(paths.splits, newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f) if r["split"] == SPLIT}


def write_selection(sel: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "selection.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEL_FIELDS)
        w.writeheader()
        w.writerows(sel)
    return p


def read_selection(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["correct"] = r["correct"] == "True"
        r["tb_label"] = int(r["tb_label"])
        r["p_tb_cal"] = float(r["p_tb_cal"])
    return rows


# --- rendering ---------------------------------------------------------------------------------------

def boxes_from_c10(path: Path) -> Callable[[str], list]:
    """Detector boxes above the C10 threshold for an image id (as CaseResult-style dicts)."""
    doc = json.loads(path.read_text())
    thr = doc["conf_threshold"]
    by_id = {im["image_id"]: im["boxes"] for im in doc["images"]}
    return lambda iid: [{"xyxy": b[:4], "score": b[4]} for b in by_id.get(iid, []) if b[4] >= thr]


def render(selection: list[dict], *, paths: DataPaths | None = None, out_dir: Path = OUT_DIR,
           decisions_dir: Path | None = None, heatmap_fn: Callable | None = None,
           boxes_fn: Callable | None = None) -> Path:
    """40 overlays (P3-11 renderer) + review_sheet.html. Defaults: R3 explain() for heatmaps; C10 boxes
    (or R3 detect() if the C10 development file does not exist)."""
    import pyarrow.parquet as pq

    from p3_det import api as det_api

    from . import api as xai_api
    paths = paths or DataPaths.default()
    margin, clahe, _ = resolve_preprocessing(decisions_dir=decisions_dir)
    man = {r["image_id"]: r for r in pq.read_table(paths.manifest).to_pylist()}
    roi = {r["image_id"]: r for r in pq.read_table(paths.lung_roi).to_pylist() if abs(r["margin"] - margin) < 1e-6}
    heatmap_fn = heatmap_fn or xai_api.explain
    if boxes_fn is None and C10_PATH.exists():
        c10_boxes = boxes_from_c10(C10_PATH)
        boxes_fn = lambda img, crop, iid: c10_boxes(iid)  # noqa: E731
    boxes_fn = boxes_fn or (lambda img, crop, iid: det_api.detect(img, crop))
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in selection:
        iid = r["image_id"]
        rel = Path(man[iid]["path_img512_clahe"] if clahe else man[iid]["path_img512"])
        img = np.asarray(Image.open(rel if rel.is_absolute() else paths.root / rel).convert("L"))
        ro = roi[iid]
        crop = [ro["crop_x0"], ro["crop_y0"], ro["crop_x1"], ro["crop_y1"]]
        r["overlay"] = f"{r['review_no']}_{iid}.png"
        xai_api.render_overlay_to(img, heatmap_fn(img, crop), boxes_fn(img, crop, iid),
                               out_path=out_dir / r["overlay"], case_id=iid)
    return write_sheet(selection, out_dir)


def _label(tb: int, category: str) -> str:
    return f"{'TB' if tb == 1 else 'non-TB'} ({category})"


def write_sheet(selection: list[dict], out_dir: Path) -> Path:
    e = html.escape
    cards = []
    for r in selection:
        pred = e(r["category_final"]) + (f" (link rule: {e(r['category_pre_link'])} &rarr; REFER)"
                                         if str(r.get("link_fired")) == "True" else "")
        cards.append(f"""
<section class="case">
  <h2>{e(r['review_no'])} &middot; case {e(r['image_id'])}</h2>
  <div class="row">
    <img src="{e(r['overlay'])}" alt="overlay {e(r['review_no'])}">
    <div class="facts">
      <table>
        <tr><th>True label</th><td>{e(_label(int(r['tb_label']), r['true_category']))}</td></tr>
        <tr><th>Predicted category</th><td>{pred}</td></tr>
        <tr><th>p_tb_cal</th><td>{float(r['p_tb_cal']):.3f}</td></tr>
        <tr><th>Decision</th><td>{'correct' if r['correct'] else 'incorrect'}</td></tr>
      </table>
      <p class="label">Guide's comment</p>
      <div class="comment"></div>
      <p class="label">Action</p>
      <div class="comment small"></div>
    </div>
  </div>
</section>""")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Grad-CAM review sheet</title>
<style>
  body {{ font-family: system-ui, Arial, sans-serif; margin: 12mm; color: #111; }}
  header p, footer p {{ font-size: 10pt; }}
  .caveat {{ border: 2px solid #780000; padding: 6px 10px; font-weight: 600; }}
  .case {{ border-top: 1px solid #999; padding: 6mm 0; break-inside: avoid; page-break-inside: avoid; }}
  .case h2 {{ font-size: 13pt; margin: 0 0 3mm; }}
  .row {{ display: flex; gap: 6mm; }}
  .row img {{ width: 72mm; height: auto; border: 1px solid #ccc; }}
  .facts {{ flex: 1; font-size: 10.5pt; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; width: 38mm; padding: 1mm 2mm 1mm 0; }}
  td {{ padding: 1mm 0; }}
  .label {{ margin: 3mm 0 1mm; font-weight: 600; }}
  .comment {{ border: 1px solid #555; height: 32mm; }}
  .comment.small {{ height: 12mm; }}
  @media print {{ body {{ margin: 8mm; }} .case:nth-of-type(2n) {{ page-break-after: always; }} }}
</style></head><body>
<header>
  <h1>Grad-CAM review &middot; development cases</h1>
  <p class="caveat">{e(CAVEAT)}</p>
  <p>{len(selection)} cases, shuffled (selection rule: docs/p3_gradcam_review.md section 1, seed {SEED}).
     Date: ____________ &nbsp; Guide: ______________________</p>
</header>
{''.join(cards)}
<footer><p class="caveat">{e(CAVEAT)}</p></footer>
</body></html>
"""
    p = out_dir / "review_sheet.html"
    p.write_text(doc, encoding="utf-8")
    return p


def main(argv=None) -> int:
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "render"])
    a = ap.parse_args(argv)
    paths = DataPaths.default()
    c12 = C12_PATH if os.environ.get("TB_MOCK") != "1" else \
        dec.REPO / "mock" / "artifacts" / "p4" / "triage" / "triage_tbx11k_development_v1.parquet"
    if a.cmd == "select":
        sel = select(eligible(load_c12(c12), load_c01(paths)))
        p = write_selection(sel, OUT_DIR)
        cells = {n: sum(r["cell"] == n for r in sel) for n, _, _ in CELLS}
        print(f"selected {len(sel)} cases {cells} -> {p}")
        return 0
    sel = read_selection(OUT_DIR / "selection.csv")
    sheet = render(sel, paths=paths)
    print(f"{len(sel)} overlays + {sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
