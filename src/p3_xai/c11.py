"""P3-10: C11 files (README 4.2): heatmaps artifacts/p3/xai/{model_id}/{split}/{image_id}.npy and the
localisation table artifacts/p3/xai/loc_{model_id}_{split}_v1.parquet.

Rows exist only for TB images with annotated boxes (n_gt_boxes > 0), as decided in P3-10.
``validate_table`` / ``validate_heatmap`` use tbcore's validator once P4 publishes it; until then they
apply the C11 table directly.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SUBTYPES = ("active", "latent", "active+latent")
COLUMNS = ("image_id", "model_id", "peak_x", "peak_y", "pointing_hit", "energy_in_box", "n_gt_boxes", "tb_subtype")
TABLE_RE = re.compile(r"^loc_(?P<model>.+)_(?P<split>development|calibration|internal_test|challenge_test|external_test)_v\d+\.parquet$")


def schema():
    import pyarrow as pa
    return pa.schema([("image_id", pa.string()), ("model_id", pa.string()), ("peak_x", pa.int16()),
                      ("peak_y", pa.int16()), ("pointing_hit", pa.bool_()), ("energy_in_box", pa.float32()),
                      ("n_gt_boxes", pa.int16()), ("tb_subtype", pa.string())])


def table_path(root: Path, model_id: str, split: str) -> Path:
    return Path(root) / f"loc_{model_id}_{split}_v1.parquet"


def heatmap_path(root: Path, model_id: str, split: str, image_id: str) -> Path:
    return Path(root) / model_id / split / f"{image_id}.npy"


def write_table(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema()), path)
    validate_table(path)
    return path


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"C11 invalid: {msg}")


def validate_heatmap(path: Path, crop_box=None) -> np.ndarray:
    h = np.load(path)
    _check(h.dtype == np.float16 and h.shape == (512, 512), f"{path.name}: must be float16 [512,512]")
    _check(bool(np.isfinite(h).all()) and float(h.min()) >= 0 and float(h.max()) <= 1, f"{path.name}: range [0,1]")
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        outside = np.ones(h.shape, bool)
        outside[y0:y1, x0:x1] = False
        _check(not h[outside].any(), f"{path.name}: non-zero outside the crop")
    return h


def validate_table(path: Path):
    path = Path(path)
    m = TABLE_RE.match(path.name)
    _check(m is not None, f"file name {path.name} does not follow loc_{{model_id}}_{{split}}_v1.parquet")
    try:
        from tbcore import validators  # P4 (P4-01)
    except ImportError:
        validators = None
    if validators is not None:
        validators.validate("C11", path)   # API name to confirm with P4; a mismatch raises, never skips
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    _check(tuple(t.schema.names) == COLUMNS, f"columns {t.schema.names} != {COLUMNS}")
    for f in schema():
        _check(t.schema.field(f.name).type == f.type, f"{f.name} must be {f.type}")
    d = t.to_pydict()
    _check(len(set(d["image_id"])) == len(d["image_id"]), "duplicate image_id")
    _check(set(d["model_id"]) <= {m.group("model")}, "model_id column must match the file name")
    _check(all(0 <= v < 512 for v in d["peak_x"] + d["peak_y"]), "peak outside [0, 512)")
    _check(all(0.0 <= e <= 1.0 for e in d["energy_in_box"]), "energy_in_box outside [0, 1]")
    _check(all(n > 0 for n in d["n_gt_boxes"]), "rows only for TB images with boxes (n_gt_boxes > 0)")
    _check(all(s in SUBTYPES for s in d["tb_subtype"]), f"tb_subtype must be one of {SUBTYPES}")
    return t
