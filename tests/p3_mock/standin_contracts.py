"""STAND-IN validators for C10, C11, C14, C15, written from the README section 4.2 tables.

contracts/ (schemas) and src/tbcore (validators) do not exist yet. validate_c14 uses
contracts/case_result.schema.json automatically once it exists; the others must be switched to
tbcore by hand when P4 publishes its API (test_tbcore_switch_reminder warns when it appears).
Nothing here is a contract: if this file and contracts/ ever disagree, contracts/ wins.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = REPO / "contracts"

_num = {"type": "number"}
_nnum = {"type": ["number", "null"]}
_nbool = {"type": ["boolean", "null"]}
_int4 = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
_strs = {"type": "array", "items": {"type": "string"}}
_CATS = ["TB", "REFER", "NOT_TB"]


def _obj(props: dict, nullable: bool = False) -> dict:
    return {"type": ["object", "null"] if nullable else "object", "required": list(props),
            "properties": props}


C10_SCHEMA = _obj({
    "schema_version": {"type": "string"},
    "model_id": {"type": "string"},
    "input_size": {"enum": [512, 640]},
    "conf_threshold": {"type": "number", "minimum": 0, "maximum": 1},
    "coords": {"const": "full512"},
    "images": {"type": "array", "items": _obj({
        "image_id": {"type": "string", "minLength": 1},
        "crop_box": _int4,
        "boxes": {"type": "array", "items": {
            "type": "array", "minItems": 5, "maxItems": 5, "items": _num}},
        "n_lesions_above_thr": {"type": "integer", "minimum": 0},
        "max_score": {"type": "number", "minimum": 0, "maximum": 1},
    })},
})

C15_SCHEMA = _obj({
    "ts": {"type": "string", "format": "date-time"},
    "run_id": {"type": "string", "minLength": 1},
    "case_id": {"type": "string", "pattern": r"^[A-Za-z0-9_-]{1,64}$"},
    "status": {"enum": ["ANALYSED", "REJECTED_GATE", "UNREADABLE"]},
    "category_final": {"enum": _CATS + [None]},
    "link_fired": _nbool, "gate_pass": _nbool,
    "p_tb_cal": _nnum, "t_lower": _nnum, "t_upper": _nnum,
    "operating_point": {"enum": ["locked_default", "user_override"]},
    "model_version": {"type": "string"}, "split_version": {"type": "string"},
    "decisions_hash": {"type": "string"}, "git_sha": {"type": "string"},
    "latency_ms": {"type": ["integer", "null"]},
})
C15_SCHEMA["additionalProperties"] = False
# ANALYSED lines must carry the triage values
C15_SCHEMA["allOf"] = [{"if": {"properties": {"status": {"const": "ANALYSED"}}},
                        "then": {"properties": {"category_final": {"enum": _CATS},
                                                "link_fired": {"type": "boolean"},
                                                "gate_pass": {"const": True},
                                                "p_tb_cal": _num, "t_lower": _num, "t_upper": _num}}}]

C14_SCHEMA = _obj({
    "schema_version": {"type": "string"},
    "case_id": {"type": "string", "pattern": r"^[A-Za-z0-9_-]{1,64}$"},
    "status": {"enum": ["ANALYSED", "REJECTED_GATE", "UNREADABLE"]},
    "input": _obj({"format": {"type": "string"}, "phi_tags_removed": {"type": "integer"},
                   "error": {"type": ["string", "null"]}}),
    "gate": _obj({"quality_pass": {"type": "boolean"}, "quality_reasons": _strs, "ood_score": _num,
                  "ood_threshold": _num, "gate_pass": {"type": "boolean"},
                  "view": {"type": ["string", "null"]}}, nullable=True),
    "segmentation": _obj({"sanity_pass": {"type": "boolean"}, "fail_reasons": _strs, "crop_box": _int4,
                          "fallback_full": {"type": "boolean"}}, nullable=True),
    "classifier": _obj({"model_id": {"type": "string"}, "head": {"type": "string"}, "p_tb_raw": _num,
                        "p_tb_cal": _num, "temperature": _num,
                        "three_class": _obj({"healthy": _num, "sick_non_tb": _num, "tb": _num})},
                       nullable=True),
    "detector": _obj({"model_id": {"type": "string"}, "conf_threshold": _num,
                      "n_lesions": {"type": "integer"},
                      "boxes": {"type": "array", "items": _obj({"xyxy": _int4, "score": _num})}},
                     nullable=True),
    "triage": _obj({"t_lower": _num, "t_upper": _num, "category_pre_link": {"enum": _CATS},
                    "category_final": {"enum": _CATS}, "link_fired": {"type": "boolean"},
                    "screen_positive": {"type": "boolean"},
                    "operating_point": {"enum": ["locked_default", "user_override"]}}, nullable=True),
    "explainability": _obj({"heatmap_path": {"type": ["string", "null"]},
                            "overlay_path": {"type": ["string", "null"]}}, nullable=True),
    "versions": _obj({k: {"type": "string"} for k in
                      ("model_version", "split_version", "decisions_hash", "git_sha")}),
    "timing_ms": {"type": ["object", "null"]},
    "disclaimer": {"type": ["string", "null"]},
})
_NULL_IF = {"REJECTED_GATE": ["segmentation", "classifier", "detector", "triage"],
            "UNREADABLE": ["gate", "segmentation", "classifier", "detector", "triage"]}
C14_SCHEMA["allOf"] = [
    {"if": {"properties": {"status": {"const": s}}},
     "then": {"properties": {b: {"type": "null"} for b in blocks}}}
    for s, blocks in _NULL_IF.items()]


def _formal_schema(name: str) -> dict | None:
    p = CONTRACTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _validate(instance, schema: dict) -> None:
    jsonschema.validate(instance, schema, format_checker=jsonschema.FormatChecker())


# --- C10 ---------------------------------------------------------------------------------------

def validate_c10(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    _validate(doc, C10_SCHEMA)
    assert re.fullmatch(r"detections_[a-z0-9]+_[a-z_]+_v\d+\.json", Path(path).name), Path(path).name
    seen = set()
    for im in doc["images"]:
        assert im["image_id"] not in seen, f"duplicate image_id {im['image_id']}"
        seen.add(im["image_id"])
        x0, y0, x1, y1 = im["crop_box"]
        assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512, im["crop_box"]
        scores = [b[4] for b in im["boxes"]]
        for bx0, by0, bx1, by1, s in im["boxes"]:
            assert 0 <= bx0 < bx1 <= 512 and 0 <= by0 < by1 <= 512, (im["image_id"], bx0, by0, bx1, by1)
            assert 0.01 <= s <= 1, s
        assert im["n_lesions_above_thr"] == sum(s >= doc["conf_threshold"] for s in scores)
        assert im["max_score"] == (max(scores) if scores else 0.0)
    return doc


# --- C11 ---------------------------------------------------------------------------------------

C11_COLUMNS = {"image_id": "string", "model_id": "string", "peak_x": "int16", "peak_y": "int16",
               "pointing_hit": "bool", "energy_in_box": "float", "n_gt_boxes": "int16",
               "tb_subtype": "string"}


def validate_c11_heatmap(path: Path, crop_box: list[int] | None = None) -> np.ndarray:
    h = np.load(path)
    assert h.dtype == np.float16 and h.shape == (512, 512), (h.dtype, h.shape)
    assert np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        outside = np.ones_like(h, bool)
        outside[y0:y1, x0:x1] = False
        assert not h[outside].any(), "heatmap must be zero outside the crop"
    return h


def validate_c11_table(path: Path):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    assert re.fullmatch(r"loc_.+_[a-z_]+_v\d+\.parquet", Path(path).name), Path(path).name
    assert t.schema.names == list(C11_COLUMNS), t.schema.names
    for name, kind in C11_COLUMNS.items():
        assert str(t.schema.field(name).type) == kind, (name, t.schema.field(name).type)
    d = t.to_pydict()
    assert all(0 <= v < 512 for v in d["peak_x"] + d["peak_y"])
    assert all(0 <= e <= 1 for e in d["energy_in_box"])
    assert all(n > 0 for n in d["n_gt_boxes"])                  # rows only for TB images with boxes (P3-10)
    assert all(s in ("active", "latent", "active+latent") for s in d["tb_subtype"])
    assert len(set(d["image_id"])) == len(d["image_id"])
    return t


# --- C14 / C15 ---------------------------------------------------------------------------------

def validate_c14(case_result: dict) -> None:
    _validate(case_result, _formal_schema("case_result.schema.json") or C14_SCHEMA)


def validate_c15_line(line: dict) -> None:
    _validate(line, C15_SCHEMA)


def validate_c15_file(path: Path) -> list[dict]:
    lines = []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        assert raw.strip(), f"blank line {n}"
        line = json.loads(raw)
        validate_c15_line(line)
        lines.append(line)
    return lines
