"""Stand-in for contracts/decision.schema.json (C16, P4), written from the README 4.2 C16 table.

``validate_decision`` uses the real contract schema automatically once it exists.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA = REPO / "contracts" / "decision.schema.json"

C16_STANDIN = {
    "type": "object",
    "required": ["decision_id", "owner", "title", "criterion_preregistered", "preregistered_at",
                 "candidates", "value", "evidence", "data_used", "reviewed_by", "guide_ack",
                 "locked", "locked_at", "git_sha"],
    "properties": {
        "decision_id": {"type": "string", "pattern": "^D(0[1-9]|1[0-9]|2[0-3])$"},
        "owner": {"enum": ["p1", "p2", "p3", "p4"]},
        "title": {"type": "string", "minLength": 1},
        "criterion_preregistered": {"type": "string", "minLength": 1},
        "preregistered_at": {"type": "string", "format": "date"},
        "candidates": {"type": "array"},
        "value": {"type": "object"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "data_used": {"type": "array", "minItems": 1, "items": {
            "enum": ["none", "train", "development", "calibration", "images_only_all_splits"]}},
        "reviewed_by": {"type": "string"},
        "guide_ack": {"type": "boolean"},
        "locked": {"type": "boolean"},
        "locked_at": {"type": ["string", "null"], "format": "date"},
        "git_sha": {"type": ["string", "null"]},
    },
}


def validate_decision(doc: dict) -> None:
    import jsonschema
    schema = json.loads(CONTRACT_SCHEMA.read_text()) if CONTRACT_SCHEMA.exists() else C16_STANDIN
    jsonschema.validate(doc, schema, format_checker=jsonschema.FormatChecker())
