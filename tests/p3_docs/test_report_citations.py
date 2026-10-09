"""FIX 6: every citation in report/p3_*.md is a concrete path: no template placeholders (<...>, {...}), and every
cited path outside a TBD line exists on disk."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTS = sorted((REPO / "report").glob("p3_*.md"))


def citations(text: str) -> list[str]:
    out = []
    for c in re.findall(r"\[([^\]]+)\]", text):
        for part in c.split(";"):
            p = part.strip().split(",")[0].split(" ")[0]
            if "/" in p or p.endswith((".md", ".json", ".csv", ".parquet", ".py", ".yaml")):
                out.append(p)
    return out


def test_reports_exist():
    assert REPORTS, "no report/p3_*.md files"


def test_no_placeholder_in_any_cited_path():
    bad = [(f.name, p) for f in REPORTS for p in citations(f.read_text(encoding="utf-8")) if re.search(r"[<>{}]", p)]
    assert bad == [], f"template placeholders in citations: {bad}"


def test_cited_paths_exist_outside_tbd_lines():
    missing = []
    for f in REPORTS:
        for line in f.read_text(encoding="utf-8").splitlines():
            if "TBD" in line:
                continue
            missing += [(f.name, p) for p in citations(line) if not (REPO / p).exists()]
    assert missing == [], f"cited paths that do not exist: {missing}"


def test_placeholder_check_can_fail():
    assert citations("x [artifacts/p3/xai/loc_<model_id>_development_v1.parquet] y") == \
        ["artifacts/p3/xai/loc_<model_id>_development_v1.parquet"]
