"""P3 audit harness core: each check ends in exactly one status and records it for the report.

PASS -> the test passes; FAIL -> pytest.fail; MANUAL / NOT_YET -> pytest.skip (with the status in the reason).
Results are appended as JSON lines to $P3_VERIFY_OUT (read by p3_ops.verify_all).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PY = sys.executable
STATUSES = ("PASS", "FAIL", "MANUAL", "NOT_YET")
OUT = Path(os.environ.get("P3_VERIFY_OUT", REPO / "artifacts" / "p3" / "verify" / "results.jsonl"))


def finish(check_id: str, desc: str, status: str, evidence: str, *, fix: str = "", owner: str = "",
           command: str = "") -> None:
    assert status in STATUSES, status
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": check_id, "desc": desc, "status": status, "evidence": evidence.strip(),
                            "fix": fix, "owner": owner, "command": command}) + "\n")
    if status == "FAIL":
        pytest.fail(f"{check_id} FAIL: {evidence}\nFIX: {fix}", pytrace=False)
    if status in ("MANUAL", "NOT_YET"):
        pytest.skip(f"{check_id} {status}: {evidence[:300]}")


def run_pytest(*args: str, timeout: int = 3600) -> tuple[int, str]:
    """Run pytest in a subprocess (own throwaway basetemp); return (returncode, last lines)."""
    import shutil
    import tempfile
    bt = Path(tempfile.mkdtemp(prefix="p3v_"))
    try:
        p = subprocess.run([PY, "-m", "pytest", "-p", "no:cacheprovider", "-q", f"--basetemp={bt / 't'}", *args],
                           cwd=REPO, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-6:])
        return p.returncode, tail
    finally:
        shutil.rmtree(bt, ignore_errors=True)


def summary_line(tail: str) -> str:
    m = re.findall(r"\d+ (?:passed|failed|skipped|deselected|errors?|warnings?)[^\n]*", tail)
    return m[-1] if m else tail.splitlines()[-1] if tail else ""


def git(*args: str, cwd: Path = REPO) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def own_repo() -> bool:
    rc, top = git("rev-parse", "--show-toplevel")
    return rc == 0 and Path(top).resolve() == REPO.resolve()


def load_json(rel: str):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
