"""P3-14: prediction logger (contract C15, logs/predictions.jsonl, append-only).

log_case(case_result, operating_point) appends exactly one JSON line per call.

No PHI by construction: the line is built ONLY from whitelisted, type-checked CaseResult fields.
Free text (input.error, quality_reasons, fail_reasons), file paths and anything else in the
CaseResult never reach the log. Strings that are written are restricted:
    case_id, run_id      [A-Za-z0-9_-]{1,64}   (salted case hash / run id)
    version fields       [A-Za-z0-9._:+-]{1,128} and REQUIRED on every line
    status, category, operating point   fixed enums
so a path, a name or a DICOM tag cannot be smuggled in through them.

Concurrency: an exclusive lock on <log>.lock (msvcrt on Windows, fcntl elsewhere) serialises
writers across threads and processes; each line is written with one O_APPEND write and fsync.

Path: TB_LOG_PATH, else mock/logs/predictions.jsonl when TB_MOCK=1, else <repo>/logs/predictions.jsonl.
run_id: C14 has no run_id (docs/p3_contract_notes.md item 1): TB_RUN_ID, else the runs/<run_id>/
segment of the CaseResult's artefact paths (the path itself is never logged), else "unknown".
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATUSES = ("ANALYSED", "REJECTED_GATE", "UNREADABLE")
CATEGORIES = ("TB", "REFER", "NOT_TB")
OPERATING_POINTS = ("locked_default", "user_override")
VERSION_FIELDS = ("model_version", "split_version", "decisions_hash", "git_sha")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9._:+-]{1,128}$")
_RUN_ID_RE = re.compile(r"(?:^|/)runs/([A-Za-z0-9_-]{1,64})/")
FIELDS = ("ts", "run_id", "case_id", "status", "category_final", "link_fired", "gate_pass", "p_tb_cal",
          "t_lower", "t_upper", "operating_point", "model_version", "split_version", "decisions_hash",
          "git_sha", "latency_ms")
LOCK_TIMEOUT_S = 30.0
_thread_lock = threading.Lock()


class LogError(ValueError):
    """The CaseResult cannot be logged (missing required field, unsafe value, ...)."""


def log_path() -> Path:
    if os.environ.get("TB_LOG_PATH"):
        return Path(os.environ["TB_LOG_PATH"])
    base = REPO / "mock" if os.environ.get("TB_MOCK") == "1" else REPO
    return base / "logs" / "predictions.jsonl"


def _block(cr: dict, name: str) -> dict:
    b = cr.get(name)
    return b if isinstance(b, dict) else {}


def _num(v, name: str):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise LogError(f"{name} must be a number or null")
    return float(v)


def _bool(v, name: str):
    if v is None or isinstance(v, bool):
        return v
    raise LogError(f"{name} must be true/false or null")


def _run_id(cr: dict) -> str:
    env = os.environ.get("TB_RUN_ID")
    if env:
        if not ID_RE.match(env):
            raise LogError("TB_RUN_ID must match [A-Za-z0-9_-]{1,64}")
        return env
    for p in _block(cr, "explainability").values():
        m = _RUN_ID_RE.search(str(p or "").replace("\\", "/"))
        if m:
            return m.group(1)
    return "unknown"


def build_record(case_result: dict, operating_point: str, ts: datetime | None = None) -> dict:
    """The C15 line for one CaseResult (validated; raises LogError)."""
    if operating_point not in OPERATING_POINTS:
        raise LogError(f"operating_point must be one of {OPERATING_POINTS}, got {operating_point!r}")
    case_id = case_result.get("case_id")
    if not isinstance(case_id, str) or not ID_RE.match(case_id):
        raise LogError("case_id must be the salted case hash ([A-Za-z0-9_-]{1,64})")
    status = case_result.get("status")
    if status not in STATUSES:
        raise LogError(f"status must be one of {STATUSES}")
    versions = _block(case_result, "versions")
    for k in VERSION_FIELDS:
        v = versions.get(k)
        if not isinstance(v, str) or not v:
            raise LogError(f"required version field {k!r} is missing from CaseResult.versions")
        if not VERSION_RE.match(v):
            raise LogError(f"version field {k!r} has characters outside [A-Za-z0-9._:+-]")
    gate, triage, cls = (_block(case_result, k) for k in ("gate", "triage", "classifier"))
    category = triage.get("category_final")
    if category is not None and category not in CATEGORIES:
        raise LogError(f"category_final must be one of {CATEGORIES} or null")
    total = _block(case_result, "timing_ms").get("total")
    if total is not None and (isinstance(total, bool) or not isinstance(total, (int, float))):
        raise LogError("timing_ms.total must be a number or null")
    return {
        "ts": (ts or datetime.now(timezone.utc)).isoformat(timespec="milliseconds"),
        "run_id": _run_id(case_result),
        "case_id": case_id,
        "status": status,
        "category_final": category,
        "link_fired": _bool(triage.get("link_fired"), "link_fired"),
        "gate_pass": _bool(gate.get("gate_pass"), "gate_pass"),
        "p_tb_cal": _num(cls.get("p_tb_cal"), "p_tb_cal"),
        "t_lower": _num(triage.get("t_lower"), "t_lower"),
        "t_upper": _num(triage.get("t_upper"), "t_upper"),
        "operating_point": operating_point,
        **{k: versions[k] for k in VERSION_FIELDS},
        "latency_ms": None if total is None else int(round(total)),
    }


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock, open(lock_path, "a+b") as f:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"could not lock {lock_path}") from None
                    time.sleep(0.005)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_line(path: Path, record: dict) -> None:
    data = (json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    with _file_lock(path.with_name(path.name + ".lock")):
        fd = os.open(path, flags, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size:                                      # previous writer died mid-line: keep lines separate
                with open(path, "rb") as r:
                    r.seek(size - 1)
                    if r.read(1) != b"\n":
                        data = b"\n" + data
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


def log_case(case_result: dict, operating_point: str) -> None:
    """Append exactly one C15 line for this CaseResult (raises LogError, writes nothing, if invalid)."""
    append_line(log_path(), build_record(case_result, operating_point))
