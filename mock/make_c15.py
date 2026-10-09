"""P3-03: MOCK generator for C15 (prediction log, JSONL). Fake data, valid shape.

    python mock/make_c15.py --n 40
writes mock/logs/predictions.jsonl (overwritten; the real log is append-only)

Lines cover ANALYSED (TB / REFER / NOT_TB, with and without the link rule), REJECTED_GATE and
UNREADABLE cases, plus a few user_override lines as written by P1's slider via log_case.
For non-ANALYSED cases the triage fields are null (C14 nulls those blocks; C15 is silent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _p3_common as c  # noqa: E402

T_LOWER, T_UPPER = 0.18, 0.83     # stand in for D23
VERSIONS = {"model_version": "reg-v1-mock", "split_version": "v1",
            "decisions_hash": "sha256:" + "0" * 64, "git_sha": "0000000"}


def fake_case_id(i: int, seed: int) -> str:
    """Looks like a salted hash, as in C14; derived from a counter, never from patient data."""
    return "c_" + hashlib.sha256(f"mock-{seed}-{i}".encode()).hexdigest()[:8]


def fake_line(i: int, seed: int = 0, run_id: str = "r_mock01",
              start: datetime = datetime(2026, 11, 30, 10, 0, tzinfo=timezone.utc)) -> dict:
    r = c.rng_for(f"line{i}", seed, salt=6)
    roll = float(r.random())
    status = "UNREADABLE" if roll < 0.05 else "REJECTED_GATE" if roll < 0.15 else "ANALYSED"
    line = {
        "ts": (start + timedelta(seconds=7 * i)).isoformat(timespec="seconds"),
        "run_id": run_id, "case_id": fake_case_id(i, seed), "status": status,
        "category_final": None, "link_fired": None, "gate_pass": None,
        "p_tb_cal": None, "t_lower": None, "t_upper": None,
        "operating_point": "locked_default", **VERSIONS,
        "latency_ms": int(r.integers(80, 400)),
    }
    if status == "REJECTED_GATE":
        line["gate_pass"] = False
    elif status == "ANALYSED":
        p = round(float(r.beta(0.7, 2.0)), 3)
        pre = "TB" if p >= T_UPPER else "REFER" if p >= T_LOWER else "NOT_TB"
        link = pre == "NOT_TB" and r.random() < 0.15        # link rule only moves NOT_TB -> REFER
        line.update(gate_pass=True, p_tb_cal=p, t_lower=T_LOWER, t_upper=T_UPPER, link_fired=bool(link),
                    category_final="REFER" if link else pre, latency_ms=int(r.integers(700, 2400)))
    return line


def generate(n: int = 40, seed: int = 0, root: Path | str | None = None) -> Path:
    lines = [fake_line(i, seed) for i in range(n)]
    # P1 slider overrides: same case logged again with operating_point=user_override
    analysed = [l for l in lines if l["status"] == "ANALYSED"]
    for k, src in enumerate(analysed[:3]):
        ts = datetime.fromisoformat(lines[-1]["ts"]) + timedelta(seconds=30 * (k + 1))
        lines.append({**src, "ts": ts.isoformat(timespec="seconds"), "operating_point": "user_override"})
    path = c.out_path(root, "logs", "predictions.jsonl")
    path.write_text("".join(json.dumps(l, separators=(",", ":")) + "\n" for l in lines))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None, help="output root (default: mock/)")
    a = ap.parse_args(argv)
    print(generate(a.n, a.seed, a.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
