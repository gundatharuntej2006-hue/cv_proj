"""I3: deliberate-kill test - analyse the demo output and append the pasted result to the runbook log.

notebooks/p3/kill_resume_test.ipynb prints one line per test:
    KILLTEST_RESULT {"part": "B", "platform": "kaggle", ...}
Paste those lines into a file (or stdin), then:

    python -m p3_ops.kill_test_log --paste results.txt            # shows the rows it would add
    python -m p3_ops.kill_test_log --paste results.txt --write    # appends them to docs/p3_kaggle_runbook.md s.10

Pass rule (runbook section 10): the run resumes from the last checkpoint, at most ONE epoch is lost (the one
that was running when the session died), and the run finishes. Only Part B (a real Kaggle "Stop session") is
written as a "Kaggle" row, which is what the I3 acceptance asks for; Part A (a process kill inside a running
notebook) is logged as a supporting row.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "docs" / "p3_kaggle_runbook.md"
TAG = "KILLTEST_RESULT "
MAX_LOST = 1
_EPOCH = re.compile(r"^epoch\s+(\d+)\b")
_START = re.compile(r"^start at epoch (\d+)")
KILLED_HOW = {"A": "process kill inside the notebook (SIGKILL)", "B": "Stop session"}


class Refused(ValueError):
    pass


def last_epoch(lines: list[str]) -> int | None:
    eps = [int(m.group(1)) for l in lines if (m := _EPOCH.match(l.strip()))]
    return max(eps) if eps else None


def analyse(part: str, before: list[str] | None, after: list[str], epochs: int, *, last_printed: int | None = None,
            ckpts_on_drive: list[str] | None = None, platform: str = "", gpu: str | None = None,
            run: str = "", day: str = "") -> dict:
    """Result of one kill test. ``before`` = output up to the kill (Part A); Part B uses ``last_printed``."""
    last = last_printed if last_printed is not None else last_epoch(before or [])
    start_line = next((l.strip() for l in after if _START.match(l.strip())), None)
    resumed_at = int(_START.match(start_line).group(1)) if start_line else None
    final = last_epoch(after)
    finished = any(l.startswith("done:") for l in after) and final == epochs - 1
    problems = []
    if last is None:
        problems.append("no epoch completed before the kill (kill later, after 'epoch 4' or so)")
    if resumed_at is None:
        problems.append("no 'start at epoch' line after the restart")
    elif resumed_at >= epochs:
        problems.append("the run had already finished before the kill; repeat with a new run name")
    elif resumed_at == 0:
        problems.append("restart began at epoch 0: nothing was resumed")
    lost = (last + 2 - resumed_at) if None not in (last, resumed_at) else None   # re-run epochs incl. the interrupted one
    ok = not problems and lost is not None and lost <= MAX_LOST and finished
    how = KILLED_HOW.get(part, part)
    if part == "B" and platform != "kaggle":
        how = "process kill (local smoke test, not I3 evidence)"
    return {"part": part, "date": day, "platform": platform, "gpu": gpu, "run": run,
            "killed_how": how, "last_epoch_before_kill": last,
            "ckpts_on_drive_after_kill": ckpts_on_drive, "first_line_after_restart": start_line,
            "resumed_at": resumed_at, "epochs_lost": lost, "finished": finished, "final_epoch": final,
            "pass": ok, "problems": problems}


def table(results: list[dict]) -> str:
    L = ["| Part | Killed how | Last epoch before kill | Resumed at | Epochs lost (<= 1) | Finished | Pass |",
         "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['part']} | {r['killed_how']} | {r['last_epoch_before_kill']} | {r['resumed_at']} | "
                 f"{r['epochs_lost']} | {r['finished']} | {'PASS' if r['pass'] else 'FAIL'} |")
    return "\n".join(L)


def parse_paste(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if TAG.strip() not in line:
            continue
        try:
            r = json.loads(line.split(TAG.strip(), 1)[1])
        except json.JSONDecodeError as e:
            raise Refused(f"malformed KILLTEST_RESULT line: {e}") from None
        for k in ("part", "date", "platform", "run", "last_epoch_before_kill", "resumed_at", "epochs_lost", "pass"):
            if k not in r:
                raise Refused(f"KILLTEST_RESULT without {k!r}")
        if not re.fullmatch(r"\d{4}-\d\d-\d\d", str(r["date"])):
            raise Refused(f"bad date {r['date']!r}")
        out.append(r)
    if not out:
        raise Refused("no KILLTEST_RESULT line found in the paste")
    return out


def runbook_row(r: dict) -> str:
    if r["part"] == "B" and r["platform"] != "kaggle":
        raise Refused(f"Part B result comes from {r['platform']!r}, not Kaggle: it cannot be logged as the I3 test")
    platform = (f"Kaggle / {r.get('gpu') or 'no GPU'}" if r["part"] == "B"
                else f"In-notebook process kill ({r['platform']} / {r.get('gpu') or 'no GPU'})")
    drive = r.get("ckpts_on_drive_after_kill")
    drive = ", ".join(drive) if drive else "not checked" if drive is None else "none"
    notes = "; ".join(r.get("problems") or []) or ("supporting test, not the I3 acceptance" if r["part"] == "A" else "")
    cells = [r["date"], platform, r["run"], r["last_epoch_before_kill"], r["killed_how"], drive,
             f"`{r.get('first_line_after_restart')}`", r["epochs_lost"], "yes" if r["pass"] else "no", notes]
    return "| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |"


def append_rows(text: str, rows: list[str]) -> str:
    """Insert rows at the end of the section-10 log table."""
    start = text.index("## 10.")
    head = text.index("| Date | Platform / GPU |", start)
    lines = text[head:].split("\n")
    n = next(i for i, l in enumerate(lines) if not l.startswith("|"))
    end = head + len("\n".join(lines[:n]))
    return text[:end] + "\n" + "\n".join(rows) + text[end:]


def main(argv=None, runbook: Path | None = None) -> int:
    runbook = RUNBOOK if runbook is None else runbook
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paste", required=True, help="file with the KILLTEST_RESULT lines ('-' = stdin)")
    ap.add_argument("--write", action="store_true", help="append to the runbook (default: show only)")
    a = ap.parse_args(argv)
    try:
        text = sys.stdin.read() if a.paste == "-" else Path(a.paste).read_text(encoding="utf-8")
        results = parse_paste(text)
        rows = [runbook_row(r) for r in results]
    except (Refused, OSError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(table(results) + "\n\nRows for docs/p3_kaggle_runbook.md section 10:\n" + "\n".join(rows))
    if a.write:
        runbook.write_text(append_rows(runbook.read_text(encoding="utf-8"), rows), encoding="utf-8")
        print(f"\nappended {len(rows)} row(s) to {runbook}")
    else:
        print("\nShown only. Re-run with --write to append.")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
