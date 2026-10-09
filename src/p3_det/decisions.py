"""P3 decision files (C16): pre-register first, fill later, prove the order with git.

Every P3 decision follows the same two commits (README 4.5: "the criterion is committed before the
evidence; the git log is checked"):

  1. ``write_criterion``: file with criterion_preregistered + preregistered_at, value = {}, locked
     false.  Commit it.
  2. the decision tool fills value/evidence (refusing unless step 1 is committed unchanged),
     optionally locks. Commit that.

``python -m p3_det.decisions verify decisions/p3_det_conf_threshold.json`` prints the file's git
history and checks that the rule commit precedes the first commit with a result.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Refused(RuntimeError):
    """The tool will not proceed (inputs invalid, rule not committed, lock not allowed, ...)."""


def new_decision(decision_id: str, title: str, criterion: str, candidates: list, data_used: list[str],
                 today: str | None = None, reviewed_by: str = "p4") -> dict:
    return {
        "decision_id": decision_id, "owner": "p3", "title": title,
        "criterion_preregistered": criterion, "preregistered_at": today or date.today().isoformat(),
        "candidates": candidates, "value": {}, "evidence": [], "data_used": data_used,
        "reviewed_by": reviewed_by, "guide_ack": False, "locked": False, "locked_at": None, "git_sha": None,
    }


def save(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def write_criterion(path: Path, template: dict) -> dict:
    """Create the decision file from ``template``; never rewrite an existing, different criterion."""
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("criterion_preregistered") != template["criterion_preregistered"]:
            raise Refused(f"{path.name} exists with a different criterion; criteria are never rewritten "
                          "(raise it with P4 instead)")
        return old
    save(path, template)
    return template


def load_for_result(path: Path, criterion: str) -> dict:
    if not path.exists():
        raise Refused(f"{path} missing: write the criterion and commit it first")
    doc = json.loads(path.read_text())
    if doc.get("criterion_preregistered") != criterion:
        raise Refused("criterion in the decision file differs from the pre-registered one")
    if doc.get("locked"):
        raise Refused(f"{doc.get('decision_id')} is already locked")
    return doc


# --- git ---------------------------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout


def _rel_in_repo(path: Path) -> tuple[Path, str]:
    top = Path(_git(path.resolve().parent, "rev-parse", "--show-toplevel").strip()).resolve()
    return top, path.resolve().relative_to(top).as_posix()


def head_sha(path: Path) -> str:
    return _git(path.resolve().parent, "rev-parse", "HEAD").strip()


@dataclass(frozen=True)
class Commit:
    sha: str
    date: str
    subject: str
    criterion: str | None
    has_result: bool
    locked: bool


def history(path: Path) -> list[Commit]:
    """Commits touching ``path``, oldest first, with what each committed version contained."""
    try:
        top, rel = _rel_in_repo(path)
        lines = _git(top, "log", "--reverse", "--format=%H%x09%cI%x09%s", "--", rel).splitlines()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return []
    out = []
    for line in lines:
        sha, when, subject = line.split("\t", 2)
        try:
            d = json.loads(_git(top, "show", f"{sha}:{rel}"))
        except (subprocess.CalledProcessError, ValueError):
            d = {}
        out.append(Commit(sha, when, subject, d.get("criterion_preregistered"), bool(d.get("value")),
                          bool(d.get("locked"))))
    return out


def criterion_commit(path: Path, criterion: str) -> str | None:
    """First commit whose version holds ``criterion`` and no result yet; None if there is none."""
    for c in history(path):
        if c.criterion == criterion and not c.has_result:
            return c.sha
    return None


def verify_order(path: Path) -> tuple[bool, str]:
    """True if a rule-only commit precedes the first commit carrying a result."""
    h = history(path)
    if not h:
        return False, f"{path}: no git history (not committed, or not inside a git repository)"
    rows = [f"{c.sha[:10]}  {c.date}  {'rule ' if c.criterion else '-    '}"
            f"{'result' if c.has_result else '      '} {'LOCKED' if c.locked else '      '}  {c.subject}" for c in h]
    first_rule = next((i for i, c in enumerate(h) if c.criterion and not c.has_result), None)
    first_result = next((i for i, c in enumerate(h) if c.has_result), None)
    criteria = {c.criterion for c in h if c.criterion}
    if first_rule is None:
        verdict = "FAIL: no commit with the rule alone (rule and result were committed together)"
    elif len(criteria) > 1:
        verdict = "FAIL: the criterion text changed between commits"
    elif first_result is None:
        verdict = "OK so far: rule committed, no result committed yet"
    elif first_rule < first_result:
        verdict = f"OK: rule commit {h[first_rule].sha[:10]} precedes result commit {h[first_result].sha[:10]}"
    else:
        verdict = "FAIL: result committed before the rule"
    return verdict.startswith("OK"), "\n".join(rows + [verdict])


def repo_rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="show git history of a decision file and check rule-before-result")
    v.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args(argv)
    ok_all = True
    for p in a.paths:
        ok, report = verify_order(p)
        print(f"== {p}\n{report}\n")
        ok_all &= ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
