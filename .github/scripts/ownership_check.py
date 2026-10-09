#!/usr/bin/env python3
"""P3-00: ownership check and data guard for CI.

Ownership: a branch named ``pN/...`` may only touch paths owned by person N
(or paths nobody owns). Frozen paths may not be touched by any ``pN/`` branch.
Data guard: no image file (.png .jpg .jpeg .dcm) may be added or modified
under ``data/``, whatever the branch.

Standard library only, so CI can run it before installing anything.

Usage:
    # PR / branch diff (what CI runs)
    python .github/scripts/ownership_check.py --base origin/main --head HEAD --branch p2/P2-11-knn-gate
    # what is staged right now (branch read from git)
    python .github/scripts/ownership_check.py --staged
    # data guard over every tracked file (CI on push to main)
    python .github/scripts/ownership_check.py --all-tracked

Exit code 0 = pass, 1 = violations, 2 = usage / git error.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase

FROZEN = "FROZEN"

# Pattern -> owner. A pattern matches a path when its "/"-separated segments
# match the leading segments of the path (fnmatch per segment), so "src/p1_*"
# covers "src/p1_data/api.py". The longest matching pattern wins.
OWNERSHIP: list[tuple[str, str]] = [
    # --- P3-00 task spec ---------------------------------------------------
    ("src/p1_*", "P1"), ("data", "P1"), ("docs/p1_*", "P1"), ("decisions/p1_*", "P1"),
    ("environment.yml", "P1"), ("pyproject.toml", "P1"), ("Makefile", "P1"),
    ("requirements/base.txt", "P1"), (".gitignore", "P1"), (".gitattributes", "P1"),
    (".pre-commit-config.yaml", "P1"),
    ("src/p2_*", "P2"), ("docs/p2_*", "P2"), ("decisions/p2_*", "P2"),
    ("src/p3_*", "P3"), ("third_party/D-FINE", "P3"), ("docs/p3_*", "P3"),
    ("decisions/p3_*", "P3"), (".github/workflows/ci.yml", "P3"),
    ("src/tbcore", "P4"), ("src/p4_*", "P4"), ("contracts", "P4"), ("README.md", "P4"),
    (".github/CODEOWNERS", "P4"), ("decisions/p4_*", "P4"), ("decisions/README.md", "P4"),
    ("docs/p4_*", "P4"),
    # --- Added from README section 3 / 4.6 (CODEOWNERS); not in the task spec -
    *[(p.format(n=n), f"P{n}") for n in range(1, 5) for p in (
        "tests/p{n}_*", "notebooks/p{n}", "artifacts/p{n}", "report/p{n}_*",
        "mk/p{n}.mk", "requirements/p{n}.txt")],
    ("tests/tbcore", "P4"),
    ("data/splits/ood_*", "P2"), ("report/main.md", "P2"), ("report/slides", "P2"),
    ("report/demo_video.md", "P3"), (".github/scripts/ownership_check.py", "P3"),
    ("docs/methodology.md", FROZEN), ("docs/proposal.md", FROZEN),
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".dcm")
BRANCH_RE = re.compile(r"^p([1-4])/")


@dataclass(frozen=True)
class Change:
    status: str  # git name-status letter: A, M, D, T, ...
    path: str


def _matches(pattern: str, path: str) -> bool:
    pat, parts = pattern.split("/"), path.split("/")
    return len(parts) >= len(pat) and all(fnmatchcase(s, p) for s, p in zip(parts, pat))


def owner_of(path: str) -> str | None:
    """Owner of ``path`` (P1..P4, FROZEN) or None if nobody owns it."""
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    best: tuple[int, str] | None = None
    for pattern, owner in OWNERSHIP:
        if _matches(pattern, path):
            depth = pattern.count("/") + 1
            if best is None or depth > best[0]:
                best = (depth, owner)
    return best[1] if best else None


def branch_person(branch: str) -> str | None:
    m = BRANCH_RE.match(branch or "")
    return f"P{m.group(1)}" if m else None


def check_ownership(branch: str, changes: list[Change]) -> list[str]:
    person = branch_person(branch)
    if person is None:
        return []
    errors = []
    for c in changes:
        owner = owner_of(c.path)
        if owner is not None and owner != person:
            what = "is frozen" if owner == FROZEN else f"is owned by {owner}"
            errors.append(f"ownership: {c.path} {what}; branch '{branch}' belongs to {person}")
    return errors


def check_data_guard(changes: list[Change]) -> list[str]:
    return [
        f"data guard: image file staged under data/: {c.path}"
        for c in changes
        if c.status != "D" and c.path.startswith("data/") and c.path.lower().endswith(IMAGE_EXTS)
    ]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def parse_name_status(out: str) -> list[Change]:
    changes = []
    for line in out.splitlines():
        if line.strip():
            status, path = line.split("\t", 1)
            changes.append(Change(status[0], path))
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="base ref; diff is base...head (merge-base)")
    mode.add_argument("--staged", action="store_true", help="check the git index")
    mode.add_argument("--all-tracked", action="store_true", help="data guard over every tracked file")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--branch", help="branch name (default: current branch)")
    args = ap.parse_args(argv)

    try:
        if args.all_tracked:
            changes = [Change("A", p) for p in _git("ls-files").splitlines() if p]
        elif args.staged:
            changes = parse_name_status(_git("diff", "--cached", "--name-status", "--no-renames"))
        else:
            changes = parse_name_status(
                _git("diff", "--name-status", "--no-renames", f"{args.base}...{args.head}"))
        branch = args.branch or _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    except subprocess.CalledProcessError as e:
        print(f"git failed: {e.stderr.strip()}", file=sys.stderr)
        return 2

    errors = check_data_guard(changes)
    if not args.all_tracked:
        errors += check_ownership(branch, changes)

    person = branch_person(branch)
    scope = "data guard only" if args.all_tracked else (
        f"ownership as {person}" if person else "data guard only (not a pN/ branch)")
    print(f"checked {len(changes)} path(s) on '{branch}' ({scope})")
    for e in errors:
        print(f"::error::{e}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"FAIL {e}")
    if not errors:
        print("OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
