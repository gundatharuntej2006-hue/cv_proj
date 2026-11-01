# -*- coding: utf-8 -*-
"""Phase 6: SHA-256 manifest of every file this run created."""
import hashlib
import io
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for p in sorted(RUN.rglob("*")):
        if p.is_file() and p.name != "FILE_MANIFEST.md":
            rel = p.relative_to(RUN).as_posix()
            rows.append((rel, p.stat().st_size,
                         hashlib.sha256(p.read_bytes()).hexdigest()))
    body = ["# File manifest — `prevtb_subgroup_20261021_1055`", "",
            "SHA-256 of every file created by this run. "
            "`FILE_MANIFEST.md` itself is excluded.", "",
            "| file | bytes | sha256 |", "|---|---|---|"]
    body += ["| `%s` | %d | `%s` |" % r for r in rows]
    io.open(RUN / "FILE_MANIFEST.md", "w", encoding="utf-8").write("\n".join(body) + "\n")
    print("%d files" % len(rows))
    for rel, n, h in rows:
        print("  %-52s %8d  %s" % (rel, n, h[:16]))


if __name__ == "__main__":
    main()
