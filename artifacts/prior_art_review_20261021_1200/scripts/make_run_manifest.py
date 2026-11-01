# -*- coding: utf-8 -*-
"""Append a SHA-256 table of this run's own outputs to RUN_MANIFEST.md."""
import hashlib
import io
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
MARK = "## Output hashes"


def main():
    rows = []
    for p in sorted(RUN.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "RUN_MANIFEST.md":
            rows.append((p.relative_to(RUN).as_posix(), p.stat().st_size,
                         hashlib.sha256(p.read_bytes()).hexdigest()))
    txt = io.open(RUN / "RUN_MANIFEST.md", encoding="utf-8").read()
    txt = txt.split("\n" + MARK)[0].rstrip("\n")
    txt += "\n\n" + MARK + "\n\nSHA-256 of every file this run created, "
    txt += "excluding `RUN_MANIFEST.md` itself.\n\n| file | bytes | sha256 |\n|---|---|---|\n"
    txt += "\n".join("| `%s` | %d | `%s` |" % r for r in rows) + "\n"
    io.open(RUN / "RUN_MANIFEST.md", "w", encoding="utf-8").write(txt)
    print("%d output files hashed" % len(rows))
    for r in rows:
        print("  %-52s %8d  %s" % (r[0], r[1], r[2][:16]))


if __name__ == "__main__":
    main()
