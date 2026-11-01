# -*- coding: utf-8 -*-
"""SHA-256 manifest for the collaboration package, in sha256sum format."""
import hashlib
import io
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
OUTNAME = "PACKAGE_MANIFEST.sha256"


def main():
    lines, n = [], 0
    for p in sorted(RUN.rglob("*")):
        if not p.is_file() or p.name == OUTNAME or "__pycache__" in p.parts:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append("%s  %s" % (h, p.relative_to(RUN).as_posix()))
        n += 1
    io.open(RUN / OUTNAME, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("%d files hashed into %s" % (n, OUTNAME))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
