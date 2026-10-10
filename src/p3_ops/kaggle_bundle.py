"""Zip the P3 code for upload as a PRIVATE Kaggle dataset (there is no team git repo yet).

    python -m p3_ops.kaggle_bundle            # -> artifacts/p3/kaggle/p3_code_bundle.zip

Contains code only: src/p3_det, src/p3_ops, src/p3_xai, notebooks/p3 (*.py, *.ipynb), requirements/p3.txt.
No data, no checkpoints, no decisions, no third_party (the notebook clones D-FINE at the pinned commit).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "p3" / "kaggle" / "p3_code_bundle.zip"
INCLUDE = [("src/p3_det", "*.py"), ("src/p3_det", "*.yaml"), ("src/p3_ops", "*.py"), ("src/p3_xai", "*.py"),
           ("notebooks/p3", "*.py"), ("notebooks/p3", "*.ipynb")]
FILES = ["requirements/p3.txt"]
ALLOWED_SUFFIXES = {".py", ".yaml", ".ipynb", ".txt"}


def bundle_files(repo: Path = REPO) -> list[Path]:
    out = [repo / f for f in FILES]
    for folder, pattern in INCLUDE:
        out += [p for p in (repo / folder).rglob(pattern) if "__pycache__" not in p.parts]
    return sorted(set(out))


def build(out: Path = OUT, repo: Path = REPO) -> Path:
    files = bundle_files(repo)
    bad = [p for p in files if p.suffix not in ALLOWED_SUFFIXES]
    if bad:
        raise RuntimeError(f"unexpected file types in the bundle: {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(repo).as_posix())
    return out


def main() -> int:
    p = build()
    n = len(zipfile.ZipFile(p).namelist())
    print(f"wrote {p} ({n} files, {p.stat().st_size / 1e6:.2f} MB). Upload it as a PRIVATE Kaggle dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
