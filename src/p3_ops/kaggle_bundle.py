"""Zip the current P3 code and configuration snapshots for a PRIVATE Kaggle dataset.

    python -m p3_ops.kaggle_bundle            # -> artifacts/p3/kaggle/p3_code_bundle.zip

Contains code only: src/p3_det, src/p3_ops, src/p3_xai, notebooks/p3 (*.py, *.ipynb), requirements/p3.txt.
No images, checkpoints, credentials or third_party. Includes D08/D09 and P3
decision snapshots so training cannot silently switch to missing preprocessing.
The hash manifest identifies the exact local sources, including uncommitted fixes.
"""
from __future__ import annotations

import sys
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "p3" / "kaggle" / "p3_code_bundle.zip"
INCLUDE = [("src/p3_det", "*.py"), ("src/p3_det", "*.yaml"), ("src/p3_ops", "*.py"), ("src/p3_xai", "*.py"),
           ("notebooks/p3", "*.py"), ("notebooks/p3", "*.ipynb")]
FILES = ["requirements/p3.txt", "decisions/p2_clahe.json", "decisions/p2_crop_margin.json"]
ALLOWED_SUFFIXES = {".py", ".yaml", ".ipynb", ".txt", ".json"}


def bundle_files(repo: Path = REPO) -> list[Path]:
    out = [repo / f for f in FILES]
    for folder, pattern in INCLUDE:
        out += [p for p in (repo / folder).rglob(pattern) if "__pycache__" not in p.parts]
    out += list((repo / "decisions").glob("p3_*.json"))
    plan = repo / "decisions/p4_analysis_plan.json"
    if plan.is_file():
        out.append(plan)
    return sorted(set(out))


def build(out: Path = OUT, repo: Path = REPO) -> Path:
    files = bundle_files(repo)
    bad = [p for p in files if p.suffix not in ALLOWED_SUFFIXES]
    if bad:
        raise RuntimeError(f"unexpected file types in the bundle: {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "p3_code_snapshot_v1", "files": {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "note": "Configuration snapshot; does not establish decision approval or pre-registration"}
    try:
        manifest["git_sha"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        manifest["working_tree_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest["git_sha"], manifest["working_tree_dirty"] = None, None
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(repo).as_posix())
        z.writestr("p3_bundle_manifest.json", json.dumps(manifest, indent=2))
    return out


def main() -> int:
    p = build()
    n = len(zipfile.ZipFile(p).namelist())
    print(f"wrote {p} ({n} files, {p.stat().st_size / 1e6:.2f} MB). Upload it as a PRIVATE Kaggle dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
