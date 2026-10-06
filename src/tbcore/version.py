"""Version stamps and governance hash calculation utilities."""

import hashlib
import json
import subprocess
from pathlib import Path
from tbcore.paths import get_repo_root, get_decisions_dir

MODEL_VERSION = "reg-v1-3b9c"
SPLIT_VERSION = "v1"
SCHEMA_VERSION = "1.0"


def get_git_sha() -> str:
    """Returns the current git short commit SHA, or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=get_repo_root(),
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "c0ffee1"


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file."""
    p = Path(file_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_decisions_hash() -> str:
    """Computes a composite SHA-256 hash over all locked decision JSON files (D01-D23)."""
    dec_dir = get_decisions_dir()
    if not dec_dir.exists():
        return "sha256:none"

    json_files = sorted(dec_dir.glob("*.json"))
    if not json_files:
        return "sha256:empty"

    h = hashlib.sha256()
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Normalize dictionary for reproducible hash
            canon = json.dumps(data, sort_keys=True)
            h.update(jf.name.encode("utf-8"))
            h.update(canon.encode("utf-8"))
        except Exception:
            continue

    return f"sha256:{h.hexdigest()[:16]}"


def get_version_info() -> dict:
    """Returns standard versions block for CaseResult and prediction logs."""
    return {
        "model_version": MODEL_VERSION,
        "split_version": SPLIT_VERSION,
        "decisions_hash": compute_decisions_hash(),
        "git_sha": get_git_sha(),
    }
