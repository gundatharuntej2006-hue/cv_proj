"""Install/check the pinned D-FINE checkout without requiring a registered submodule.

python -m p3_det.setup [--fetch]
An existing checkout is never reset or overwritten.
"""
import argparse
import subprocess
from pathlib import Path

PIN = "956d1709314c2c6a4df6f34de232054578a7449f"
ROOT = Path(__file__).resolve().parents[2]


def setup(path, fetch=False):
    path = Path(path)
    if not path.exists():
        if not fetch:
            raise RuntimeError("D-FINE missing; rerun with --fetch to download the pinned checkout")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/Peterande/D-FINE.git", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", PIN], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise RuntimeError(f"D-FINE pin mismatch: {actual}; expected {PIN}. Existing checkout left unchanged.")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("D-FINE checkout has local changes; refusing to use unrecorded training code")
    return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    print(setup(ROOT / "third_party/D-FINE", a.fetch))


if __name__ == "__main__":
    main()
