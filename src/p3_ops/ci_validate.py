"""CI step: validate P3's contract files (freshly generated mocks of C10, C11, C15) with P4's tbcore validators.

    python -m p3_ops.ci_validate

Exit 0 only if tbcore is installed, provides every assumed name (p3_ops._tbcore_adapter.ASSUMED_API) and
accepts all files. CI runs this as a HARD step as soon as src/tbcore/ exists (before that the workflow prints
"SKIPPED: waiting on P4 tbcore" to the job summary).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import _tbcore_adapter as ad

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if not ad.tbcore_available():
        print("tbcore is not importable: cannot validate (this step must not run before src/tbcore/ exists)")
        return 2
    missing = ad.check_api()
    if missing:
        print(f"tbcore does not provide {missing}: update ASSUMED_API in src/p3_ops/_tbcore_adapter.py")
        return 1
    sys.path.insert(0, str(REPO / "mock"))
    import make_c10
    import make_c11
    import make_c15
    with tempfile.TemporaryDirectory() as tmp:
        files = {"C10": make_c10.generate(root=tmp), "C11": make_c11.generate(root=tmp),
                 "C15": make_c15.generate(root=tmp)}
        for cid, path in files.items():
            ad.validate(cid, path)                     # raises on an invalid file
            print(f"{cid}: {Path(path).name} valid (tbcore)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
