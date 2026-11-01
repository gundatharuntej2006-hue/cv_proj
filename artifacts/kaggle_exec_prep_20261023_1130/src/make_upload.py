# -*- coding: utf-8 -*-
"""Build the two folders to upload to Kaggle, and verify they match the notebook.

One command, two folders, nothing to decide:

    py -3 src/make_upload.py            # dry run: says what it would build
    py -3 src/make_upload.py --write    # builds them

    D:/tb-triage-kaggle-upload/code/    -> upload as the CODE dataset
    D:/tb-triage-kaggle-upload/data/    -> upload as the DATA dataset

`code/` holds exactly the modules and configs the notebook imports — no `__pycache__`,
no tests, no documentation. `data/` holds the single zip built by `stage_kaggle_data.py`,
rebuilt here if it is missing or stale.

The last step checks the notebook's own discovery globs against the folder layout, under
both of the shapes Kaggle produces (contents at the dataset root, or wrapped in one
folder). If that check fails, the upload would not be found at run time and the script
refuses to call itself done.
"""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

import paths as P                                                   # noqa: E402

OUT = Path("D:/tb-triage-kaggle-upload")
STAGED = Path("D:/tb-triage-kaggle-staging/concept-audit-tune")
ZIP = STAGED.with_suffix(".zip")

# Exactly what the notebook imports, and nothing else.
CODE_FILES = ["src/paths.py", "src/concept_audit.py", "src/preflight.py",
              "src/stage_kaggle_data.py", "src/build_notebook.py", "src/make_upload.py",
              "configs/concept_indices.json", "configs/model_assets.json",
              "configs/tune_cohort_fingerprint.json", "requirements.txt"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_notebook_contract(code_root, zip_name, log=print):
    """The notebook finds its inputs by glob. Verify those globs against this layout.

    Simulated for both Kaggle shapes: files at the dataset root, and files wrapped in one
    extra folder. A mismatch here is a run that dies at cell 1 or cell 3.
    """
    nb = json.loads((PKG / "kaggle/concept_audit_tune_only.ipynb").read_text(
        encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    if "rglob('concept_audit.py')" not in src:
        raise SystemExit("the notebook no longer searches recursively for the code "
                         "package; rebuild it with build_notebook.py before uploading")

    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for shape, prefix in (("contents at dataset root", ""),
                              ("wrapped in a folder", "code/"),
                              ("wrapped twice", "datasets/code/")):
            root = Path(td) / shape.replace(" ", "_")
            ds = root / "tb-triage-clear-audit-code"
            (ds / prefix / "src").mkdir(parents=True)
            (ds / prefix / "src/concept_audit.py").write_text("x", encoding="utf-8")
            hit = next(iter(sorted(root.rglob("concept_audit.py"))), None)
            log("  code dataset, %-26s -> %s" % (shape, "found" if hit else "NOT FOUND"))
            ok &= bool(hit)

            dr = root / "data-ds"
            (dr / prefix).mkdir(parents=True, exist_ok=True)
            (dr / prefix / zip_name).write_bytes(b"")
            zhit = list(root.rglob(zip_name))
            log("  data dataset, %-26s -> %s" % (shape, "found" if zhit else "NOT FOUND"))
            ok &= bool(zhit)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--rebuild-zip", action="store_true",
                    help="re-stage and re-zip the data even if the zip already exists")
    a = ap.parse_args()

    print("CODE dataset -> %s" % (OUT / "code"))
    for rel in CODE_FILES:
        src = PKG / rel
        if not src.is_file():
            raise SystemExit("ABORT: %s is missing from the package" % rel)
        print("   %-40s %7d B  %s" % (rel, src.stat().st_size, sha(src)[:16]))

    print("\nDATA dataset -> %s" % (OUT / "data"))
    need_zip = a.rebuild_zip or not ZIP.is_file()
    if need_zip and a.write:
        print("   building the zip with stage_kaggle_data.py --write --zip ...")
        subprocess.run([sys.executable, str(HERE / "stage_kaggle_data.py"),
                        "--write", "--zip"], check=True)
    if ZIP.is_file():
        print("   %-40s %7.1f MB" % (ZIP.name, ZIP.stat().st_size / 1e6))
    else:
        print("   %-40s  will be built on --write" % (ZIP.name,))

    print("\nnotebook discovery contract")
    if not check_notebook_contract(OUT / "code", ZIP.name):
        raise SystemExit("ABORT: the notebook would not find these uploads")

    if not a.write:
        print("\nDRY RUN. Nothing was written. Re-run with --write.")
        return

    for sub in ("code", "data"):
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for rel in CODE_FILES:
        dst = OUT / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PKG / rel, dst)
    shutil.copy2(ZIP, OUT / "data" / ZIP.name)

    stray = [p.relative_to(OUT).as_posix() for p in (OUT / "code").rglob("*")
             if p.is_file() and p.relative_to(OUT / "code").as_posix() not in CODE_FILES]
    if stray:
        raise SystemExit("ABORT: unexpected files in the code upload: %s" % stray)

    lines = ["# Upload checksums, %s" % time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    for sub in ("code", "data"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file():
                lines.append("%s  %s" % (sha(p), p.relative_to(OUT).as_posix()))
    (OUT / "UPLOAD_CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Last check: the package can actually read the data upload it just produced.
    import os
    os.environ["TB_AUDIT_DATA"] = str(OUT / "data" / ZIP.name)
    import importlib
    importlib.reload(P)
    import concept_audit as A
    importlib.reload(A)
    p = P.resolve()
    d = A.load_tune_only(p)
    missing, extra = A.check_images(p, d["pid"])
    print("\nread-back check")
    print("   layout            : %s" % p["layout"])
    print("   cases             : %d (%d TB-positive, %d previous TB)"
          % (len(d["pid"]), int(d["y"].sum()), int(d["past_tb"].sum())))
    print("   fingerprint       : %s  (matches pinned)" % d["fingerprint"][:16])
    print("   images            : %d missing, %d not approved" % (len(missing), len(extra)))
    if missing or extra:
        raise SystemExit("ABORT: the produced upload does not read back cleanly")

    print("\nBuilt. Upload these two folders as PRIVATE datasets:")
    print("   CODE : %s   (%d files)" % (OUT / "code", len(CODE_FILES)))
    print("   DATA : %s   (1 file, %.0f MB)"
          % (OUT / "data", ZIP.stat().st_size / 1e6))
    print("   checksums: %s" % (OUT / "UPLOAD_CHECKSUMS.txt"))


if __name__ == "__main__":
    main()
