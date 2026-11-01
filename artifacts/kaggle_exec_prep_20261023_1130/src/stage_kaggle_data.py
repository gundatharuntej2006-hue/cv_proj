# -*- coding: utf-8 -*-
"""Build the private Kaggle upload. LOCAL ONLY. Uploads nothing itself.

Writes a staging directory containing **only Vietnam TUNE material**:

  * the 1,352 TUNE images, copied by id from the allow-list
  * the case-scores CSV and the clinical CSV, filtered to TUNE rows
  * the split manifest (ids only), so the REPORT-overlap guard can still fire
  * tune_manifest.json: the sorted TUNE ids and their sha256 fingerprint

Vietnam REPORT clinical rows, REPORT scores and REPORT images are therefore not in the
upload at all. The allow-list in the loader remains the enforcement boundary; this is a
second, physical boundary in front of it.

The staging directory is written **outside** this artifact package, because it holds
images and clinical rows and the package must stay free of both.

    py -3 src/stage_kaggle_data.py                 # dry run: counts only, writes nothing
    py -3 src/stage_kaggle_data.py --write         # build the upload
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

import paths as P                                                   # noqa: E402

DEFAULT_OUT = Path("D:/tb-triage-kaggle-staging/concept-audit-tune")


def fingerprint(ids):
    """sha256 over the sorted id list. Contains no identifiers once hashed."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids))
                          .encode("utf-8")).hexdigest()


def _filter_csv(src, dst, keep, write):
    rows = list(csv.DictReader(io.open(src, encoding="utf-8-sig")))
    sel = [r for r in rows if str(r["pid"]) in keep]
    if write:
        with io.open(dst, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sel)
    return len(rows), len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="actually copy files")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--zip", action="store_true",
                    help="also write <out>.zip, which is how it has to be uploaded: "
                         "Kaggle's uploader caps the number of files per dataset and "
                         "1,352 images exceeds it")
    a = ap.parse_args()

    p = P.resolve(prefer="repo")
    out = Path(a.out)
    print(P.describe(p))
    print("\nstaging to: %s%s\n" % (out, "" if a.write else "   (DRY RUN)"))

    split = json.load(io.open(p["split"], encoding="utf-8"))
    tune = [str(x) for x in split["tune"]]
    report = set(str(x) for x in split["report"])
    if set(tune) & report:
        raise SystemExit("ABORT: TUNE and REPORT overlap in the split manifest")
    if len(set(tune)) != len(tune):
        raise SystemExit("ABORT: duplicate id in the TUNE allow-list")
    keep = set(tune)

    missing = [t for t in tune if not (p["img_dir"] / ("%s.png" % t)).is_file()]
    if missing:
        raise SystemExit("ABORT: %d TUNE images are missing, first: %s"
                         % (len(missing), missing[:5]))

    if a.write:
        (out / P.STAGED_IMG_DIR).mkdir(parents=True, exist_ok=True)

    n_scores = _filter_csv(p["scores"], out / P.STAGED_FILES["scores"], keep, a.write)
    n_meta = _filter_csv(p["meta"], out / P.STAGED_FILES["meta"], keep, a.write)
    print("case scores : %d rows in source -> %d TUNE rows staged" % n_scores)
    print("clinical    : %d rows in source -> %d TUNE rows staged" % n_meta)
    if n_scores[1] != len(tune) or n_meta[1] != len(tune):
        raise SystemExit("ABORT: expected %d TUNE rows in both CSVs, got %d and %d"
                         % (len(tune), n_scores[1], n_meta[1]))

    fp = fingerprint(tune)
    man = {"cohort": "vietnam_tune", "n": len(tune),
           "fingerprint_sha256": fp,
           "split_source": p["split"].name,
           "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pids": sorted(tune)}
    if a.write:
        shutil.copy2(p["split"], out / P.STAGED_FILES["split"])
        json.dump(man, io.open(out / P.STAGED_FILES["manifest"], "w", encoding="utf-8"),
                  indent=1)

    nb = 0
    t0 = time.perf_counter()
    for i, t in enumerate(sorted(tune), 1):
        src = p["img_dir"] / ("%s.png" % t)
        nb += src.stat().st_size
        if a.write:
            shutil.copy2(src, out / P.STAGED_IMG_DIR / src.name)
            if i % 400 == 0:
                print("  %d/%d images (%.0f s)" % (i, len(tune), time.perf_counter() - t0))
    print("images      : %d, %.0f MB" % (len(tune), nb / 1e6))
    print("fingerprint : %s" % fp)

    expect = json.load(io.open(PKG / "configs/tune_cohort_fingerprint.json",
                               encoding="utf-8"))
    if expect["fingerprint_sha256"] != fp:
        raise SystemExit("ABORT: cohort fingerprint %s does not match the pinned %s.\n"
                         "The TUNE allow-list has changed since this package was prepared."
                         % (fp, expect["fingerprint_sha256"]))
    print("fingerprint matches the pinned value: OK")

    if not a.write:
        print("\nDRY RUN. Nothing was written. Re-run with --write.")
        return
    target = out
    if a.zip:
        import zipfile
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for q in sorted(out.rglob("*")):
                if q.is_file():
                    zf.write(q, q.relative_to(out.parent).as_posix())
        print("zip         : %s  (%.0f MB)" % (z, z.stat().st_size / 1e6))
        print("              PNGs are already compressed, so this is STORED rather than "
              "deflated:\n              a container to get under the file-count cap, not "
              "a size reduction.")
        target = z

    print("\nStaged. Upload %s as a PRIVATE Kaggle dataset.\n"
          "It contains de-identified images and clinical rows: do not make it public."
          % target)


if __name__ == "__main__":
    main()
