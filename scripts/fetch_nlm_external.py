# -*- coding: utf-8 -*-
"""Download the NLM Shenzhen and Montgomery TB chest X-ray sets.

These are the project's external validation sites (methodology Section 13, task P1-11).
The work plan said they were request-only from the US National Library of Medicine and
had to be requested in Week 1. That is out of date: as of 2026-10-06 both sets sit in a
public S3-backed bucket with open directory listings and download without any form,
email or agreement. Verified by downloading from it.

    https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/

What arrives:

    Shenzhen    662 images   3.77 GB   + 250 consensus TB boxes + lung masks + clinical readings
    Montgomery  138 images   0.61 GB   + 109 consensus TB boxes + left/right lung masks

The consensus ROI files matter more than their size suggests. They are radiologist TB
bounding boxes on *external* sites, which means lesion localisation can be evaluated
under a site change and not only on TBX11K. Very few published TB papers do that.

Ownership note: the external manifest (C02) is Person 1's contract, task P1-11. This
script only fetches bytes and records checksums; it writes no contract file. P1 builds
C02 from what lands here.

    py -3 scripts/fetch_nlm_external.py --out data/raw --what all
    py -3 scripts/fetch_nlm_external.py --out data/raw --what annotations   # 25 MB, no images
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
SETS = {
    "shenzhen": "Shenzhen-Hospital-CXR-Set",
    "montgomery": "Montgomery-County-CXR-Set/MontgomerySet",
}
IMAGE_DIRS = ("CXR_png",)
ANNOT_DIRS = ("ClinicalReadings", "ManualMask", "Annotations")


def get_json(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def walk(prefix):
    """Every file under a bucket prefix, following CommonPrefixes. Yields (key, size)."""
    listing = get_json("%s/%s/index.json" % (BASE, prefix))
    for c in listing.get("Contents") or []:
        key = c["Key"]
        if key.startswith("index."):
            continue
        yield "%s/%s" % (prefix, key), c["Size"]
    for sub in listing.get("CommonPrefixes") or []:
        yield from walk("%s/%s" % (prefix, sub["Prefix"].rstrip("/")))


def download(rel, size, out_root, retries=3):
    """Skips a file already present at the right size, so the run is resumable."""
    dest = out_root / rel
    if dest.exists() and dest.stat().st_size == size:
        return dest, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = "%s/%s" % (BASE, rel.replace(os.sep, "/"))
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(tmp, dest)
            return dest, True
        except Exception as e:
            if k == retries - 1:
                raise RuntimeError("failed after %d tries: %s (%s)" % (retries, url, e))
            time.sleep(3 * (k + 1))


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Fetch the NLM Shenzhen/Montgomery TB sets")
    ap.add_argument("--out", default="data/raw", help="root to write under")
    ap.add_argument("--sets", default="shenzhen,montgomery")
    ap.add_argument("--what", default="all", choices=["all", "images", "annotations", "list"],
                    help="'list' only prints what would be fetched")
    ap.add_argument("--no-checksums", action="store_true", help="skip the SHA-256 pass")
    a = ap.parse_args()

    out_root = Path(a.out).resolve()
    wanted = [s.strip() for s in a.sets.split(",") if s.strip()]
    plan = []
    for name in wanted:
        if name not in SETS:
            raise SystemExit("unknown set %r; choose from %s" % (name, sorted(SETS)))
        print("listing %s ..." % name, flush=True)
        for rel, size in walk(SETS[name]):
            tail = rel[len(SETS[name]) + 1:]
            top = tail.split("/")[0]
            is_img = top in IMAGE_DIRS
            is_ann = top in ANNOT_DIRS or "/" not in tail        # loose files = readmes, ROI csvs
            if a.what == "images" and not is_img:
                continue
            if a.what == "annotations" and not is_ann:
                continue
            plan.append((name, rel, size))

    total = sum(s for _, _, s in plan)
    print("\n%d files, %.2f GB" % (len(plan), total / 1e9))
    for name in wanted:
        sub = [p for p in plan if p[0] == name]
        print("  %-12s %4d files  %6.2f GB" % (name, len(sub), sum(s for _, _, s in sub) / 1e9))
    if a.what == "list":
        return
    free = __import__("shutil").disk_usage(out_root.anchor).free
    if free < total * 1.2:
        raise SystemExit("only %.1f GB free at %s; need about %.1f GB"
                         % (free / 1e9, out_root.anchor, total * 1.2 / 1e9))

    t0, done, fetched = time.perf_counter(), 0, 0
    for i, (name, rel, size) in enumerate(plan, 1):
        _, got = download(rel, size, out_root)
        done += size
        fetched += int(got)
        if i % 25 == 0 or i == len(plan):
            el = time.perf_counter() - t0
            print("  %4d/%d  %6.2f GB  %5.1f MB/s" % (i, len(plan), done / 1e9,
                                                      done / 1e6 / max(el, 1e-9)), flush=True)
    print("\n%d files present, %d newly downloaded" % (len(plan), fetched))

    if not a.no_checksums:
        man = out_root / "nlm_external_checksums.csv"
        print("checksumming -> %s" % man, flush=True)
        with open(man, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "rel_path", "bytes", "sha256"])
            for name, rel, size in plan:
                w.writerow([name, rel, size, sha256(out_root / rel)])
        print("wrote %s (hand this to Person 1 for C02 and D01)" % man)


if __name__ == "__main__":
    sys.exit(main())
