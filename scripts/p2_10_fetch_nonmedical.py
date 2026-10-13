# -*- coding: utf-8 -*-
"""P2-10: build the `non_medical` out-of-distribution pool from Wikimedia Commons.

    py -3 scripts/p2_10_fetch_nonmedical.py --n 250

Why Commons rather than an image dataset off the shelf: every file's licence and author
come back from the API per image, so provenance is recorded per file rather than asserted
for a whole corpus. D15's rule is that no source enters C03 without its licence read at
source, and this is the only non-medical option that satisfies it without an agreement.

Categories are chosen to be unambiguously non-medical — landscapes, vehicles, food,
architecture, animals. "Quality images" is a curated Commons grade, which keeps out
thumbnails, screenshots and clip art. A general CC-Zero sweep would be larger but could
quietly include a radiograph, which in a set defined as *non-medical* would be a silent
labelling error.

Only permissive licences are accepted, and each file's licence, author and source page
are written to the manifest. Attribution is required for the CC BY family; the manifest
is what the report's acknowledgements are built from.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "tb-triage-research/1.0 (BMSIT&M academic project; OOD evaluation set)"
CATEGORIES = [
    "Category:Quality images of landscapes",
    "Category:Quality images of vehicles",
    "Category:Quality images of food",
    "Category:Quality images of architecture",
    "Category:Quality images of animals",
]
# Accepted licences. Anything else is skipped rather than guessed at.
OK_LICENCE = re.compile(r"^(cc0|public domain|cc by( |-)\d|cc by-sa( |-)\d)", re.I)
TAG = re.compile(r"<[^>]+>")


def api(**params):
    params.update(format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def titles_from(category, want):
    """File titles from a category, following continuation."""
    out, cont = [], None
    while len(out) < want:
        p = dict(action="query", list="categorymembers", cmtitle=category,
                 cmtype="file", cmlimit="200")
        if cont:
            p["cmcontinue"] = cont
        r = api(**p)
        out += [m["title"] for m in r.get("query", {}).get("categorymembers", [])]
        cont = r.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return out[:want]


def image_records(titles, width=1024):
    """Licence, author and a scaled URL for each title, in batches of 50."""
    recs = []
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        r = api(action="query", titles="|".join(batch), prop="imageinfo",
                iiprop="url|extmetadata|size", iiurlwidth=str(width))
        for pg in r.get("query", {}).get("pages", []):
            ii = (pg.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata", {})
            lic = (em.get("LicenseShortName", {}) or {}).get("value", "")
            if not OK_LICENCE.match(lic.strip()):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            recs.append({
                "title": pg["title"],
                "url": url,
                "licence": lic.strip(),
                "licence_url": (em.get("LicenseUrl", {}) or {}).get("value", ""),
                "author": TAG.sub("", (em.get("Artist", {}) or {}).get("value", "")).strip()[:160],
                "credit": TAG.sub("", (em.get("Credit", {}) or {}).get("value", "")).strip()[:160],
                "description_page": ii.get("descriptionurl", ""),
            })
    return recs


def safe_name(title, idx):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title.replace("File:", ""))[:90]
    return "commons_%04d_%s" % (idx, stem)


def main():
    ap = argparse.ArgumentParser(description="P2-10: non-medical OOD pool from Wikimedia Commons")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default="data/raw/ood/non_medical_commons")
    ap.add_argument("--width", type=int, default=1024)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cat = a.n // len(CATEGORIES) + 15        # headroom for licence rejections
    print("listing %d categories, %d candidates each ..." % (len(CATEGORIES), per_cat))
    titles, seen = [], set()
    for c in CATEGORIES:
        got = [t for t in titles_from(c, per_cat) if t not in seen]
        seen.update(got)
        titles += got
        print("  %-45s %d" % (c.replace("Category:", ""), len(got)))

    print("\nreading licences ...")
    recs = image_records(titles, a.width)
    print("  %d of %d candidates carry an accepted licence" % (len(recs), len(titles)))
    recs = recs[:a.n]

    print("\ndownloading ...")
    kept, failed = [], 0
    for i, r in enumerate(recs, 1):
        dest = out_dir / (safe_name(r["title"], i) + ".jpg")
        if not dest.exists() or dest.stat().st_size == 0:
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = resp.read()
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            except Exception as e:
                failed += 1
                print("   failed %s: %s" % (r["title"][:50], str(e)[:60]))
                continue
        r["file"] = dest.name
        kept.append(r)
        if i % 50 == 0:
            print("   %d/%d" % (i, len(recs)))

    licences = {}
    for r in kept:
        licences[r["licence"]] = licences.get(r["licence"], 0) + 1
    prov = {
        "task": "P2-10", "ood_type": "non_medical",
        "source": "Wikimedia Commons, Quality images in five non-medical subject categories",
        "categories": CATEGORIES,
        "api": API,
        "selection": "curated Quality-image categories only; a general CC-Zero sweep was "
                     "rejected because it could silently include a radiograph in a set "
                     "defined as non-medical",
        "licence_filter": "CC0, Public Domain, CC BY, CC BY-SA only; anything else skipped",
        "licence_counts": licences,
        "attribution_required": True,
        "n_images": len(kept), "n_failed": failed,
        "max_width_px": a.width,
        "images": kept,
        "note": "Per-image licence, author and description page are recorded above. The "
                "report's acknowledgements are built from this file.",
    }
    json.dump(prov, open(out_dir / "_provenance.json", "w", encoding="utf-8"), indent=1)
    mb = sum((out_dir / r["file"]).stat().st_size for r in kept) / 1e6
    print("\n%d images, %.0f MB -> %s" % (len(kept), mb, out_dir))
    print("licences:", licences)
    print("provenance:", out_dir / "_provenance.json")


if __name__ == "__main__":
    sys.exit(main())
