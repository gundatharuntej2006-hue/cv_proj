# -*- coding: utf-8 -*-
"""P2-10: pull lateral chest X-rays out of the CheXpert Plus archive without downloading it.

CheXpert Plus ships as five zips of about 155 GB each. We need a few hundred lateral
views. A zip is randomly accessible and Azure honours Range requests, so only the wanted
members are transferred: roughly 700 MB instead of 155 GB.

    set CHEXPERT_PLUS_URL=<the SAS url for png_chexpert_plus_chunk_0.zip>
    py -3 scripts/p2_10_fetch_laterals.py --n 500

**The URL is a credential and must never be committed.** It carries a signed SAS token
tied to an accepted licence agreement. It is read from the environment, and the provenance
file records only the blob path with the query string stripped.

One image per patient. The archive holds several studies for many patients, and three
laterals of one person are not three independent OOD samples: they would shrink the
effective n while appearing to grow it, and any per-patient quirk would be counted three
times. Selection is a seeded sample over patients, so the set is reproducible.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2_gate import remote_zip as RZ                                 # noqa: E402

PATIENT = re.compile(r"(patient\d+)")


def main():
    ap = argparse.ArgumentParser(description="P2-10: fetch lateral CXRs from CheXpert Plus")
    ap.add_argument("--url", default=os.environ.get("CHEXPERT_PLUS_URL"),
                    help="SAS url of a png_chexpert_plus_chunk_N.zip (or set CHEXPERT_PLUS_URL)")
    ap.add_argument("--out", default="data/raw/ood/lateral_chexpertplus")
    ap.add_argument("--n", type=int, default=500, help="how many patients to sample")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--view", default="lateral", choices=["lateral", "frontal"],
                    help="frontal is a CONTROL set, not OOD: same archive, resolution and "
                         "preprocessing as the laterals, so it isolates view from source")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.url:
        raise SystemExit("no url: pass --url or set CHEXPERT_PLUS_URL. Never commit it.")

    root = Path(__file__).resolve().parents[1]
    out_dir = root / a.out

    print("reading the archive index ...")
    t0 = time.perf_counter()
    members = RZ.central_directory(a.url)
    lateral = {n: m for n, m in members.items() if a.view in n.lower()}
    print("  %d members, %d lateral, index read in %.1f s"
          % (len(members), len(lateral), time.perf_counter() - t0))

    by_patient = defaultdict(list)
    for name in sorted(lateral):
        g = PATIENT.search(name)
        if g:
            by_patient[g.group(1)].append(name)
    print("  %d distinct patients with a lateral view" % len(by_patient))

    rng = random.Random(a.seed)
    patients = sorted(by_patient)
    rng.shuffle(patients)
    chosen = []
    for p in patients[:a.n]:
        chosen.append(sorted(by_patient[p])[0])        # earliest study, deterministic
    chosen.sort()

    total = sum(lateral[n].file_size for n in chosen)
    print("\nselected %d images, one per patient, about %.0f MB" % (len(chosen), total / 1e6))
    if a.dry_run:
        for n in chosen[:10]:
            print("   ", n)
        return

    t1 = time.perf_counter()
    written = RZ.fetch_many(a.url, [lateral[n] for n in chosen], out_dir)
    secs = time.perf_counter() - t1

    blob = urlsplit(a.url)                              # drop the query: it is the token
    prov = {
        "task": "P2-10", "ood_type": a.view + ("_cxr" if a.view=="lateral" else "_cxr_CONTROL_not_ood"),
        "source": "CheXpert Plus (Stanford AIMI), PNG chunk",
        "blob": "%s://%s%s" % (blob.scheme, blob.netloc, blob.path),
        "licence": "Stanford University Dataset Research Use Agreement",
        "licence_accepted_by": "Atul Dhull, 2026-10-06",
        "archive_bytes": RZ.probe(a.url)["size"],
        "members_in_archive": len(members),
        "lateral_members": len(lateral),
        "patients_with_lateral": len(by_patient),
        "selection": "one image per patient (earliest study), seeded sample over patients",
        "seed": a.seed, "n_images": len(written),
        "bytes_transferred_approx": total,
        "fetch_seconds": round(secs, 1),
        "files": [p.name for p in written],
        "note": ("Only the selected members were transferred, via HTTP Range. The SAS token is "
                 "deliberately absent from this record."),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(prov, open(out_dir / "_provenance.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote %d files to %s in %.0f s (%.0f MB)"
          % (len(written), out_dir, secs, total / 1e6))
    print("provenance: %s" % (out_dir / "_provenance.json"))


if __name__ == "__main__":
    main()
