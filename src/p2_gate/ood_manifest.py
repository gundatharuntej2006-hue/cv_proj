# -*- coding: utf-8 -*-
"""P2-10: build and freeze the held-out OOD collection (contract C03).

Acceptance criteria from the work plan:
  ">= 200 images per ood_type in ood_eval; role-disjointness test; C03 SHA in the D13
   draft precedes the first OOD score"

That last clause is handshake H5 and it is the reason this module exists rather than a
notebook. The manifest must be **frozen before anything is scored against it**, so that
the OOD set cannot be quietly adjusted once the gate's numbers are known. `freeze()`
writes the CSV and a SHA-256 sidecar; D13 records that SHA; the gate refuses to report an
OOD number whose manifest SHA does not match.

Two rules enforced here, both of which are easy to break by hand:

**No source enters without a verified licence.** A source carries its licence, the URL the
licence was read from, and who verified it. `licence_verified=False` is refused unless the
caller explicitly passes `allow_unverified=True`, and every such row is then stamped so it
cannot be mistaken for cleared data later. This is D15's rule, applied to data rather than
to model weights.

**The lateral pool is split disjointly.** Laterals do double duty: they are OOD for the
gate and they are training data for Person 1's view-head extension. An image used to train
the view head must never appear in the sealed `ood_eval` set, or the extension is scored on
its own training data. Splitting happens once, here, by seeded hash of the image id, so it
is reproducible and cannot drift between runs.
"""
import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

C03_COLUMNS = ["image_id", "rel_path", "source", "ood_type", "role", "licence"]
OOD_TYPES = ["lateral_cxr", "other_body_part", "non_medical"]
ROLES = ["ood_eval", "viewhead_train", "viewhead_dev", "viewhead_test"]
VIEWHEAD_ROLES = ["viewhead_train", "viewhead_dev", "viewhead_test"]
MIN_PER_TYPE = 200                      # work plan acceptance criterion
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class OodSource:
    """One place images come from, with the licence we are relying on to use them."""
    name: str
    root: Path
    ood_type: str
    licence: str
    licence_url: str = ""
    licence_verified: bool = False
    verified_by: str = ""
    verified_on: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.ood_type not in OOD_TYPES:
            raise ValueError("ood_type %r not in %s" % (self.ood_type, OOD_TYPES))
        self.root = Path(self.root)

    def images(self):
        return sorted(p for p in self.root.rglob("*")
                      if p.suffix.lower() in IMAGE_SUFFIXES)


def _bucket(image_id: str, seed: int, n_buckets: int = 1000) -> int:
    """Deterministic bucket from a seeded hash of the id. Not rng.shuffle: adding a new
    source must not reshuffle images that were already assigned a role."""
    h = hashlib.sha256(("%d:%s" % (seed, image_id)).encode()).hexdigest()
    return int(h[:8], 16) % n_buckets


def build(sources: List[OodSource], seed: int = 20260929,
          viewhead_share: float = 0.5,
          viewhead_fracs=(0.6, 0.2, 0.2),
          allow_unverified: bool = False) -> List[dict]:
    """Returns C03 rows. Laterals are split between `ood_eval` and the view-head pools;
    every other type is entirely `ood_eval`.

    `viewhead_share` is the fraction of laterals handed to the view head. The rest stay
    sealed for OOD evaluation.
    """
    if abs(sum(viewhead_fracs) - 1.0) > 1e-9:
        raise ValueError("viewhead_fracs must sum to 1, got %s" % (viewhead_fracs,))

    rows: List[dict] = []
    for src in sources:
        if not src.licence_verified and not allow_unverified:
            raise ValueError(
                "source %r has licence_verified=False and would enter the committed C03 "
                "file. Verify the licence at its primary source first, or pass "
                "allow_unverified=True knowing every row is stamped UNVERIFIED and the "
                "gate will refuse to report it." % src.name)
        licence = src.licence if src.licence_verified else "UNVERIFIED:" + src.licence
        images = src.images()
        if not images:
            raise ValueError("source %r has no images under %s" % (src.name, src.root))

        for p in images:
            image_id = "ood_%s_%s" % (src.ood_type[:3], p.stem)
            rel = p.as_posix()
            if src.ood_type != "lateral_cxr":
                role = "ood_eval"
            else:
                b = _bucket(image_id, seed)
                if b >= viewhead_share * 1000:
                    role = "ood_eval"
                else:                               # inside the view-head share
                    sub = _bucket(image_id, seed + 1)
                    c1 = viewhead_fracs[0] * 1000
                    c2 = (viewhead_fracs[0] + viewhead_fracs[1]) * 1000
                    role = ("viewhead_train" if sub < c1 else
                            "viewhead_dev" if sub < c2 else "viewhead_test")
            rows.append({"image_id": image_id, "rel_path": rel, "source": src.name,
                         "ood_type": src.ood_type, "role": role, "licence": licence})
    return rows


def validate(rows: List[dict], min_per_type: int = MIN_PER_TYPE) -> List[str]:
    """Every check the work plan asks for. Returns a list of problems; empty means pass."""
    problems = []
    if not rows:
        return ["C03 is empty"]

    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dupes = [i for i in set(ids) if ids.count(i) > 1][:5]
        problems.append("duplicate image_id, e.g. %s" % dupes)

    for r in rows:
        if sorted(r) != sorted(C03_COLUMNS):
            problems.append("row has columns %s, expected %s" % (sorted(r), sorted(C03_COLUMNS)))
            break
    for r in rows:
        if r["ood_type"] not in OOD_TYPES:
            problems.append("bad ood_type %r" % r["ood_type"]); break
    for r in rows:
        if r["role"] not in ROLES:
            problems.append("bad role %r" % r["role"]); break

    # >= 200 per ood_type in the sealed evaluation set
    for t in OOD_TYPES:
        n = sum(1 for r in rows if r["ood_type"] == t and r["role"] == "ood_eval")
        if n < min_per_type:
            problems.append("ood_type %s has %d images in ood_eval, need >= %d"
                            % (t, n, min_per_type))

    # role disjointness: no image in two roles, and no view-head image in ood_eval
    by_id: Dict[str, set] = {}
    for r in rows:
        by_id.setdefault(r["image_id"], set()).add(r["role"])
    multi = [i for i, rs in by_id.items() if len(rs) > 1]
    if multi:
        problems.append("images carrying more than one role, e.g. %s" % multi[:5])

    vh = {r["image_id"] for r in rows if r["role"] in VIEWHEAD_ROLES}
    ev = {r["image_id"] for r in rows if r["role"] == "ood_eval"}
    overlap = vh & ev
    if overlap:
        problems.append("%d images are in both a view-head pool and ood_eval, e.g. %s"
                        % (len(overlap), sorted(overlap)[:5]))

    # only laterals may serve the view head
    bad_vh = {r["image_id"] for r in rows
              if r["role"] in VIEWHEAD_ROLES and r["ood_type"] != "lateral_cxr"}
    if bad_vh:
        problems.append("non-lateral images in a view-head pool, e.g. %s" % sorted(bad_vh)[:5])

    unver = [r["source"] for r in rows if r["licence"].startswith("UNVERIFIED:")]
    if unver:
        problems.append("sources with an unverified licence: %s" % sorted(set(unver)))

    return problems


def freeze(rows: List[dict], out_path, extra: Optional[dict] = None) -> dict:
    """Write C03 and its SHA-256 sidecar. H5: this SHA goes into D13 *before* the first
    OOD score exists, so the collection cannot be adjusted once the gate's numbers are
    known."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, C03_COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image_id"]))

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        counts.setdefault(r["ood_type"], {}).setdefault(r["role"], 0)
        counts[r["ood_type"]][r["role"]] += 1
    meta = {"contract": "C03", "task": "P2-10", "frozen_on": str(date.today()),
            "path": out.as_posix(), "sha256": sha, "n_rows": len(rows),
            "counts": counts,
            "sources": sorted({r["source"] for r in rows}),
            "note": ("H5: this SHA must appear in the D13 draft before any OOD score is "
                     "produced. A gate run whose manifest SHA differs is not reportable.")}
    if extra:
        meta.update(extra)
    json.dump(meta, open(str(out) + ".json", "w", encoding="utf-8"), indent=1)
    return meta
