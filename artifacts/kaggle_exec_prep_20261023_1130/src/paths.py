# -*- coding: utf-8 -*-
"""Where the inputs live, in each of the two environments this run supports.

The previous package hard-coded repository-relative paths (`ROOT = HERE.parents[2]`).
That is correct locally and **wrong on Kaggle**, where `src/` sits inside a read-only
input dataset: `parents[2]` resolves to `/kaggle`, and every data path underneath it is
missing. This module resolves the layout instead of assuming it, and fails with an
actionable message rather than a `FileNotFoundError` deep inside the loader.

Three layouts are recognised:

**STAGED** — a flat directory holding only TUNE material, produced by
`stage_kaggle_data.py` and uploaded as a private Kaggle dataset:

    vietnam_img512_tune/<pid>.png      only the 1,352 TUNE images
    vietnam_case_scores_tune.csv       TUNE rows only
    vietnam_meta_tune.csv              TUNE rows only
    vietnam_tune_report_split.json     full split (ids only) so the guards can fire
    tune_manifest.json                 sorted TUNE ids + sha256 of that id list

**STAGED_ZIP** — the same material inside a `.zip`, which is how it has to be uploaded in
practice: Kaggle's uploader caps the number of files per dataset, and 1,352 images exceeds
it. Images are read straight out of the archive, so nothing is extracted and no second
copy of the images is made. The four small files are extracted once, to a scratch
directory, because they are read line by line.

**REPO** — the working copy on this machine, with files in their original places.

A staged layout is preferred wherever both it and REPO exist, because it is the layout in
which Vietnam REPORT clinical rows and REPORT scores are not present at all.
"""
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

STAGED_FILES = {
    "scores": "vietnam_case_scores_tune.csv",
    "meta": "vietnam_meta_tune.csv",
    "split": "vietnam_tune_report_split.json",
    "manifest": "tune_manifest.json",
}
STAGED_IMG_DIR = "vietnam_img512_tune"


# The Kaggle input root, as a patchable constant so the Kaggle branch can be exercised
# by tests instead of only in production, which is where it has failed twice.
KAGGLE_INPUT = Path("/kaggle/input")
LOCAL_STAGING = Path("D:/tb-triage-kaggle-staging")

STAGED_LAYOUTS = ("staged", "staged_zip")

# Packaged files (configs, requirements.txt) are located by searching, not by assuming
# `src/`'s parent. Kaggle's uploader may FLATTEN a dataset, dropping the src/ and configs/
# folders and leaving every file in the dataset root, in which case `HERE.parent` is the
# wrong directory and nothing packaged resolves.
_ASSET_DIRS = [HERE, HERE / "configs", PKG, PKG / "configs",
               HERE.parent / "configs", HERE.parent]


def asset(name):
    """Path to a packaged config or requirements file, in whichever layout it arrived."""
    seen = []
    for d in _ASSET_DIRS:
        p = d / name
        if p not in seen:
            seen.append(p)
            if p.is_file():
                return p
    raise LayoutError(
        "packaged file %r not found. The code dataset may have been uploaded without its\n"
        "folder structure, or incompletely. Searched:\n  %s"
        % (name, "\n  ".join(str(p) for p in seen)))


def code_dir():
    """The directory the modules themselves live in, whatever the layout."""
    return HERE

# One open archive handle per zip, so encoding 1,352 images does not reopen it 1,352
# times. Read-only and single-threaded, which is how the encoder uses it.
_ZIP_CACHE = {}


class LayoutError(RuntimeError):
    pass


def _is_staged(d):
    return (all((d / f).is_file() for f in STAGED_FILES.values())
            and (d / STAGED_IMG_DIR).is_dir())


def _staged_candidates():
    """`TB_AUDIT_DATA` if set, else Kaggle inputs, then the local staging directory.

    The override is **exclusive**: when it is set, nothing else is searched. Otherwise a
    run told exactly where to look could silently fall back to a different cohort that
    happens to be lying around, which is the sort of thing that is only noticed after the
    numbers are written down.
    """
    env = os.environ.get("TB_AUDIT_DATA")
    if env:
        yield Path(env)
        return
    inp = KAGGLE_INPUT
    if inp.is_dir():
        # Recursive, because Kaggle nests dataset contents differently depending on how
        # they were uploaded. Anchored on the manifest file, so the candidate is the
        # directory that actually holds the staged material, at whatever depth.
        for m in sorted(inp.rglob(STAGED_FILES["manifest"])):
            yield m.parent
        for d in sorted(inp.iterdir()):
            if d.is_dir():
                yield d
    yield LOCAL_STAGING / "concept-audit-tune"


def _zip_candidates():
    """Any .zip that might hold the staged upload. Kaggle may or may not auto-extract."""
    env = os.environ.get("TB_AUDIT_DATA")
    roots = [Path(env)] if env else \
        (([KAGGLE_INPUT] if KAGGLE_INPUT.is_dir() else [])
         + [LOCAL_STAGING])
    for r in roots:
        if r.is_file() and r.suffix.lower() == ".zip":
            yield r
            continue
        if not r.is_dir():
            continue
        try:
            for z in sorted(r.rglob("*.zip")):      # any depth, for the same reason
                yield z
        except OSError:
            continue


def _zip_scratch(z):
    """A scratch directory keyed to this exact archive, not to its name.

    Keying on the stem alone let two different archives called `upload.zip` share
    extracted CSVs, so a second run could read the first one's cohort and pass every
    guard. The key is the full path, size and mtime.
    """
    st = z.stat()
    key = hashlib.sha256(("%s|%d|%d" % (z.resolve(), st.st_size, st.st_mtime_ns))
                         .encode("utf-8")).hexdigest()[:16]
    base = Path("/kaggle/working") if Path("/kaggle/working").is_dir() \
        else Path(tempfile.gettempdir())
    d = base / "_staged_from_zip" / ("%s-%s" % (z.stem, key))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _staged_from_zip(z):
    """Recognise a staged upload inside `z`; return the resolved paths or None.

    The four small files are extracted once (they are read line by line). The images are
    NOT extracted: members are indexed by patient id and read from the archive on demand,
    so no second copy of 172 MB of radiographs is made anywhere.
    """
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        # The zip may have been built from the folder or from inside it, so find the
        # prefix under which the four required files all appear.
        prefix = None
        for n in names:
            if n.endswith(STAGED_FILES["manifest"]):
                cand = n[:-len(STAGED_FILES["manifest"])]
                if all(any(m == cand + f for m in names) for f in STAGED_FILES.values()):
                    prefix = cand
                    break
        if prefix is None:
            return None
        imgs = {}
        for n in names:
            if n.lower().endswith(".png") and ("/%s/" % STAGED_IMG_DIR in "/" + n):
                imgs[Path(n).stem] = n
        if not imgs:
            return None

        scratch = _zip_scratch(z)
        out = {}
        for k, f in STAGED_FILES.items():
            dst = scratch / f
            if not dst.is_file() or dst.stat().st_size == 0:
                dst.write_bytes(zf.read(prefix + f))
            out[k] = dst
    out.update({"img_dir": z, "img_zip": z, "img_members": imgs,
                "zip_prefix": prefix, "scratch": scratch})
    return out


def _repo_root():
    for p in [PKG.parents[1]] + list(PKG.parents):
        if (p / "data/raw/vietnam").is_dir() and (p / "artifacts/p2/external").is_dir():
            return p
    return None


def resolve(prefer=None):
    """Return {layout, img_dir, scores, meta, split, manifest, out_dir, writable}."""
    if prefer in (None, "staged"):
        for d in _staged_candidates():
            try:
                if _is_staged(d):
                    return _finish("staged", d, {
                        "img_dir": d / STAGED_IMG_DIR,
                        **{k: d / v for k, v in STAGED_FILES.items()}})
            except OSError:
                continue
        for z in _zip_candidates():
            try:
                got = _staged_from_zip(z)
            except (OSError, zipfile.BadZipFile):
                continue
            if got:
                return _finish("staged_zip", z, got)
    if prefer in (None, "repo"):
        r = _repo_root()
        if r is not None:
            return _finish("repo", r, {
                "img_dir": r / "data/processed/v1/vietnam_img512",
                "scores": (r / "artifacts/overnight_scanner_run_20261021_0208"
                             / "tables/vietnam_case_scores.csv"),
                "meta": r / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv",
                "split": r / "artifacts/p2/external/vietnam_tune_report_split.json",
                "manifest": PKG / "tables/tune_manifest.json",
            })
    raise LayoutError(
        "No usable input layout found.\n"
        "  On Kaggle : attach the private dataset built by stage_kaggle_data.py, either as\n"
        "              a folder or as a single .zip. It must contain %s\n"
        "              and %s/<pid>.png.\n"
        "  Locally   : run from inside the tb-triage working copy, or set TB_AUDIT_DATA to\n"
        "              a staged directory or to the .zip itself.\n"
        "  Searched folders: %s\n"
        "  Searched zips   : %s"
        % (", ".join(sorted(STAGED_FILES.values())), STAGED_IMG_DIR,
           ", ".join(str(p) for p in list(_staged_candidates())[:6]) or "none",
           ", ".join(str(p) for p in list(_zip_candidates())[:6]) or "none"))


def _finish(layout, base, p):
    out = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PKG
    p.update({"layout": layout, "base": base, "out_dir": out,
              "writable": os.access(str(out), os.W_OK)})
    missing = [k for k in ("img_dir", "scores", "meta", "split") if not p[k].exists()]
    if missing:
        raise LayoutError("layout %r at %s is missing: %s"
                          % (layout, base, ", ".join("%s=%s" % (k, p[k]) for k in missing)))
    return p


def image_ids(p):
    """Patient ids for which an image is available, however the images are packaged."""
    if p["layout"] == "staged_zip":
        return set(p["img_members"])
    return {q.stem for q in p["img_dir"].glob("*.png")}


def open_image(p, pid):
    """A binary file object for one approved image. Caller closes it (or uses `with`).

    Reading from inside the archive keeps the zip upload a first-class layout rather than
    something to be unpacked first: no extraction step, no second copy of the images, and
    nothing written to the session's disk.
    """
    if p["layout"] == "staged_zip":
        member = p["img_members"].get(str(pid))
        if member is None:
            raise FileNotFoundError("%s.png is not in %s" % (pid, p["img_zip"].name))
        key = str(p["img_zip"])
        zf = _ZIP_CACHE.get(key)
        if zf is None:
            zf = _ZIP_CACHE[key] = zipfile.ZipFile(p["img_zip"])
        return io.BytesIO(zf.read(member))
    return io.open(p["img_dir"] / ("%s.png" % pid), "rb")


def describe(p):
    lines = ["layout            : %s" % p["layout"], "base              : %s" % p["base"]]
    for k in ("img_dir", "scores", "meta", "split", "manifest", "out_dir"):
        mark = "ok " if p[k].exists() else "ABSENT"
        extra = ""
        if k == "img_dir":
            if p["layout"] == "staged_zip":
                extra = "  (zip, %d png members, prefix %r)" % (len(p["img_members"]),
                                                                p["zip_prefix"])
            elif p[k].is_dir():
                extra = "  (%d png)" % sum(1 for _ in p[k].glob("*.png"))
        lines.append("%-18s: [%s] %s%s" % (k, mark, p[k], extra))
    if p["layout"] == "staged_zip":
        lines.append("small files from  : %s  (images are NOT extracted)" % p["scratch"])
    lines.append("output writable   : %s" % p["writable"])
    return "\n".join(lines)


def load_manifest(p):
    """The expected TUNE id set and its hash, or None when no manifest is present."""
    if not p["manifest"].exists():
        return None
    return json.load(io.open(p["manifest"], encoding="utf-8"))
