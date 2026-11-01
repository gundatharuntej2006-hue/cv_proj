# -*- coding: utf-8 -*-
"""Preflight: refuse to start an expensive run that cannot finish correctly.

Checks the runtime, the dependencies, the pinned model revision, the approved cohort and
the partition boundary, then prints a verdict. **It never relaxes a guard to make the run
possible** — a failed check is reported as a blocker with the action that would clear it.

    py -3 src/preflight.py                 # checks that need no network
    py -3 src/preflight.py --check-weights # also resolves the pinned CLEAR revision
"""
import argparse
import importlib
import io
import json
import os
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

import paths as P                                                   # noqa: E402
import concept_audit as A                                           # noqa: E402

# Minimums are measured requirements, not guesses: the CLEAR checkpoint is 603 MB and the
# concept-embedding tensor is 1,131 MB, and torch itself needs room to unpack them.
MIN_RAM_GB = 3.0
MIN_DISK_GB = 6.0
REQ = ["numpy", "pandas", "scipy", "sklearn", "torch", "torchvision", "PIL",
       "huggingface_hub", "psutil"]
FORBIDDEN = ("internal_test", "vietnam_img512_report", "report_split_only")


def _v(mod):
    try:
        return getattr(importlib.import_module(mod), "__version__", "present")
    except Exception as e:
        return "ABSENT (%s)" % type(e).__name__


def run(check_weights=False):
    res, blockers, warnings = {}, [], []

    # ---- runtime
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_avail, ram_total, ram_pct = vm.available / 1e9, vm.total / 1e9, vm.percent
    except Exception:
        ram_avail = ram_total = ram_pct = None
    # The CUDA probe is skippable. Initialising a CUDA context inside a process that has
    # already used torch on CPU, on a machine this short of RAM, has crashed the
    # interpreter outright (a Windows access violation inside mem_get_info, which no
    # try/except can catch). Tests set TB_AUDIT_NO_GPU_PROBE=1; a real run never does.
    gpu, vram = None, None
    if os.environ.get("TB_AUDIT_NO_GPU_PROBE") == "1":
        gpu = "probe skipped (TB_AUDIT_NO_GPU_PROBE=1)"
    else:
        try:
            import torch
            if torch.cuda.device_count() > 0:
                gpu = torch.cuda.get_device_name(0)
                vram = torch.cuda.mem_get_info()[0] / 1e9
        except Exception:
            pass
    disk = shutil.disk_usage(str(PKG)).free / 1e9
    res["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                      "ram_available_gb": ram_avail and round(ram_avail, 2),
                      "ram_total_gb": ram_total and round(ram_total, 2),
                      "ram_percent_used": ram_pct,
                      "gpu": gpu, "vram_free_gb": vram and round(vram, 2),
                      "disk_free_gb": round(disk, 1)}
    if ram_avail is not None and ram_avail < MIN_RAM_GB:
        blockers.append("host RAM available is %.2f GB, below the %.1f GB needed to hold "
                        "the CLEAR checkpoint and a slice of the concept tensor. Run on "
                        "Kaggle, or free memory. Do NOT reduce the image resolution or "
                        "swap the architecture to fit."
                        % (ram_avail, MIN_RAM_GB))
    if disk < MIN_DISK_GB:
        blockers.append("free disk is %.1f GB, below %.1f GB; the pinned assets are "
                        "~1.73 GB before unpacking" % (disk, MIN_DISK_GB))
    if gpu is None:
        warnings.append("no CUDA device visible. The run works on CPU but will be slow; "
                        "on Kaggle, switch the accelerator on before starting.")

    # ---- dependencies
    res["dependencies"] = {m: _v(m) for m in REQ}
    pinned = {}
    for line in io.open(P.asset("requirements.txt"), encoding="utf-8"):
        line = line.split("#")[0].strip()
        if "==" in line:
            n, v = line.split("==", 1)
            pinned[n.strip()] = v.strip()
    res["pinned_requirements"] = pinned
    alias = {"scikit-learn": "sklearn", "pillow": "PIL"}
    for name, want in pinned.items():
        mod = alias.get(name, name)
        have = res["dependencies"].get(mod) or _v(mod)
        if str(have).startswith("ABSENT"):
            blockers.append("dependency %s is missing (pinned %s)" % (name, want))
        elif not str(have).startswith(want):
            warnings.append("%s is %s, pinned %s. Results may differ slightly; this is "
                            "recorded, not silently accepted." % (name, have, want))

    # ---- data layout, cohort and partition boundary
    try:
        p = P.resolve()
        res["layout"] = {k: (len(v) if k == "img_members" else str(v))
                         for k, v in p.items()}
        d = A.load_tune_only(p)
        missing, extra = A.check_images(p, d["pid"])
        _, primary = A.concept_map()
        res["cohort"] = {
            "n_cases": len(d["pid"]), "n_tb_positive": int(d["y"].sum()),
            "n_previous_tb": int(d["past_tb"].sum()),
            "cohort_fingerprint_sha256": d["fingerprint"],
            "fingerprint_matches_pinned": True,
            "images_missing": len(missing), "images_present_not_approved": len(extra),
            "n_concepts_primary": len(primary),
            "expected_similarity_shape": [len(d["pid"]), len(primary)],
        }
        if missing:
            blockers.append("%d approved TUNE images are missing from the input, first: %s"
                            % (len(missing), missing[:5]))
        if extra:
            warnings.append("%d images are present that are not on the allow-list; they "
                            "will never be read, but a TUNE-only upload would not contain "
                            "them. First: %s" % (len(extra), extra[:5]))
    except Exception as e:
        blockers.append("%s: %s" % (type(e).__name__, e))
        res["cohort"] = None

    # ---- partition boundary: nothing forbidden may be visible to the run
    seen = []
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").is_dir() else []
    # Only an *attached* staged input is scanned. The repository working copy legitimately
    # contains TBX11K, so scanning it would report the locked partition as a blocker for
    # merely existing on disk; the allow-list, not the filesystem, is what keeps it out
    # of the run there.
    if res.get("layout", {}).get("layout") in P.STAGED_LAYOUTS:
        roots.append(Path(res["layout"]["base"]))
    res["partition_scan_scope"] = [str(r) for r in roots] or \
        ["none (repo layout: enforcement is the allow-list, not a filesystem scan)"]
    for r in roots:
        try:
            for q in list(r.rglob("*"))[:200000]:
                low = q.name.lower()
                if any(f in low for f in FORBIDDEN):
                    seen.append(str(q))
        except OSError:
            pass
    # A zipped upload hides its contents from a filesystem walk, so its member names are
    # scanned too. Otherwise the scan would silently pass on exactly the layout in use.
    if res.get("layout", {}).get("layout") == "staged_zip":
        import zipfile
        try:
            with zipfile.ZipFile(res["layout"]["img_zip"]) as zf:
                for n in zf.namelist():
                    if any(f in n.lower() for f in FORBIDDEN):
                        seen.append("%s!%s" % (res["layout"]["img_zip"], n))
        except (OSError, zipfile.BadZipFile) as e:
            warnings.append("could not scan the zip members (%s)" % e)
    res["forbidden_paths_visible"] = seen[:20]
    if seen:
        blockers.append("a locked partition appears among the attached inputs: %s. Remove "
                        "it from the dataset; the guard will not be relaxed." % seen[:3])
    # Every module that touches data is scanned for a reference to the locked partition.
    # This file is excluded only because it must name the partition in order to exclude
    # it; the exclusion is by filename, so a new module cannot opt itself out.
    for f in sorted(P.code_dir().glob("*.py")):
        if f.name == Path(__file__).name:
            continue
        src = f.read_text(encoding="utf-8")
        for tok in ('"internal_test"', "'internal_test'", "internal_test/"):
            if tok in src:
                blockers.append("%s references %s" % (f.name, tok))

    # ---- model revision and asset identity
    res["model"] = {"repo": A.HF_REPO, "pinned_revision": A.HF_REV,
                    "input_px": A.CLEAR_INPUT_PX,
                    "expected_concept_rows": A.CLEAR_EMBED_ROWS,
                    "revision_is_a_commit": len(A.HF_REV) == 40,
                    "resolved": None}
    pinned_assets = json.load(io.open(P.asset("model_assets.json"), encoding="utf-8"))
    res["model"]["pinned_assets"] = pinned_assets["files"]
    if check_weights:
        try:
            from huggingface_hub import HfApi
            info = HfApi().repo_info(A.HF_REPO, revision=A.HF_REV, files_metadata=True)
            got = {s.rfilename: {"bytes": s.size,
                                 "sha256": s.lfs.get("sha256") if s.lfs else None}
                   for s in info.siblings if s.rfilename in pinned_assets["files"]}
            res["model"]["resolved"] = {"sha": info.sha, "files": got}
            if info.sha != A.HF_REV:
                blockers.append("the resolved repo sha %s is not the pinned %s"
                                % (info.sha, A.HF_REV))
            for name, want in pinned_assets["files"].items():
                have = got.get(name)
                if not have:
                    blockers.append("pinned asset %s is absent at revision %s"
                                    % (name, A.HF_REV[:12]))
                elif have["sha256"] and have["sha256"] != want["sha256"]:
                    blockers.append(
                        "%s has sha256 %s at the pinned revision, but this package was "
                        "prepared against %s. The asset changed; do not proceed."
                        % (name, have["sha256"][:16], want["sha256"][:16]))
                elif have["bytes"] != want["bytes"]:
                    warnings.append("%s is %d bytes, expected %d"
                                    % (name, have["bytes"], want["bytes"]))
        except Exception as e:
            warnings.append("could not reach Hugging Face to confirm the revision (%s: "
                            "%s). The run itself needs the network, so confirm Internet "
                            "is enabled before starting."
                            % (type(e).__name__, str(e)[:120]))
    res["network_required"] = {
        "required": True,
        "why": ["the pinned CLEAR assets are fetched from huggingface.co at run time",
                "torch.hub fetches the DINOv2 ViT-B/14-reg architecture from github.com"],
        "offline_possible": "only if both are pre-attached as Kaggle datasets and the "
                            "loaders are pointed at them; that is not how this package "
                            "is written, so a fully offline run is NOT currently possible"}

    res["blockers"] = blockers
    res["warnings"] = warnings
    res["verdict"] = "GO" if not blockers else "STOP"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-weights", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = run(a.check_weights)

    def sec(t):
        print("\n" + t + "\n" + "-" * len(t))

    sec("runtime")
    for k, v in r["runtime"].items():
        print("  %-22s %s" % (k, v))
    sec("dependencies  (pinned -> installed)")
    alias = {"scikit-learn": "sklearn", "pillow": "PIL"}
    for k, v in r["pinned_requirements"].items():
        print("  %-18s %-10s -> %s" % (k, v, r["dependencies"].get(alias.get(k, k), "?")))
    sec("cohort and partition boundary")
    if r["cohort"]:
        for k, v in r["cohort"].items():
            print("  %-34s %s" % (k, v))
    print("  %-34s %s" % ("forbidden paths visible",
                          r["forbidden_paths_visible"] or "none"))
    sec("model")
    for k, v in r["model"].items():
        print("  %-22s %s" % (k, v if k != "resolved" else json.dumps(v)[:300]))
    sec("network")
    print("  required: %s" % r["network_required"]["required"])
    for w in r["network_required"]["why"]:
        print("    - %s" % w)
    if r["warnings"]:
        sec("warnings")
        for w in r["warnings"]:
            print("  ! %s" % w)
    if r["blockers"]:
        sec("BLOCKERS")
        for b in r["blockers"]:
            print("  X %s" % b)
    print("\nVERDICT: %s" % r["verdict"])
    if a.json:
        json.dump(r, io.open(a.json, "w", encoding="utf-8"), indent=1, default=str)
        print("wrote %s" % a.json)
    raise SystemExit(0 if r["verdict"] == "GO" else 2)


if __name__ == "__main__":
    main()
