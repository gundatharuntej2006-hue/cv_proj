# -*- coding: utf-8 -*-
"""Priority 0: hash every input this run touches, and record the environment."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]

INPUTS = [
    "artifacts/p2/labels_tbx11k_trainval_v1.csv",
    "artifacts/p2/dedup/provisional_official_splits.csv",
    "artifacts/p2/dryrun/provisional_c01.csv",
    "artifacts/p2/external/vietnam_tune_report_split.json",
    "artifacts/p2/external/vietnam_eval.json",
    "artifacts/p2/external/gate_operational_audit.json",
    "artifacts/p2/audit/vietnam_case_level_gate_audit.csv",
    "artifacts/p2/audit/cross_dataset_case_predictions.csv",
    "artifacts/p2/audit/cross_dataset_metrics.csv",
    "artifacts/p2/audit/cross_dataset_manifest.json",
    "artifacts/p2/regional/mil_sweep.json",
    "artifacts/p2/regional/mil_final.json",
    "artifacts/p2/regional/attention_mil.json",
    "artifacts/p2/dryrun/hard_negatives.json",
    "artifacts/p2/dryrun/learning_curve.json",
    "artifacts/p2/bundle/head.npz",
    "artifacts/p2/bundle/bank.npz",
    "artifacts/p2/bundle/bundle.json",
    "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv",
    "src/p2_cls/pipeline.py",
    "src/p2_gate/api.py",
    "scripts/p2_33_mil_final.py",
    "scripts/p2_36_cross_dataset_pooling.py",
]
BIG = [
    "artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz",
    "artifacts/p2/regional/regional_vietnam_lungs_g12_v1.npz",
    "artifacts/p2/regional/regional_nlm_lungs_g12_v1.npz",
]


def sha(p, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest(), n


def git(*a):
    return subprocess.run(["git"] + list(a), cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    files = {}
    for rel in INPUTS:
        p = ROOT / rel
        if not p.exists():
            files[rel] = {"status": "MISSING"}
            continue
        d, n = sha(p)
        files[rel] = {"sha256": d, "bytes": n, "full_file": True}
    for rel in BIG:
        p = ROOT / rel
        if not p.exists():
            files[rel] = {"status": "MISSING"}
            continue
        d, n = sha(p)
        files[rel] = {"sha256": d, "bytes": n, "full_file": True}

    try:
        import numpy, sklearn, scipy, pandas, torch
        vers = {"numpy": numpy.__version__, "sklearn": sklearn.__version__,
                "scipy": scipy.__version__, "pandas": pandas.__version__,
                "torch": torch.__version__}
        cuda = {"available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as e:                                          # pragma: no cover
        vers, cuda = {"error": str(e)}, {}

    import psutil
    v = psutil.virtual_memory()

    man = {
        "run_id": RUN.name,
        "start_utc": "2026-10-21T19:50:00Z",
        "git": {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "status_short": git("status", "--short").splitlines(),
                "working_tree_changes_outside_run_dir": [
                    l for l in git("status", "--short").splitlines() if RUN.name not in l]},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "packages": vers, "cuda": cuda,
                        "ram_total_gb": round(v.total / 1e9, 2),
                        "ram_available_gb_at_start": round(v.available / 1e9, 2)},
        "resource_caps": {"gpu_memory_fraction_max": 0.8,
                          "note": "RAM availability was under 1 GB for the whole run; "
                                  "no analysis that would load a multi-GB feature cache was attempted."},
        "split_roles": {
            "tbx11k_train": "PERMITTED for fitting",
            "tbx11k_development": "PERMITTED for early stopping and selection; already used",
            "tbx11k_internal_test": "RESERVED, untouched",
            "vietnam_TUNE": "development half; used for the pooling selection",
            "vietnam_REPORT": "RESERVED evaluation, read once",
            "shenzhen": "EXPOSED (informed preprocessing selection) - exploratory only, never for fitting",
            "montgomery": "EXPOSED (informed preprocessing selection) - exploratory only, never for fitting",
            "dadb": "external, scored under the global representation only",
        },
        "feature_spec": {"backbone": "microsoft/rad-dino ViT-B/14, frozen",
                         "revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
                         "input_px": 518, "dim": 768, "regional_grid": "12x12 = 144 regions",
                         "dtype": "float16"},
        "seeds": [17, 29, 43, 61, 79],
        "files": files,
    }
    (RUN / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=2))
    print("hashed %d files" % len(files))


if __name__ == "__main__":
    main()
