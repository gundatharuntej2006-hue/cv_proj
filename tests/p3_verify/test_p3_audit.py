"""P3 audit: one test per check ID (A1 ... K3). Read-only except throwaway temp dirs.

Never touches sealed data, never requests a gate token. Statuses: PASS / FAIL / MANUAL / NOT_YET.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcore import REPO, finish, git, load_json, own_repo, run_pytest, summary_line  # noqa: E402

PY = sys.executable
DECISIONS = {"D16": "decisions/p3_det_input_size.json", "D17": "decisions/p3_det_checkpoint.json",
             "D18": "decisions/p3_det_conf_threshold.json", "D19": "decisions/p3_gradcam_config.json",
             "D20": "decisions/p3_licences.json"}
DFINE_PIN = "956d1709314c2c6a4df6f34de232054578a7449f"
NO_REPO = ("cv_proj is not its own git repository (git toplevel = the home folder, no commits), so "
           "git-based evidence cannot exist yet")
SUITE = "pytest tests -m 'not gpu' (venv: ~/.venvs/p3)"
TBD_RESULTS = "no real training runs / P1 data / P2 classifier / P4 pipeline yet"


def files_mine() -> list[str]:
    """Every file under cv_proj except the three pre-existing ones, caches and the D-FINE checkout."""
    out = []
    for p in REPO.rglob("*"):
        rel = p.relative_to(REPO).as_posix()
        if p.is_file() and "__pycache__" not in rel and not rel.startswith("third_party/D-FINE/") \
                and rel not in ("README.md", "docs/methodology.md", "docs/proposal.md") \
                and not rel.startswith("artifacts/p3/verify/"):
            out.append(rel)
    out.append("third_party/D-FINE")
    return sorted(out)


# ===================================================================================== A
def test_A1():
    sys.path.insert(0, str(REPO / ".github" / "scripts"))
    import ownership_check as oc
    changes = [oc.Change("A", f) for f in files_mine()]
    errors = oc.check_ownership("p3/audit", changes) + oc.check_data_guard(changes)
    sim = f"simulated ownership check of all {len(changes)} files I created as branch p3/audit: " \
          f"{len(errors)} violation(s)" + (": " + "; ".join(errors[:5]) if errors else "")
    if errors:
        finish("A1", "Only my paths touched (ownership check)", "FAIL", sim,
               fix="move or remove the listed files")
    if not own_repo():
        finish("A1", "Only my paths touched (ownership check)", "NOT_YET", f"{NO_REPO}. Content-level: {sim}.",
               owner="P1 (repo bootstrap P1-01) / you", command="git fetch origin && python .github/scripts/"
               "ownership_check.py --base origin/main --head HEAD --branch $(git branch --show-current)")
    rc, out = git("rev-parse", "--abbrev-ref", "HEAD")
    p = __import__("subprocess").run([PY, ".github/scripts/ownership_check.py", "--base", "origin/main", "--head",
                                      "HEAD", "--branch", out], cwd=REPO, capture_output=True, text=True)
    finish("A1", "Only my paths touched (ownership check)", "PASS" if p.returncode == 0 else "FAIL",
           p.stdout + p.stderr, fix="revert edits outside P3 paths")


def test_A2():
    import yaml
    ci = REPO / ".github" / "workflows" / "ci.yml"
    doc = yaml.safe_load(ci.read_text())
    text = ci.read_text()
    steps = [s for j in doc["jobs"].values() for s in j["steps"]]
    pyver = {s.get("with", {}).get("python-version") for s in steps if "setup-python" in str(s.get("uses"))}
    validators = [s for s in steps if "tbcore" in str(s.get("name", "")).lower() and "run" in s]
    problems = []
    if pyver != {"3.11"}:
        problems.append(f"python versions {pyver}")
    if "pytest tests" not in text:
        problems.append("no pytest step")
    if "ownership_check.py --base" not in text or "--all-tracked" not in text:
        problems.append("ownership check / data guard missing")
    if not validators or any(str(s.get("if")).lower() == "false" for s in validators):
        problems.append("tbcore validator step is a disabled placeholder (if: false), so validators never run")
    if "requirements/p3.txt" not in text:
        problems.append("CI installs only requirements/base.txt: torch/pyarrow/D-FINE deps/grad-cam/reportlab are "
                        "missing, so most P3 tests are importorskip-ed (silently skipped) in CI")
    ev = f"{ci.relative_to(REPO)}: python {sorted(pyver)}; " + ("; ".join(problems) if problems else "all present")
    finish("A2", "CI: Python 3.11, pytest, tbcore validators, ownership check, data guard",
           "FAIL" if problems else "PASS", ev,
           fix="(mine) install requirements/p3.txt with the CPU torch index in the test job, or add a p3 job, so P3 "
               "tests actually run; (P4) enable the validator step once tbcore exposes its entry point",
           owner="P3 (deps), P4 (validator entry point)")


def test_A3():
    rc, tail = run_pytest("tests", "-m", "not gpu", "--ignore=tests/p3_verify", timeout=5400)
    rc2, gpu = run_pytest("tests", "-m", "gpu", "--co", "--ignore=tests/p3_verify")
    gpu_ids = [l for l in gpu.splitlines() if "::" in l] or re.findall(r"\S+::\S+", gpu)
    ev = f"{summary_line(tail)} (CPU, venv {sys.version.split()[0]}). gpu-marked (MANUAL, see F6): " \
         f"{', '.join(gpu_ids) or summary_line(gpu)}"
    finish("A3", "Full pytest on CPU", "PASS" if rc == 0 else "FAIL", ev, fix="see failing tests above")


def test_A4():
    imgs = [f for f in files_mine() if f.startswith("data/") and re.search(r"\.(png|jpe?g|dcm)$", f, re.I)]
    weights = [f for f in files_mine() if re.search(r"\.(pt|pth|ckpt|safetensors|onnx)$", f, re.I)]
    sizes = sorted(((REPO / f).stat().st_size, f) for f in files_mine() if (REPO / f).is_file())[-3:]
    big = [f for s, f in sizes if s > 10_000_000]
    ev = f"images under data/: {len(imgs)}; model weight files in my paths: {len(weights)}; largest files: " + \
         ", ".join(f"{f} ({s / 1e6:.1f} MB)" for s, f in reversed(sizes)) + \
         ". Note: nothing is 'staged' because there is no repository yet (A1)."
    bad = imgs or weights or big
    finish("A4", "No images under data/, no checkpoints/large files outside LFS", "FAIL" if bad else "PASS", ev,
           fix="remove the listed files from the commit; checkpoints stay on Drive with SHA in D17")


def test_A5():
    rc, head = git("rev-parse", "HEAD", cwd=REPO / "third_party" / "D-FINE")
    _, dirty = git("status", "--porcelain", cwd=REPO / "third_party" / "D-FINE")
    pinned = head == DFINE_PIN
    ev = f"third_party/D-FINE HEAD {head[:12]} (pin {DFINE_PIN[:12]}: {'match' if pinned else 'MISMATCH'}); " \
         f"working tree {'clean' if not dirty else 'DIRTY: ' + dirty[:200]}; .gitmodules: " \
         f"{'present' if (REPO / '.gitmodules').exists() else 'absent'}"
    if not pinned or dirty:
        finish("A5", "D-FINE pinned submodule, clean", "FAIL", ev, fix=f"git -C third_party/D-FINE checkout {DFINE_PIN}")
    if not (REPO / ".gitmodules").exists():
        finish("A5", "D-FINE pinned submodule, clean", "NOT_YET",
               ev + ". It is a plain clone at the pin, not yet a registered submodule (needs the repo; .gitmodules "
                    "is at the repo root).", owner="P1 (repo) then P3",
               command=f"git submodule add https://github.com/Peterande/D-FINE third_party/D-FINE && "
                       f"git -C third_party/D-FINE checkout {DFINE_PIN} && git add .gitmodules third_party/D-FINE")
    finish("A5", "D-FINE pinned submodule, clean", "PASS", ev)


# ===================================================================================== B
def test_B1():
    missing = [f"{k} {v}" for k, v in DECISIONS.items() if not (REPO / v).exists()]
    finish("B1", "Decision files D16-D20 exist", "FAIL" if missing else "PASS",
           "missing: " + ", ".join(missing) if missing else "all five present: " + ", ".join(DECISIONS.values()),
           fix="python -m p3_det.select_size criterion / lock_checkpoint criterion / lock_threshold criterion / "
               "p3_xai.d19 criterion")


def test_B2():
    sys.path.insert(0, str(REPO / "tests" / "p3_decisions"))
    from c16_standin import CONTRACT_SCHEMA, validate_decision
    rows, bad_paths = [], []
    for k, rel in DECISIONS.items():
        d = load_json(rel)
        validate_decision(d)
        for e in d.get("evidence", []):
            if not e.startswith("http") and not (REPO / e).exists():
                bad_paths.append(f"{k}: {e}")
        rows.append(f"{k} locked={d['locked']} value={'yes' if d['value'] else 'empty'}")
    schema = "contracts/decision.schema.json" if CONTRACT_SCHEMA.exists() else "C16 STAND-IN (contracts/ missing)"
    ev = f"validated against {schema}; " + "; ".join(rows) + (f"; missing evidence paths: {bad_paths}" if bad_paths else "")
    if bad_paths:
        finish("B2", "Decisions valid, locked, evidence exists", "FAIL", ev, fix="restore or correct evidence paths")
    finish("B2", "Decisions valid, locked, evidence exists", "NOT_YET",
           ev + f". None can be locked: D16-D19 have no results ({TBD_RESULTS}); D20 waits for the P3-01 T4 numbers "
                "and your confirmation; the real schema is P4's.",
           owner="P4 (schema), P1/P2 (data, classifier), you (Kaggle runs, D20 confirmation)")


def test_B3():
    allowed = {"none", "train", "development", "calibration"}
    rows = {k: load_json(v)["data_used"] for k, v in DECISIONS.items()}
    bad = {k: v for k, v in rows.items() if not set(v) <= allowed or not v}
    finish("B3", "data_used has no sealed split", "FAIL" if bad else "PASS", json.dumps(rows),
           fix="data_used must be a subset of {none, train, development, calibration}")


def test_B4():
    from p3_det.decisions import verify_order
    reps = {k: verify_order(REPO / DECISIONS[k]) for k in ("D16", "D17", "D18", "D19")}
    if any(ok for ok, _ in reps.values()) and not all(ok for ok, _ in reps.values()):
        finish("B4", "Criterion committed before result (git log)", "FAIL",
               "\n".join(f"{k}: {r}" for k, (ok, r) in reps.items()), fix="see per-decision verdicts")
    if not own_repo():
        finish("B4", "Criterion committed before result (git log)", "NOT_YET",
               f"{NO_REPO}; verify_order: " + "; ".join(f"{k}: {r.splitlines()[-1]}" for k, (ok, r) in reps.items()),
               owner="you (commit the four criteria first, then results)",
               command="python -m p3_det.decisions verify decisions/p3_det_input_size.json decisions/p3_det_checkpoint.json "
                       "decisions/p3_det_conf_threshold.json decisions/p3_gradcam_config.json")
    finish("B4", "Criterion committed before result (git log)",
           "PASS" if all(ok for ok, _ in reps.values()) else "FAIL",
           "\n".join(f"{k}: {r}" for k, (ok, r) in reps.items()), fix="re-register a criterion via P4 (CCR)")


def test_B5():
    d = load_json(DECISIONS["D18"])
    if d["guide_ack"] and not d["value"].get("guide_ack_record"):
        finish("B5", "D18 guide_ack only after guide approval", "FAIL", "guide_ack=true without guide_ack_record",
               fix="set guide_ack via lock_threshold ack-guide only")
    if not d["value"]:
        finish("B5", "D18 guide_ack only after guide approval", "NOT_YET",
               "D18 has no threshold yet (needs D17); guide_ack=false", owner="you + guide",
               command="after the guide approves: python -m p3_det.lock_threshold ack-guide --date YYYY-MM-DD --note '...'")
    finish("B5", "D18 guide_ack only after guide approval", "MANUAL",
           f"guide_ack={d['guide_ack']}, record={d['value'].get('guide_ack_record')}", owner="you",
           command="confirm in chat that the guide approved this threshold")


def test_B6():
    d = load_json(DECISIONS["D17"])
    if not d["value"]:
        finish("B6", "D17 checkpoint SHA matches the file", "NOT_YET", "D17 has no checkpoint yet (no training run)",
               owner="you (Kaggle training)", command="sha256sum artifacts/p3/det/checkpoints/<run>/ckpt_best.pt")
    ck = REPO / "artifacts/p3/det/checkpoints" / d["value"]["run_dir"] / "ckpt_best.pt"
    if not ck.exists():
        finish("B6", "D17 checkpoint SHA matches the file", "MANUAL", f"{ck} not downloaded", owner="you",
               command=f"download ckpt_best.pt from Drive to {ck.relative_to(REPO)}")
    h = hashlib.sha256(ck.read_bytes()).hexdigest()
    finish("B6", "D17 checkpoint SHA matches the file", "PASS" if h == d["value"]["checkpoint"]["sha256"] else "FAIL",
           f"recomputed {h} vs D17 {d['value']['checkpoint']['sha256']}", fix="wrong checkpoint file")


def test_B7():
    from importlib import metadata
    d = load_json(DECISIONS["D20"])["value"]
    lic = REPO / "third_party/D-FINE/LICENSE"
    h = hashlib.sha256(lic.read_bytes()).hexdigest()
    ok_dfine = h == d["dfine"]["licence_sha256"] and "Apache License" in lic.read_text() and "Version 2.0" in lic.read_text()
    dist = metadata.distribution("grad-cam")
    gl = next((f for f in dist.files or [] if f.name.upper().startswith("LICENSE")), None)
    gtext = (Path(dist.locate_file(gl)).read_text() if gl else "")
    norm = lambda s: " ".join(s.split())  # noqa: E731
    ok_gc = "MIT License" in gtext and "Jacob Gildenblat" in gtext and dist.version == d["pytorch_grad_cam"]["pypi"].split("==")[1]
    ev = f"D-FINE LICENSE sha256 {h[:16]}... {'==' if ok_dfine else '!='} D20 (Apache-2.0); installed grad-cam " \
         f"{dist.version}, LICENSE {'MIT, Jacob Gildenblat' if ok_gc else 'MISMATCH'} (text-identical to D20's: " \
         f"{norm(gtext)[:60] == 'MIT License Copyright (c) 2021 Jacob Gildenblat Permission is'}); RT-DETRv2 not " \
         "installed (extension, not started) - D20 records commit 29320b6"
    finish("B7", "D20 licences match actual LICENSE files", "PASS" if ok_dfine and ok_gc else "FAIL", ev,
           fix="re-read the licence file and update D20")


# ===================================================================================== C
def test_C1():
    from p3_det import c10
    real = sorted(glob.glob(str(REPO / "artifacts/p3/det/detections_*.json")))
    mocks = sorted(glob.glob(str(REPO / "mock/artifacts/p3/det/detections_*.json")))
    for m in mocks:
        c10.validate_doc(json.loads(Path(m).read_text()))
    if not real:
        finish("C1", "C10 files valid", "NOT_YET",
               f"no real C10 (needs locked D16-D18 + checkpoint). Mock C10 {[Path(m).name for m in mocks]} validate "
               "with the producer validator", owner="you (P3-08 after D16-D18)", command="make p3-export-c10")
    d16 = load_json(DECISIONS["D16"])["value"].get("input_size")
    d18 = load_json(DECISIONS["D18"])["value"].get("conf_threshold")
    errs = []
    for p in real:
        try:
            c10.validate_c10(Path(p))
            doc = json.loads(Path(p).read_text())
            if doc["input_size"] != d16 or doc["conf_threshold"] != d18:
                errs.append(f"{Path(p).name}: input_size/threshold != D16/D18")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{Path(p).name}: {e}")
    finish("C1", "C10 files valid", "FAIL" if errs else "PASS", "; ".join(errs) or f"{len(real)} files valid",
           fix="re-export with make p3-export-c10")


def test_C2():
    real = [p for p in glob.glob(str(REPO / "artifacts/p3/xai/*/*/*.npy"))]
    if not real:
        finish("C2", "C11 heatmaps float16 [512,512] in [0,1], zero outside crop", "NOT_YET",
               "no real C11 heatmaps (needs P2's classifier and P1's data)", owner="P2, P1, then you",
               command="python -m p3_xai.export_c11 --crop-model <id> --full-model <id>")
    finish("C2", "C11 heatmaps", "MANUAL", f"{len(real)} heatmaps present; run the dedicated check",
           command="pytest tests/p3_xai -k c11")


def test_C3():
    import tempfile

    from p3_xai import c11
    real = sorted(glob.glob(str(REPO / "artifacts/p3/xai/loc_*.parquet")))
    mock_err = []
    sys.path.insert(0, str(REPO / "mock"))
    import make_c11
    with tempfile.TemporaryDirectory() as tmp:                     # regenerate: catches generator drift
        try:
            c11.validate_table(make_c11.generate(n=24, seed=0, root=tmp))
        except Exception as e:  # noqa: BLE001
            mock_err.append(f"freshly generated mock: {e}")
    for m in sorted(glob.glob(str(REPO / "mock/artifacts/p3/xai/loc_*.parquet"))):   # on disk: catches staleness
        try:
            c11.validate_table(Path(m))
        except Exception as e:  # noqa: BLE001
            mock_err.append(f"{Path(m).relative_to(REPO)}: {e}")
    if mock_err:
        finish("C3", "loc_*.parquet columns/dtypes/ranges", "FAIL",
               "STALE MOCK on disk: " + "; ".join(mock_err) + ". It was generated in P3-03 (rows for non-TB images, "
               "tb_subtype null) before P3-10 changed C11 to TB-with-box rows only; consumers using TB_MOCK=1 get the "
               "old shape", fix="make p3-mocks (regenerates mock/ with the current generators)", owner="P3")
    if not real:
        finish("C3", "loc_*.parquet columns/dtypes/ranges", "NOT_YET", "no real C11 tables yet",
               owner="P2, P1, then you", command="python -m p3_xai.export_c11 ...")
    errs = []
    for p in real:
        try:
            c11.validate_table(Path(p))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{Path(p).name}: {e}")
    finish("C3", "loc_*.parquet columns/dtypes/ranges", "FAIL" if errs else "PASS", "; ".join(errs) or "valid")


def test_C4():
    sys.path.insert(0, str(REPO / "tests" / "p3_mock"))
    import standin_contracts as sc
    from p3_ops.logger import OPERATING_POINTS, VERSION_FIELDS
    real = REPO / "logs" / "predictions.jsonl"
    mock = REPO / "mock" / "logs" / "predictions.jsonl"
    mock_lines = sc.validate_c15_file(mock) if mock.exists() else []
    ok_mock = all(all(l[k] for k in VERSION_FIELDS) and l["operating_point"] in OPERATING_POINTS for l in mock_lines)
    if not real.exists():
        finish("C4", "predictions.jsonl valid C15", "NOT_YET" if ok_mock else "FAIL",
               f"no real log (written by P4's run_case via log_case). Mock log: {len(mock_lines)} lines valid C15, "
               f"version fields present, operating points OK: {ok_mock}", owner="P4 (run_case/run_batch)")
    lines = sc.validate_c15_file(real)
    keys = [(l["ts"], l["case_id"], l["operating_point"]) for l in lines]
    ts_sorted = [l["ts"] for l in lines] == sorted(l["ts"] for l in lines)
    ok = len(set(keys)) == len(keys) and ts_sorted
    finish("C4", "predictions.jsonl valid C15", "PASS" if ok else "FAIL",
           f"{len(lines)} lines; duplicates {len(keys) - len(set(keys))}; ts non-decreasing {ts_sorted}")


def test_C5():
    refs = []
    for f in glob.glob(str(REPO / "decisions/*.json")) + glob.glob(str(REPO / "report/*.md")):
        if "mock/" in Path(f).read_text(encoding="utf-8"):
            refs.append(Path(f).name)
    fake_in_artifacts = [p for p in ("artifacts/p3/det/viz_check", "artifacts/p3/ops/pdf_preview",
                                     "artifacts/p3/xai/overlay_preview") if (REPO / p).exists()]
    pdfs_unmarked = [Path(p).name for p in glob.glob(str(REPO / "artifacts/p3/ops/pdf_preview/*.pdf"))
                     if "fake" not in Path(p).name.lower() and "stand" not in Path(p).name.lower()]
    ev = f"decisions/report referencing mock/: {refs or 'none'}. Synthetic/stand-in outputs inside the REAL " \
         f"artifacts area: {fake_in_artifacts}; PDFs with PLACEHOLDER disclaimer text but ordinary names: " \
         f"{pdfs_unmarked}. Stale mock outputs: see C3."
    bad = refs or pdfs_unmarked
    finish("C5", "Mocks separated from real files", "FAIL" if bad else "PASS", ev,
           fix="move preview outputs out of artifacts/ (e.g. artifacts/p3/_preview_synthetic/ with a README) or "
               "prefix every file SYNTHETIC_; never commit the placeholder-disclaimer PDFs", owner="P3")


# ===================================================================================== D
FROZEN = {("p3_det.api", "detect"): ["img_for_analysis", "crop_box"],
          ("p3_xai.api", "explain"): ["img_for_analysis", "crop_box"],
          ("p3_xai.api", "render_overlay"): ["img512", "heatmap", "boxes"],
          ("p3_ops.api", "make_pdf"): ["case_result"],
          ("p3_ops.api", "log_case"): ["case_result", "operating_point"]}


def test_D1():
    import importlib
    extras, wrong = [], []
    for (mod, fn), params in FROZEN.items():
        sig = inspect.signature(getattr(importlib.import_module(mod), fn)).parameters.values()
        pos = [p.name for p in sig if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if pos != params:
            wrong.append(f"{mod}.{fn}{pos}")
        extras += [f"{mod}.{fn}(*, {p.name}={p.default!r})" for p in sig if p.name not in params]
    from p3_det.result import DetResult
    fields = list(DetResult.__dataclass_fields__)
    ev = f"frozen positional parameters: {'all match' if not wrong else wrong}; DetResult fields {fields}; " \
         f"keyword-only EXTENSIONS beyond the frozen signatures: {extras}"
    status = "FAIL" if wrong or extras or fields != ["boxes_full512", "scores", "n_lesions_above_thr"] else "PASS"
    finish("D1", "R3 signatures exactly as frozen", status, ev,
           fix="either get the extensions approved (contract notes items 8 and 15, CCR if contracts-v1 is tagged) or "
               "remove them (overlay path via env, image passed by P4 another way)", owner="P4/P1 approval, then P3")


def test_D2a():
    import checks
    hits = checks.code_markers(REPO)
    finish("D2a", "No STUB/TODO/placeholder in REAL modules of src/p3_*", "FAIL" if hits else "PASS",
           f"{len(hits)} marker line(s) in real modules (mock module {sorted(checks.MOCK_MODULES)} and tbcore adapter "
           f"{sorted(checks.ADAPTER_MODULES)} are reported by D2c):\n" + "\n".join(hits),
           fix="resolve the marker, or move TB_MOCK-only code into a _mock.py module", owner="P3")


def test_D2b():
    import checks
    pending = checks.config_pending(REPO)
    if not pending:
        finish("D2b", "Training-config values set (awaiting P3-01 feasibility)", "PASS", "epochs and batch_size set")
    finish("D2b", "Training-config values set (awaiting P3-01 feasibility)", "NOT_YET",
           "RECLASSIFIED from D2: these are not code placeholders but values that need the P3-01 T4 numbers (J1). "
           "Training refuses to start while they are null. Pending: " + "; ".join(pending),
           owner="you (Kaggle T4 run, J1)", command="fill schedule.epochs and schedule.batch_size in "
           "src/p3_det/configs/dfine_512.yaml and dfine_640.yaml from docs/p3_dfine_feasibility.md")


def test_D2c():
    users = []
    for f in sorted(glob.glob(str(REPO / "src/p3_*/**/*.py"), recursive=True)):
        if Path(f).name == "_tbcore_adapter.py":
            continue
        if re.search(r"^\s*(from tbcore|import tbcore)", Path(f).read_text(encoding="utf-8"), re.M):
            users.append(Path(f).relative_to(REPO).as_posix())
    adapter = REPO / "src/p3_ops/_tbcore_adapter.py"
    if users or not adapter.exists():
        finish("D2c", "Assumed tbcore API names confined to one adapter", "FAIL",
               f"adapter present: {adapter.exists()}; modules importing tbcore directly: {users}",
               fix="route every tbcore use through src/p3_ops/_tbcore_adapter.py", owner="P3")
    from p3_ops import _tbcore_adapter as ad
    if not ad.tbcore_available():
        finish("D2c", "Assumed tbcore API names confined to one adapter", "NOT_YET",
               f"all tbcore names live in src/p3_ops/_tbcore_adapter.py ({', '.join(ad.ASSUMED_API)}); tbcore not "
               "installed, so the names are still assumptions", owner="P4 (publish tbcore API)",
               command="pytest tests/p3_ops/test_tbcore_adapter.py (fails loudly once tbcore exists and names differ)")
    missing = ad.check_api()
    finish("D2c", "Assumed tbcore API names confined to one adapter", "FAIL" if missing else "PASS",
           f"tbcore present; missing names: {missing or 'none'}", fix="update ASSUMED_API in the adapter")


def test_D3():
    src = (REPO / "src/p3_det/api.py").read_text()
    reads = all(n in src for n in ("p3_det_input_size.json", "p3_det_checkpoint.json", "p3_det_conf_threshold.json"))
    import checks
    literals = checks.hardcoded_settings(src)                        # assignments only (see checks.py)
    rc, tail = run_pytest("tests/p3_det/test_r3_detect.py::test_model_loaded_once")
    ok = reads and not literals and rc == 0
    finish("D3", "detect() reads D16-D18; model cached", "PASS" if ok else "FAIL",
           f"reads D16/D17/D18 files: {reads}; numeric settings hard-coded in api.py: {literals or 'none'} "
           f"(MIN_SCORE 0.01 is the C10 rule; the mock threshold lives in _stub.py); cache test: {summary_line(tail)}",
           fix="read every setting from the decision files")


def test_D4():
    rc, tail = run_pytest("tests/p3_det/test_r3_detect.py::test_detect_equals_c10_for_50_development_images")
    finish("D4", "detect() equals C10 on 50 development images", "NOT_YET",
           f"no real checkpoint/C10 yet. Same test on a real (tiny, random) D-FINE checkpoint + fake data: "
           f"{summary_line(tail)}", owner="you (after D16-D18 + checkpoint)",
           command="pytest tests/p3_det/test_r3_detect.py (then rerun this audit after make p3-export-c10)")


def test_D5():
    rc, tail = run_pytest("tests/p3_xai/test_gradcam.py", "-k",
                          "heatmap_shape or deterministic or api_explain_uses_cached")
    finish("D5", "explain() float16 [512,512] [0,1], zero outside crop, deterministic", "NOT_YET",
           f"P2's classifier_module() is not available; same checks on a DINOv2 stand-in: {summary_line(tail)}",
           owner="P2 (R2 classifier_module)", command="pytest tests/p3_xai/test_gradcam.py")


def test_D6():
    rc, tail = run_pytest("tests/p3_xai/test_overlay.py::test_zero_boxes_draws_no_boxes_and_header_says_heatmap_only",
                          "tests/p3_ops/test_pdf_report.py::test_detector_contingency_omits_lesion_section")
    finish("D6", "Contingency: detector disabled handled", "NOT_YET",
           "the detector_enabled flag was not implemented: its source of truth is undefined (task stopped, see "
           f"earlier report; needs P4). Partial: no-box overlay and PDF without a detector block: {summary_line(tail)}",
           owner="P4 (flag source: D21 key, C14 field or new decision)",
           command="after P4 decides: implement empty DetResult when disabled + test")


# ===================================================================================== E
def test_E1():
    rc, tail = run_pytest("tests/p3_det/test_det_metrics.py", "-k",
                          "pycocotools or froc_hand or bootstrap_resamples or ap50_result_has")
    finish("E1", "AP50 = pycocotools (1e-3), FROC hand fixture, bootstrap over images with fixed seed",
           "PASS" if rc == 0 else "FAIL", summary_line(tail), fix="see failing test")


def test_E2():
    from p3_det.metrics import MetricResult
    refused = 0
    for bad in ({"n_images": None}, {"n_images": 0, "n_tb_images": 0, "n_non_tb_images": 0}):
        try:
            MetricResult(**{"name": "x", "value": .5, "ci_low": .4, "ci_high": .6, "n_images": 10, "n_tb_images": 5,
                            "n_non_tb_images": 5, "n_boxes": 3, "ci_method": "wilson", **bad})
        except ValueError:
            refused += 1
    fields = list(MetricResult.__dataclass_fields__)
    finish("E2", "Metric results carry n and CI; no-n refused", "PASS" if refused == 2 else "FAIL",
           f"MetricResult fields {fields}; refused {refused}/2 results without n", fix="enforce n in __post_init__")


def test_E3():
    d = load_json(DECISIONS["D16"])
    finish("E3", "Development AP50 512/640 with CIs; chosen = D16; contingency", "NOT_YET",
           f"D16 value empty: no training runs yet. Contingency (AP50 < 0.05) cannot be evaluated; the tooling stops "
           f"and refuses to lock when it triggers (tests/p3_det/test_select_size.py). D16 locked={d['locked']}",
           owner="you (Kaggle training) + P1 data + P2 D08/D09",
           command="python -m p3_det.select_size compare --run512 ... --run640 ...")


def test_E4():
    import numpy as np

    from p3_det.data import AugmentConfig, augment
    r = np.random.default_rng(2024)
    bad = 0
    for i in range(1000):
        S = (512, 640)[i % 2]
        k = int(r.integers(1, 5))
        w, h = r.uniform(2, 150, (2, k))
        x0, y0 = r.uniform(0, S - w), r.uniform(0, S - h)
        boxes = np.stack([x0, y0, x0 + w, y0 + h], 1)
        _, out, _ = augment(np.full((S, S), 100, np.uint8), boxes, r, AugmentConfig(hflip_p=0.5))
        ok = len(out) == k and (out >= 0).all() and (out[:, [0, 2]] <= S).all() and (out[:, [1, 3]] <= S).all() \
            and (out[:, 2] - out[:, 0] >= 2).all() and (out[:, 3] - out[:, 1] >= 2).all()
        bad += not ok
    rc, tail = run_pytest("tests/p3_det/test_det_data.py", "-k", "round_trip or sealed")
    finish("E4", "Augmentation in bounds (1000 samples), round trip < 1 px, sealed refused",
           "PASS" if bad == 0 and rc == 0 else "FAIL",
           f"1000-sample property run here: {bad} violations; round-trip + sealed-split tests: {summary_line(tail)}",
           fix="see violations / failing tests")


def test_E5():
    from p3_det.train.config import load_run_config
    a, b = load_run_config("dfine_512.yaml"), load_run_config("dfine_640.yaml")
    same = a.schedule_hash == b.schedule_hash and a.seed == b.seed
    runs = glob.glob(str(REPO / "artifacts/p3/det/runs/*/run_meta.json"))
    if not runs:
        finish("E5", "Seeds recorded; 512 and 640 same schedule and seed", "NOT_YET" if same else "FAIL",
               f"configs: same schedule hash {same} ({a.schedule_hash[:19]}...), seed {a.seed}; no run logs yet",
               owner="you (Kaggle training)", fix="configs must differ only in run_name/input_size")
    metas = [json.loads(Path(p).read_text()) for p in runs]
    ok = all("seed" in m and m.get("seed_util") for m in metas)
    finish("E5", "Seeds recorded; 512 and 640 same schedule and seed", "PASS" if ok and same else "FAIL",
           f"{len(metas)} runs, seeds {[m.get('seed') for m in metas]}")


# ===================================================================================== F
def test_F1():
    rc, tail = run_pytest("tests/p3_xai/test_gradcam.py", "-k", "inspection or reshape or zero_patch_gradient")
    src = (REPO / "src/p3_xai/gradcam.py").read_text()
    reads_cfg = "num_register_tokens" in src and "patch_size" in src and "hw[0] // patch" in src
    finish("F1", "Last block target, CLS/register tokens from config, grid from input/patch size",
           "PASS" if rc == 0 and reads_cfg else "FAIL",
           f"{summary_line(tail)} (CLS-only and 4-register stand-ins); token counts read from config: {reads_cfg}. "
           "Target is the last block's norm1 (attention input), see D19 rationale",
           fix="see failing test")


def test_F2():
    res = glob.glob(str(REPO / "artifacts/p3/xai/sanity/randomisation_*.json"))
    finish("F2", "Randomisation Spearman < 0.5 on 20 development images, saved", "NOT_YET",
           f"saved results: {len(res)}; needs P2's classifier + P1 data. NOTE: the pre-registered signed rule can be "
           "passed by an inverted map (found on stand-ins: mean rho -0.73); awaiting your decision on |rho| before "
           "committing D19's criterion", owner="P2, P1, you", command="python -m p3_xai.sanity --write-d19")


def test_F3():
    finish("F3", "Heatmaps regenerated with the REAL classifier", "NOT_YET",
           "no real C11 exists at all (artifacts/p3/xai has no model folders); mock C11 is in mock/ only",
           owner="P2 (real classifier_module after D11)", command="python -m p3_xai.export_c11 ...")


def test_F4():
    rc, tail = run_pytest("tests/p3_xai/test_localisation.py", "-k",
                          "pointing_peak_inside or pointing_peak_outside or uniform or all_mass_inside")
    finish("F4", "Localisation synthetic tests", "PASS" if rc == 0 else "FAIL", summary_line(tail),
           fix="see failing test")


def test_F5():
    t = glob.glob(str(REPO / "artifacts/p3/xai/loc_*_development_v1.parquet"))
    finish("F5", "C11 for crop and full-image models", "NOT_YET", f"{len(t)} real tables; R2 cannot yet load the "
           "full-image model (contract notes item 14) and D14 is missing", owner="P2",
           command="python -m p3_xai.export_c11 --crop-model <id> --full-model <id>")


def test_F6():
    finish("F6", "GPU: explain() <= 5.5 GB VRAM, detect() <= 500 ms", "MANUAL",
           "no CUDA GPU on this machine; needs the real models", owner="you (RTX 4050)",
           command="pytest tests/p3_xai/test_gradcam.py -m gpu -s ; make p3-latency-detect")


# ===================================================================================== G
def test_G1():
    rc, tail = run_pytest("tests/p3_xai/test_overlay.py")
    names = [Path(p).name for p in glob.glob(str(REPO / "artifacts/p3/**/*overlay*.png"), recursive=True)]
    bad = [n for n in names if not re.fullmatch(r"(c_[0-9a-z]+_overlay|overlay_full|R\d\d_[A-Za-z0-9_-]+)\.png", n)]
    finish("G1", "Overlay caption, 0/1/many boxes, zero heatmap, no identifiers in names",
           "PASS" if rc == 0 and not bad else "FAIL", f"{summary_line(tail)}; overlay file names on disk: {names}; "
           f"suspicious: {bad or 'none'}", fix="see failing test / rename")


def test_G2():
    rc, tail = run_pytest("tests/p3_ops/test_pdf_report.py")
    finish("G2", "PDF for 7 fixture types; disclaimer + scope imported; < 2 MB; no analysis on rejected/unreadable",
           "NOT_YET" if rc == 0 else "FAIL",
           f"with STAND-INS for P4's 7 fixtures and for tbcore.disclaimer: {summary_line(tail)} (incl. grep proving "
           "the texts are not retyped). Real check needs P4's fixtures and tbcore",
           owner="P4 (fixtures P4-02, tbcore.disclaimer)", fix="see failing test")


def test_G3():
    p = __import__("subprocess").run([PY, "notebooks/p3/p3_15_pdf_preview.py"], cwd=REPO, capture_output=True, text=True)
    ok = p.returncode == 0 and (REPO / "artifacts/p3/ops/pdf_preview/contact_sheet.png").exists()
    finish("G3", "Page 1 of each fixture PDF rendered for visual check", "MANUAL" if ok else "FAIL",
           f"regenerated artifacts/p3/ops/pdf_preview/<type>.png and contact_sheet.png ({'ok' if ok else p.stderr[-300:]}); "
           "shown in chat for your layout check", owner="you",
           command="open artifacts/p3/ops/pdf_preview/contact_sheet.png", fix="see error")


def test_G4():
    rc, tail = run_pytest("tests/p3_ops/test_logger.py")
    sys.path.insert(0, str(REPO / "tests/p3_ops"))
    sys.path.insert(0, str(REPO / "tests/p3_mock"))
    from test_logger import IDENTIFIER_PATTERNS
    texts = {}
    if (REPO / "mock/logs/predictions.jsonl").exists():
        texts["mock/logs/predictions.jsonl"] = (REPO / "mock/logs/predictions.jsonl").read_text()
    import pypdf
    for pdf in glob.glob(str(REPO / "artifacts/p3/**/*.pdf"), recursive=True):
        texts[Path(pdf).relative_to(REPO).as_posix()] = " ".join(pg.extract_text() for pg in pypdf.PdfReader(pdf).pages)
    texts["overlay/preview file names"] = " ".join(Path(p).name for p in glob.glob(str(REPO / "artifacts/p3/**/*.png"), recursive=True))
    texts["tests/p3_mock/p3_case_results.py (C14 stand-ins)"] = (REPO / "tests/p3_mock/p3_case_results.py").read_text()
    # artifact file names / artifact paths legitimately end in .png: the "image file name" pattern applies to
    # log lines and PDF text only (where an input file name would be a leak)
    import checks
    hits = checks.identifier_hits(texts, IDENTIFIER_PATTERNS,
                                  {"overlay/preview file names", "tests/p3_mock/p3_case_results.py (C14 stand-ins)"})
    ok = rc == 0 and not hits
    finish("G4", "Logger: one line/call, versions required, 50 parallel, no identifiers anywhere",
           "PASS" if ok else "FAIL", f"logger tests: {summary_line(tail)}; identifier scan over {list(texts)}: "
           f"{hits or 'no matches'}", fix="remove the identifier source")


def test_G5():
    rc, tail = run_pytest("tests/p3_ops/test_logger.py::test_user_override_written_exactly")
    finish("G5", "user_override written exactly", "PASS" if rc == 0 else "FAIL", summary_line(tail))


# ===================================================================================== H
def test_H1():
    exists = (REPO / "src/p3_ops/sealed_run.py").exists()
    rc, tail = run_pytest("tests/p3_det/test_det_data.py", "-k", "sealed")
    finish("H1", "sealed_run.py refuses without a valid token", "FAIL" if exists and rc else "NOT_YET",
           f"sealed_run.py exists: {exists} (not written: P3-13 stopped at its stop conditions, no gate). Existing "
           f"guards: data adapter refuses sealed splits without tbcore.guard: {summary_line(tail)}",
           owner="P4 (gate P4-09 + token API), then P3", command="build sealed_run.py against P4's gate API")


def test_H2():
    finish("H2", "Audit: internal_test once, then challenge_test once", "NOT_YET",
           "no sealed run has happened (no token, no gate, no audit file)", owner="P4 (gate) / P3-13")


def test_H3():
    finish("H3", "No locked decision / threshold / weights changed after the internal-test audit", "NOT_YET",
           f"{NO_REPO}; no internal-test audit timestamp exists", owner="after P3-13",
           command="git log --since=<audit ts> -- decisions/p3_* artifacts/p3/det/checkpoints")


def test_H4():
    sealed = [p for p in glob.glob(str(REPO / "artifacts/p3/**/*"), recursive=True)
              if re.search(r"internal_test|challenge_test", p)]
    finish("H4", "Sealed artifacts not modified after audit", "FAIL" if sealed else "NOT_YET",
           f"sealed-split files under artifacts/p3: {sealed or 'none'}", owner="after P3-13",
           fix="sealed files exist without an audit record: investigate immediately")


# ===================================================================================== I
def test_I1():
    f = [p for p in ("artifacts/p3/latency/e6_gpu.csv", "artifacts/p3/latency/e6_cpu.csv") if (REPO / p).exists()]
    finish("I1", "E6 CSVs with required columns, n = 100", "NOT_YET",
           f"present: {f or 'none'}; needs P4's run_batch (R4). Tool + columns tested in tests/p3_ops/test_latency.py",
           owner="P4 (R4), then you", command="make p3-latency-gpu ; make p3-latency-cpu (8 GB limit)")


def test_I2():
    rc, tail = run_pytest("tests/p3_ops/test_latency.py", "-k", "verdict or known_numbers")
    finish("I2", "PASS/FAIL from the median, matches report", "NOT_YET",
           f"no E6 results; verdict logic tests: {summary_line(tail)}; report row is TBD", owner="you (after I1)")


def test_I3():
    rc, tail = run_pytest("tests/p3_ops/test_cloud_checkpoint.py")
    log = (REPO / "docs/p3_kaggle_runbook.md").read_text(encoding="utf-8")
    row = re.search(r"\| *(\d{4}-\d\d-\d\d) *\| *Kaggle", log)
    if rc != 0:
        finish("I3", "Checkpoint helper kill/resume; deliberate-kill log", "FAIL", summary_line(tail), fix="see test")
    finish("I3", "Checkpoint helper kill/resume; deliberate-kill log", "PASS" if row else "MANUAL",
           f"CPU kill/resume tests: {summary_line(tail)}; runbook log row filled: {bool(row)}", owner="you (Kaggle)",
           command="docs/p3_kaggle_runbook.md section 10 procedure, then send me the result")


# ===================================================================================== J
def test_J1():
    t = (REPO / "docs/p3_dfine_feasibility.md").read_text(encoding="utf-8")
    filled = "| D-FINE-S | 512 | 640 | |" not in t
    finish("J1", "Feasibility doc has real numbers, peak <= 14 GB", "PASS" if filled else "MANUAL",
           f"result tables {'filled' if filled else 'still empty (template)'}", owner="you (Kaggle T4)",
           command="run notebooks/p3/p3_01_dfine_feasibility.ipynb on a Kaggle T4 and send me the summary")


def test_J2():
    t = (REPO / "docs/p3_gradcam_review.md").read_text(encoding="utf-8")
    rule_first = "BEFORE any case was selected" in t
    sel = (REPO / "artifacts/p3/review/selection.csv").exists()
    finish("J2", "Review doc: 40 cases, comments, sign-off; rule before selection", "NOT_YET",
           f"selection rule pre-written: {rule_first}; selection run: {sel}; comments/sign-off: empty. Needs P4's C12 "
           "(and P1/P2 for overlays), then the session", owner="P4 (C12), then you + guide",
           command="python -m p3_xai.review select ; python -m p3_xai.review render")


def _recompute_table() -> list[tuple[str, str, str, bool]]:
    import yaml

    from p3_det import select_size
    from p3_det.data import config as dcfg
    from p3_det.metrics import detection
    from p3_det import lock_threshold
    from p3_ops import latency
    from p3_xai import d19, sanity
    meth = (REPO / "docs/methodology.md").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    latent = re.search(r"Latent TB (\d+) (\d+) (\d+) (\d+)", meth).groups()
    both = re.search(r"Active and latent TB (\d+) (\d+) (\d+) (\d+)", meth).groups()
    train = re.search(r"([\d,]+) (\d+)\nDevelopment", meth) or re.search(r"\n([\d,]+) (\d+)\n", meth)
    dev = re.search(r"Development Stratified 10% of the official training split (\d+) (\d+)", meth).groups()
    cands = [
        ("212 latent", "docs/methodology.md", latent[3], latent[3] == "212"),
        ("54 combined", "docs/methodology.md", both[3], both[3] == "54"),
        ("36 latent (validation)", "docs/methodology.md", latent[1], latent[1] == "36"),
        ("7 combined (validation)", "docs/methodology.md", both[1], both[1] == "7"),
        ("~540 TB / ~5,940 training", "docs/methodology.md", f"{train.group(2)} / {train.group(1)}",
         (train.group(2), train.group(1)) == ("540", "5,940")),
        ("~60 development TB", "docs/methodology.md", dev[1], dev[1] == "60"),
        ("seed 0", "src/p3_det/configs/dfine_512.yaml",
         str(yaml.safe_load((REPO / "src/p3_det/configs/dfine_512.yaml").read_text())["schedule"]["seed"]), True),
        ("margin 0.10 provisional", "src/p3_det/data/config.py", str(dcfg.PROVISIONAL_MARGIN), dcfg.PROVISIONAL_MARGIN == 0.10),
        ("AP50 < 0.05 contingency", "decisions/p3_det_input_size.json", str(select_size.CONTINGENCY_AP50),
         "< 0.05" in load_json(DECISIONS["D16"])["criterion_preregistered"]),
        ("5% non-TB rule", "decisions/p3_det_conf_threshold.json", f"{lock_threshold.MAX_SHARE:.0%}",
         "at most 5%" in load_json(DECISIONS["D18"])["criterion_preregistered"]),
        ("0.001 grid", "decisions/p3_det_conf_threshold.json", str(lock_threshold.STEP),
         "0.010, 0.011" in load_json(DECISIONS["D18"])["criterion_preregistered"]),
        ("Spearman < 0.5, 20 images", "decisions/p3_gradcam_config.json", f"{sanity.PASS_BELOW}, 20",
         "< 0.5" in d19.CRITERION and "20 development images" in d19.CRITERION),
        ("5.5 GB VRAM", "decisions/p3_gradcam_config.json", str(sanity.VRAM_LIMIT_GB), "5.5 GB" in d19.CRITERION),
        ("956d170 D-FINE pin", "decisions/p3_licences.json", load_json(DECISIONS["D20"])["value"]["dfine"]["commit"][:7],
         load_json(DECISIONS["D20"])["value"]["dfine"]["commit"].startswith("956d170")),
        ("< 2 s GPU / < 10 s CPU", "README.md", "found" if "< 2 s (GPU) and < 10 s (CPU)" in readme else "missing",
         "< 2 s (GPU) and < 10 s (CPU)" in readme),
        ("100 images + 3 warm-up", "src/p3_ops/latency.py", f"{latency.N_MEASURE} + {latency.N_WARMUP}",
         (latency.N_MEASURE, latency.N_WARMUP) == (100, 3)),
        ("30-image small-n flag", "src/p3_det/metrics/detection.py", str(detection.SMALL_N), detection.SMALL_N == 30),
        ("<= 500 ms detector", "README.md", "found" if "≤ 500 ms/image" in readme else "missing", "≤ 500 ms/image" in readme),
    ]
    import random
    return random.Random(0).sample(cands, 10)


def test_J3():
    cited, missing = set(), []
    for f in sorted(glob.glob(str(REPO / "report/p3_*.md"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if "TBD" in line:
                continue
            for c in re.findall(r"\[([^\]]+)\]", line):
                for part in c.split(";"):
                    p = part.strip().split(",")[0].split(" ")[0]
                    if "/" in p or p.endswith(".md") or p.endswith(".json"):
                        cited.add(p)
                        if not (REPO / p).exists():
                            missing.append(f"{Path(f).name}: {p}")
    table = _recompute_table()
    mism = [t for t in table if not t[3]]
    body = "\n".join(f"| {a} | {b} | {c} | {'yes' if d else 'NO'} |" for a, b, c, d in table)
    rep = "\n".join(Path(f).read_text(encoding="utf-8") for f in glob.glob(str(REPO / "report/p3_*.md")))
    tbd_listed = all(x in (REPO / "report/p3_numbers_checklist.md").read_text(encoding="utf-8")
                     for x in ("D16", "D17", "D18", "t12_4", "e6_gpu", "loc_summary_development"))
    ev = (f"{len(cited)} distinct cited paths, missing on disk: {missing or 'none'}; TBD present "
          f"({rep.count('TBD')}x) and listed in report/p3_numbers_checklist.md: {tbd_listed}.\n"
          f"Recomputed 10 random numbers:\n| Number in report | Source | Recomputed | Match |\n|---|---|---|---|\n{body}")
    status = "FAIL" if missing or mism or not tbd_listed else "PASS"
    finish("J3", "Report numbers cite existing paths; 10 recomputed; TBD listed", status, ev,
           fix="fix the listed citations / numbers")


def test_J4():
    rep = "\n".join(Path(f).read_text(encoding="utf-8") for f in glob.glob(str(REPO / "report/p3_*.md")))
    need = {"latent caveat": "not reliable", "attention evidence": "attention evidence, not a finding",
            "not clinical proof": "not** proof", "~540 boxed": "about 540 TB"}
    missing = [k for k, v in need.items() if v not in rep]
    finish("J4", "Latent caveat, attention evidence, not-clinical-proof, ~540 limitation present",
           "FAIL" if missing else "PASS", f"missing: {missing or 'none'}", fix="add the missing statements")


def test_J5():
    finish("J5", "Every reported metric has n and 95% CI", "NOT_YET",
           "no metric values reported yet (all TBD); every metric table in report/p3_*.md has n and 95% CI columns",
           owner="you (after results)")


# ===================================================================================== K
def test_K1():
    finish("K1", "P4 run_batch on a mixed batch completes, one log line + one PDF per case", "NOT_YET",
           "p4_pipeline (R4) not in the repo", owner="P4")


def test_K2():
    finish("K2", "run_case equals offline triage on 50 dev images (detector + explainability blocks)", "NOT_YET",
           "no run_case, no C12", owner="P4")


def test_K3():
    finish("K3", "P1 dashboard opens my overlay + PDF at 1366x768", "NOT_YET", "P1 dashboard not in the repo",
           owner="P1, then you (MANUAL visual check)")
