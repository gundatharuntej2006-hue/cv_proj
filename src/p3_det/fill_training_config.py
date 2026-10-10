"""P3-01 -> P3-06: turn the pasted Kaggle T4 feasibility JSON into batch_size and epochs.

The last cell of notebooks/p3/p3_01_dfine_feasibility.ipynb prints one JSON block (``paste_block``).
Save it to a file, then:

    python -m p3_det.fill_training_config --json t4.json --hours-per-session H            # shows the diff only
    python -m p3_det.fill_training_config --json t4.json --hours-per-session H --write    # saves it

What it does:
  1. Checks the acceptance rule of docs/p3_dfine_feasibility.md section 4 for both sizes: peak reserved GPU
     memory <= 14 GB at the chosen batch (worst-case multi-scale size) and in the real epoch.
  2. Picks ONE batch_size for 512 and 640 (the smaller of the two largest batches that fit), because D16 needs
     identical schedules; warns when the two sizes differ a lot.
  3. Proposes ONE epochs value for both sizes from the slower size's seconds per epoch (training + development
     evaluation) and YOUR time budget per Kaggle session (--hours-per-session, no default). The arithmetic is
     printed. D17 later picks the best checkpoint by development AP50, so epochs is an upper bound.
  4. Writes batch_size / epochs into src/p3_det/configs/dfine_512.yaml and dfine_640.yaml, fills the result
     tables of docs/p3_dfine_feasibility.md and stores the pasted JSON under artifacts/p3/feasibility/.
     Without --write only the diff is shown.

Refuses (exit 2) on missing or malformed JSON, a failed acceptance rule, a seed / AMP / model that differs from
the configs, a non-T4 GPU, or configs that already hold different values (unless --overwrite).
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {512: REPO / "src/p3_det/configs/dfine_512.yaml", 640: REPO / "src/p3_det/configs/dfine_640.yaml"}
DOC = REPO / "docs/p3_dfine_feasibility.md"
ARTIFACT_DIR = REPO / "artifacts/p3/feasibility"
SCHEMA = "p3-01-feasibility-v1"
PEAK_LIMIT_GB = 14.0
SIZES = (512, 640)
DOC_BATCHES = (2, 4, 6, 8, 12, 16, 20, 24, 32, 48)          # columns of the section 5.1 table
DIFFER_A_LOT = 1.5                                          # largest-fit ratio that triggers the warning
KAGGLE_SESSION_MAX_H = 12.0                                 # docs/p3_kaggle_runbook.md section 8


class Refused(ValueError):
    """Input not usable; nothing was written."""


# --- producer (notebook) -------------------------------------------------------------------------------

def paste_block(results: dict, model: str | None = None) -> dict:
    """Compact JSON for the user to paste back, built from the feasibility script's results."""
    runs = results["runs"]
    models = sorted({r["model"] for r in runs})
    model = model if model in models else models[0]
    env, st = results["env"], results["settings"]
    out = {"schema": SCHEMA, "model": model, "gpu": env.get("gpu"), "gpu_total_gb": env.get("gpu_total_gb"),
           "amp": st["amp"], "seed": env["seed"], "seed_util": env.get("seed_util"), "torch": env.get("torch"),
           "cuda": env.get("cuda"), "dfine_commit": env.get("dfine_commit"), "platform": env.get("platform"),
           "date_utc": env.get("timestamp_utc"), "peak_limit_gb": st["peak_limit_gb"],
           "n_mock": st["n_mock_images"], "n_train": st.get("n_train_for_extrapolation"),
           "n_dev": st.get("n_dev_for_extrapolation"), "sizes": {}}
    for r in (r for r in runs if r["model"] == model):
        peaks = {str(p["batch"]): (p["peak_reserved_gb"] if p["ok"] else "OOM") for p in r["probes"]}
        fit = r["chosen_batch"]
        train_s, eval_s = r.get("s_per_epoch_extrapolated"), r.get("eval_s_extrapolated")
        out["sizes"][str(r["size"])] = {
            "max_train_scale": r["max_train_scale"],
            "largest_batch_fits": fit, "largest_batch_no_oom": r["largest_workable_batch"],
            "peak_gpu_mem_gb": peaks.get(str(fit)) if fit else None,
            "peak_by_batch_gb": peaks,
            "timed_at_batch": r.get("timed_at_batch"),
            "epoch_peak_gpu_mem_gb": r.get("epoch_peak_reserved_gb"),
            "train_s_per_epoch_mock": r.get("s_per_epoch_mock"), "train_s_per_image": r.get("s_per_image"),
            "train_s_per_epoch": train_s, "eval_s_per_epoch": eval_s,
            "s_per_epoch": round(train_s + eval_s, 1) if train_s is not None and eval_s is not None else None,
        }
    return out


# --- consumer ------------------------------------------------------------------------------------------

def _num(d: dict, key: str, where: str, *, positive: bool = True) -> float:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (positive and not v > 0):
        raise Refused(f"{where}.{key} must be a positive number, got {v!r}")
    return float(v)


def load_paste(text: str) -> dict:
    """Parse and validate the pasted block. Raises Refused with the first problem found."""
    if not text or not text.strip():
        raise Refused("no JSON given (empty input)")
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise Refused(f"malformed JSON: {e}") from None
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        raise Refused(f"not a P3-01 paste block (schema must be {SCHEMA!r})")
    for k, t in (("model", str), ("gpu", str), ("amp", bool), ("seed", int), ("sizes", dict)):
        if not isinstance(d.get(k), t) or (t is int and isinstance(d.get(k), bool)):
            raise Refused(f"field {k!r} missing or not {t.__name__}")
    for s in SIZES:
        e = d["sizes"].get(str(s))
        if not isinstance(e, dict):
            raise Refused(f"size {s} missing from 'sizes'")
        w = f"sizes.{s}"
        if not isinstance(e.get("largest_batch_fits"), int) or e["largest_batch_fits"] < 1:
            raise Refused(f"{w}.largest_batch_fits must be a positive integer (no batch fits in "
                          f"{PEAK_LIMIT_GB} GB?), got {e.get('largest_batch_fits')!r}")
        for k in ("peak_gpu_mem_gb", "epoch_peak_gpu_mem_gb", "s_per_epoch", "train_s_per_epoch", "eval_s_per_epoch"):
            _num(e, k, w)
        if not isinstance(e.get("peak_by_batch_gb"), dict) or not e["peak_by_batch_gb"]:
            raise Refused(f"{w}.peak_by_batch_gb missing")
        if not isinstance(e.get("timed_at_batch"), int):
            raise Refused(f"{w}.timed_at_batch missing")
    return d


def check_rule(d: dict) -> list[str]:
    """Acceptance rule (doc section 4); returns the violations (empty = pass)."""
    bad = []
    for s in SIZES:
        e = d["sizes"][str(s)]
        if e["peak_gpu_mem_gb"] > PEAK_LIMIT_GB:
            bad.append(f"{s}: peak {e['peak_gpu_mem_gb']} GB at batch {e['largest_batch_fits']} > {PEAK_LIMIT_GB} GB")
        if e["epoch_peak_gpu_mem_gb"] > PEAK_LIMIT_GB:
            bad.append(f"{s}: real-epoch peak {e['epoch_peak_gpu_mem_gb']} GB > {PEAK_LIMIT_GB} GB")
    return bad


def choose_batch(d: dict) -> tuple[int, list[str]]:
    """One batch for both sizes: the smaller largest-fit. Returns (batch, notes)."""
    fits = {s: d["sizes"][str(s)]["largest_batch_fits"] for s in SIZES}
    b = min(fits.values())
    notes = [f"largest batch that fits: 512 -> {fits[512]}, 640 -> {fits[640]}; common batch_size = min = {b}"]
    for s in SIZES:
        peak = d["sizes"][str(s)]["peak_by_batch_gb"].get(str(b))
        if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak > PEAK_LIMIT_GB:
            raise Refused(f"batch {b} has no measured peak <= {PEAK_LIMIT_GB} GB at size {s} (got {peak!r})")
        notes.append(f"peak at batch {b}, size {s}: {peak} GB (<= {PEAK_LIMIT_GB} GB)")
        if d["sizes"][str(s)]["timed_at_batch"] != b:
            notes.append(f"WARNING: size {s} was timed at batch {d['sizes'][str(s)]['timed_at_batch']}, not {b}; "
                         "its seconds per epoch are approximate")
    ratio = max(fits.values()) / b
    if ratio >= DIFFER_A_LOT:
        big = max(fits, key=fits.get)
        notes.append(f"NOTE: the two sizes differ a lot ({fits[big]} vs {b}, x{ratio:.1f}). Size {big} could use a "
                     f"bigger batch, but D16 needs identical schedules, so both use {b}; size {big} runs below its "
                     "memory capacity (slower per epoch, same results rule).")
    return b, notes


def propose_epochs(d: dict, hours_per_session: float, sessions_per_size: int = 1,
                   setup_minutes: float = 0.0) -> tuple[int, list[str]]:
    """Same epochs for both sizes, from the slower size. Returns (epochs, arithmetic lines)."""
    if not hours_per_session > 0:
        raise Refused("--hours-per-session must be > 0")
    if sessions_per_size < 1:
        raise Refused("--sessions-per-size must be >= 1")
    spe = {s: d["sizes"][str(s)]["s_per_epoch"] for s in SIZES}
    slow = max(spe, key=spe.get)
    usable = hours_per_session * 3600 - setup_minutes * 60
    if usable <= 0:
        raise Refused("setup time uses the whole session")
    total = usable * sessions_per_size
    epochs = math.floor(total / spe[slow])
    lines = []
    for s in SIZES:
        e = d["sizes"][str(s)]
        lines.append(f"size {s}: {e['train_s_per_epoch']} s training ({d.get('n_train')} images) + "
                     f"{e['eval_s_per_epoch']} s development eval ({d.get('n_dev')} images) = {spe[s]} s/epoch")
    lines += [f"slower size = {slow} ({spe[slow]} s/epoch); both sizes use the same epochs",
              f"usable seconds per session = {hours_per_session} h x 3600 - {setup_minutes} min x 60 = {usable:.0f} s",
              f"budget per size = {usable:.0f} s x {sessions_per_size} session(s) = {total:.0f} s",
              f"epochs = floor({total:.0f} / {spe[slow]}) = {epochs}"]
    if epochs < 1:
        raise Refused("\n".join(lines + ["the budget does not fit a single epoch"]))
    gpu_h = epochs * (spe[512] + spe[640]) / 3600
    lines.append(f"GPU time for both runs = {epochs} x ({spe[512]} + {spe[640]}) s = {gpu_h:.1f} h, plus setup; "
                 "compare with your weekly Kaggle quota (runbook section 8)")
    if hours_per_session > KAGGLE_SESSION_MAX_H:
        lines.append(f"WARNING: {hours_per_session} h is above the ~{KAGGLE_SESSION_MAX_H:.0f} h Kaggle session limit")
    return epochs, lines


# --- writers -------------------------------------------------------------------------------------------

_LINE = {"epochs": re.compile(r"^(?P<ind>\s*)epochs:\s*(?P<val>[^#\s]+)(?P<rest>.*)$", re.M),
         "batch_size": re.compile(r"^(?P<ind>\s*)batch_size:\s*(?P<val>[^#\s]+)(?P<rest>.*)$", re.M)}


def new_config_text(text: str, batch: int, epochs: int, stamp: str, overwrite: bool = False) -> str:
    import yaml
    sched = yaml.safe_load(text)["schedule"]
    for key, val, why in (("batch_size", batch, "common batch for 512 and 640"),
                          ("epochs", epochs, "same epochs for 512 and 640")):
        cur = sched.get(key)
        if cur is not None and cur != val and not overwrite:
            raise Refused(f"schedule.{key} is already {cur}; refusing to change it to {val} (use --overwrite)")
        m = _LINE[key].search(text)
        if m is None:
            raise Refused(f"schedule.{key} line not found")
        line = f"{m['ind']}{key}: {val}"
        line = f"{line:<47}# P3-01 Kaggle T4 run {stamp}: {why} (p3_det.fill_training_config)"
        text = text[:m.start()] + line + text[m.end():]
    if yaml.safe_load(text)["schedule"].get("batch_size") != batch or yaml.safe_load(text)["schedule"]["epochs"] != epochs:
        raise Refused("config rewrite did not produce the expected values")
    return text


def _fmt(v) -> str:
    return "" if v is None else str(v)                       # values arrive already rounded; "OOM" passes through


def new_doc_text(text: str, d: dict, batch: int, epochs: int, notes: list[str], arithmetic: list[str],
                 paste_rel: str, inputs: dict) -> str:
    m = d["model"].upper()
    day = (d.get("date_utc") or "")[:10] or "unknown date"
    text = re.sub(r"^Status: \*\*template, awaiting Kaggle T4 numbers\*\*\.",
                  f"Status: **filled from the Kaggle T4 run of {day}** (`{paste_rel}`).", text, flags=re.M)
    text = re.sub(r"^Run info:.*$", (
        f"Run info: date `{day}`, platform `{d.get('platform')}`, GPU `{d['gpu']}` (`{d.get('gpu_total_gb')}` GB), "
        f"torch `{d.get('torch')}` / CUDA `{d.get('cuda')}`, D-FINE commit `{(d.get('dfine_commit') or '')[:7]}`, "
        f"seed `{d['seed']}`, seed util `{d.get('seed_util')}`, AMP `{'on' if d['amp'] else 'off'}`, n_mock "
        f"`{d.get('n_mock')}`, n_train (extrapolation) `{d.get('n_train')}`, n_dev `{d.get('n_dev')}`, artifact "
        f"`{paste_rel}`.").replace("\\", "\\\\"), text, count=1, flags=re.M)
    s1, s2, s3 = text.index("### 5.1"), text.index("### 5.2"), text.index("### 5.3")
    t51, t52 = text[s1:s2], text[s2:s3]
    for s in SIZES:
        e = d["sizes"][str(s)]
        pb = e["peak_by_batch_gb"]
        row1 = f"| D-FINE-{m} | {s} | {e.get('max_train_scale', '')} | " + \
               " | ".join(_fmt(pb.get(str(b))) for b in DOC_BATCHES) + " |"
        t51 = re.sub(rf"^\| D-FINE-{m} \| {s} \|.*$", row1, t51, count=1, flags=re.M)
        row2 = (f"| D-FINE-{m} | {s} | {_fmt(e.get('largest_batch_no_oom'))} | {e['largest_batch_fits']} | "
                f"{_fmt(e['peak_gpu_mem_gb'])} | {_fmt(e['epoch_peak_gpu_mem_gb'])} (bs {e['timed_at_batch']}) | "
                f"{'yes' if max(e['peak_gpu_mem_gb'], e['epoch_peak_gpu_mem_gb']) <= PEAK_LIMIT_GB else 'no'} | "
                f"{_fmt(e.get('train_s_per_epoch_mock'))} | {_fmt(e.get('train_s_per_image'))} | "
                f"{_fmt(e['train_s_per_epoch'])} (+ {_fmt(e['eval_s_per_epoch'])} eval) |")
        t52 = re.sub(rf"^\| D-FINE-{m} \| {s} \|.*$", row2, t52, count=1, flags=re.M)
    end = text.index("## 6.")
    outcome = "\n".join([
        "### 5.3 Outcome", "",
        f"- Largest batch within {PEAK_LIMIT_GB:g} GB: 512 -> `{d['sizes']['512']['largest_batch_fits']}`, 640 -> "
        f"`{d['sizes']['640']['largest_batch_fits']}` (model `{d['model']}`). Acceptance rule (section 4) met for both "
        "sizes: `yes`.",
        "- Rows for models that were not run stay empty.", "",
        "### 5.4 Training schedule for P3-06 (written by `python -m p3_det.fill_training_config`)", "",
        f"Inputs: time budget per Kaggle session `{inputs['hours_per_session']} h`, sessions per size "
        f"`{inputs['sessions_per_size']}`, setup time per session `{inputs['setup_minutes']} min`.", "",
        f"- `batch_size: {batch}` and `epochs: {epochs}` in both `src/p3_det/configs/dfine_512.yaml` and "
        "`dfine_640.yaml` (identical schedule, D16). D17 then picks the checkpoint by development AP50.",
        *[f"- {n}" for n in notes], "", "Arithmetic:", "", "```", *arithmetic, "```", "", ""])
    return text[:s1] + t51 + t52 + outcome + text[end:]


def _diff(old: str, new: str, name: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"))


def plan(d: dict, hours_per_session: float, sessions_per_size: int, setup_minutes: float, *,
         configs: dict | None = None, doc: Path | None = None, artifact_dir: Path | None = None,
         overwrite: bool = False, allow_non_t4: bool = False) -> tuple[dict[Path, str], list[str]]:
    """Validate everything and return ({path: new text}, report lines). Writes nothing."""
    import yaml
    configs = CONFIGS if configs is None else configs              # resolved at call time (tests redirect them)
    doc = DOC if doc is None else doc
    artifact_dir = ARTIFACT_DIR if artifact_dir is None else artifact_dir
    bad = check_rule(d)
    if bad:
        raise Refused("acceptance rule failed (peak memory <= 14 GB): " + "; ".join(bad))
    if "T4" not in d["gpu"] and not allow_non_t4:
        raise Refused(f"GPU is {d['gpu']!r}, not a T4: the acceptance rule is defined on a Kaggle T4")
    for s, p in configs.items():
        cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        if cfg["schedule"]["seed"] != d["seed"]:
            raise Refused(f"seed {d['seed']} in the paste != seed {cfg['schedule']['seed']} in {Path(p).name}")
        if bool(cfg["schedule"]["amp"]) != d["amp"]:
            raise Refused(f"AMP {d['amp']} in the paste != amp {cfg['schedule']['amp']} in {Path(p).name}")
        if cfg["model"]["variant"] != d["model"]:
            raise Refused(f"model {d['model']!r} in the paste != variant {cfg['model']['variant']!r} in {Path(p).name}")
    batch, notes = choose_batch(d)
    epochs, arithmetic = propose_epochs(d, hours_per_session, sessions_per_size, setup_minutes)
    day = (d.get("date_utc") or "")[:10] or date.today().isoformat()
    paste_path = Path(artifact_dir) / f"t4_paste_{day}.json"
    try:
        paste_rel = paste_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        paste_rel = paste_path.as_posix()
    inputs = {"hours_per_session": hours_per_session, "sessions_per_size": sessions_per_size,
              "setup_minutes": setup_minutes}
    out = {Path(p): new_config_text(Path(p).read_text(encoding="utf-8"), batch, epochs, day, overwrite)
           for p in configs.values()}
    out[Path(doc)] = new_doc_text(Path(doc).read_text(encoding="utf-8"), d, batch, epochs, notes, arithmetic,
                                  paste_rel, inputs)
    out[paste_path] = json.dumps({**d, "fill_inputs": inputs, "batch_size": batch, "epochs": epochs}, indent=1) + "\n"
    report = notes + [""] + arithmetic + ["", f"=> batch_size {batch}, epochs {epochs} for BOTH 512 and 640"]
    return out, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, type=Path, help="file holding the pasted JSON block ('-' = stdin)")
    ap.add_argument("--hours-per-session", required=True, type=float,
                    help="YOUR time budget per Kaggle session for one training run (no default on purpose)")
    ap.add_argument("--sessions-per-size", type=int, default=1, help="sessions you allow per input size (default 1)")
    ap.add_argument("--setup-minutes", type=float, default=0.0,
                    help="minutes per session spent before training starts (install, data, resume); default 0")
    ap.add_argument("--write", action="store_true", help="save the changes (default: show the diff only)")
    ap.add_argument("--overwrite", action="store_true", help="replace epochs / batch_size that are already set")
    ap.add_argument("--allow-non-t4", action="store_true")
    a = ap.parse_args(argv)
    try:
        text = sys.stdin.read() if str(a.json) == "-" else a.json.read_text(encoding="utf-8") if a.json.exists() else ""
        if not text and str(a.json) != "-" and not a.json.exists():
            raise Refused(f"{a.json} not found")
        d = load_paste(text)
        changes, report = plan(d, a.hours_per_session, a.sessions_per_size, a.setup_minutes,
                               overwrite=a.overwrite, allow_non_t4=a.allow_non_t4)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print("\n".join(report))
    for p, new in changes.items():
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        rel = p.resolve().relative_to(REPO).as_posix() if p.resolve().is_relative_to(REPO) else str(p)
        print("\n" + (_diff(old, new, rel) or f"(no change: {rel})"))
    if not a.write:
        print("\nDiff only. Re-run with --write to save.")
        return 0
    for p, new in changes.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new, encoding="utf-8")
    print("\nSaved. Commit the two configs, the doc and the paste JSON together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
