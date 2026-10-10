"""P3-01 -> P3-06: fill_training_config (rule check, one batch for both sizes, epochs, refusals).

All numbers here are made up for the tests; nothing is written to the real configs or doc.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from p3_det import fill_training_config as ftc

REPO = Path(__file__).resolve().parents[2]


def _results(peaks512, peaks640, s_epoch=(10.0, 16.0), eval_s=(2.0, 3.0), epoch_peak=(9.0, 11.0), gpu="Tesla T4",
             amp=True, seed=0, model="s"):
    """A feasibility-script result dict (what the notebook's last cell reads)."""
    runs = []
    for size, peaks, se, ev, ep in ((512, peaks512, s_epoch[0], eval_s[0], epoch_peak[0]),
                                    (640, peaks640, s_epoch[1], eval_s[1], epoch_peak[1])):
        probes = [{"batch": b, "ok": p != "OOM", **({"oom": True} if p == "OOM" else {"peak_reserved_gb": p})}
                  for b, p in peaks.items()]
        ok = [p for p in probes if p["ok"]]
        within = [p for p in ok if p["peak_reserved_gb"] <= 14.0]
        runs.append({"model": model, "size": size, "max_train_scale": int(size * 1.25 / 32) * 32, "probes": probes,
                     "largest_workable_batch": max(p["batch"] for p in ok),
                     "chosen_batch": max(p["batch"] for p in within), "s_per_epoch_mock": 1.0, "s_per_image": 0.003,
                     "s_per_epoch_extrapolated": se, "eval_s_extrapolated": ev, "epoch_peak_reserved_gb": ep})
    common = min(r["chosen_batch"] for r in runs)
    for r in runs:
        r["timed_at_batch"] = common
    return {"env": {"gpu": gpu, "gpu_total_gb": 15.83, "seed": seed, "seed_util": "fallback", "torch": "2.x",
                    "cuda": "12.x", "dfine_commit": "956d1709", "platform": "kaggle",
                    "timestamp_utc": "2026-10-11T10:00:00+00:00"},
            "settings": {"amp": amp, "peak_limit_gb": 14.0, "n_mock_images": 320, "n_train_for_extrapolation": 5940,
                         "n_dev_for_extrapolation": 660},
            "runs": runs}

GOOD_PEAKS = ({2: 3.0, 4: 5.0, 8: 9.0, 12: 13.5, 16: "OOM"}, {2: 4.0, 4: 7.0, 8: 12.5, 12: "OOM"})
GOOD = _results(*GOOD_PEAKS)


def _paste(results=GOOD) -> dict:
    return ftc.load_paste(json.dumps(ftc.paste_block(results, "s")))


@pytest.fixture
def tree(tmp_path):
    cfgs = {}
    for s in (512, 640):
        cfgs[s] = tmp_path / f"dfine_{s}.yaml"
        shutil.copy(REPO / f"src/p3_det/configs/dfine_{s}.yaml", cfgs[s])
    doc = tmp_path / "p3_dfine_feasibility.md"
    shutil.copy(REPO / "docs/p3_dfine_feasibility.md", doc)
    return {"configs": cfgs, "doc": doc, "artifact_dir": tmp_path / "art"}


def _set_nulls(cfgs):
    """The tests must start from null configs even after the real ones are filled."""
    for p in cfgs.values():
        t = p.read_text(encoding="utf-8")
        t = ftc._LINE["epochs"].sub(lambda m: f"{m['ind']}epochs: null", t, count=1)
        p.write_text(ftc._LINE["batch_size"].sub(lambda m: f"{m['ind']}batch_size: null", t, count=1), encoding="utf-8")


# --- acceptance rule ---------------------------------------------------------------------------------

def test_rule_passes_and_paste_block_carries_the_numbers():
    d = _paste()
    assert ftc.check_rule(d) == []
    e = d["sizes"]["512"]
    assert e["largest_batch_fits"] == 12 and e["peak_gpu_mem_gb"] == 13.5 and e["peak_by_batch_gb"]["16"] == "OOM"
    assert e["s_per_epoch"] == 12.0 and d["gpu"] == "Tesla T4" and d["amp"] is True and d["seed"] == 0


def test_rule_fails_above_14_gb_and_plan_refuses(tree):
    d = _paste()
    d["sizes"]["640"]["epoch_peak_gpu_mem_gb"] = 14.2                      # real epoch over the limit
    assert ftc.check_rule(d) and "640" in ftc.check_rule(d)[0]
    with pytest.raises(ftc.Refused, match="acceptance rule"):
        ftc.plan(d, 9, 1, 0, **tree)
    d = _paste()
    d["sizes"]["512"]["peak_gpu_mem_gb"] = 14.01
    with pytest.raises(ftc.Refused, match="14 GB"):
        ftc.plan(d, 9, 1, 0, **tree)


# --- one batch for both sizes ------------------------------------------------------------------------

def test_same_batch_for_both_sizes(tree):
    _set_nulls(tree["configs"])
    d = _paste()
    b, notes = ftc.choose_batch(d)
    assert b == 8 == min(d["sizes"]["512"]["largest_batch_fits"], d["sizes"]["640"]["largest_batch_fits"])
    assert any("differ a lot" in n and "x1.5" in n for n in notes)        # 12 vs 8 reaches the x1.5 warning
    changes, _ = ftc.plan(d, 9, 1, 0, **tree)
    vals = [yaml.safe_load(changes[p])["schedule"] for p in tree["configs"].values()]
    assert vals[0]["batch_size"] == vals[1]["batch_size"] == 8
    assert vals[0]["epochs"] == vals[1]["epochs"]
    a, c = (yaml.safe_load(changes[p]) for p in tree["configs"].values())
    a.pop("run_name"), a.pop("input_size"), c.pop("run_name"), c.pop("input_size")
    assert a == c                                                          # configs still identical but for size


def test_warns_when_sizes_differ_a_lot():
    d = _paste(_results({2: 3.0, 8: 6.0, 16: 10.0, 24: 13.0}, {2: 6.0, 4: 9.0, 8: 13.9, 12: "OOM"}))
    b, notes = ftc.choose_batch(d)
    assert b == 8 and any("differ a lot" in n and "x3.0" in n for n in notes)


def test_common_batch_must_have_a_measured_peak():
    d = _paste()
    d["sizes"]["512"]["peak_by_batch_gb"].pop("8")
    with pytest.raises(ftc.Refused, match="no measured peak"):
        ftc.choose_batch(d)


# --- epochs ------------------------------------------------------------------------------------------

def test_epochs_from_slower_size_and_budget_with_arithmetic():
    d = _paste()                                       # s/epoch: 512 -> 12.0, 640 -> 19.0
    epochs, lines = ftc.propose_epochs(d, hours_per_session=1.0, sessions_per_size=2, setup_minutes=10)
    assert epochs == (3600 - 600) * 2 // 19
    assert any("slower size = 640" in l for l in lines) and any(f"= {epochs}" in l for l in lines)
    with pytest.raises(ftc.Refused, match="single epoch"):
        ftc.propose_epochs(d, hours_per_session=0.001)
    with pytest.raises(ftc.Refused):
        ftc.propose_epochs(d, hours_per_session=0)


# --- refusals on bad input ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, match", [
    ("", "empty"), ("   ", "empty"), ("{not json", "malformed"), ("[1, 2]", "schema"), ('{"schema": "x"}', "schema"),
])
def test_refuses_missing_or_malformed_json(text, match):
    with pytest.raises(ftc.Refused, match=match):
        ftc.load_paste(text)


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["sizes"].pop("640"), "size 640 missing"),
    (lambda d: d.pop("seed"), "seed"),
    (lambda d: d.update(amp="yes"), "amp"),
    (lambda d: d["sizes"]["512"].update(s_per_epoch=None), "s_per_epoch"),
    (lambda d: d["sizes"]["512"].update(peak_gpu_mem_gb="9"), "peak_gpu_mem_gb"),
    (lambda d: d["sizes"]["640"].update(largest_batch_fits=None), "largest_batch_fits"),
    (lambda d: d["sizes"]["640"].update(s_per_epoch=-1), "positive"),
])
def test_refuses_incomplete_block(mutate, match):
    d = ftc.paste_block(GOOD, "s")
    mutate(d)
    with pytest.raises(ftc.Refused, match=match):
        ftc.load_paste(json.dumps(d))


def test_cpu_smoke_output_is_refused():
    """A CPU smoke run has no memory numbers, so it can never fill the configs."""
    r = _results({2: 1.0}, {2: 1.0})
    d = ftc.paste_block(r, "s")
    for s in ("512", "640"):
        d["sizes"][s]["peak_gpu_mem_gb"] = d["sizes"][s]["epoch_peak_gpu_mem_gb"] = None
    with pytest.raises(ftc.Refused, match="peak_gpu_mem_gb"):
        ftc.load_paste(json.dumps(d))


@pytest.mark.parametrize("kw, match", [({"gpu": "Tesla P100-PCIE-16GB"}, "not a T4"), ({"seed": 1}, "seed"),
                                       ({"amp": False}, "AMP"), ({"model": "m"}, "model")])
def test_refuses_mismatch_with_configs(tree, kw, match):
    d = ftc.load_paste(json.dumps(ftc.paste_block(_results(GOOD_PEAKS[0], GOOD_PEAKS[1], **kw), None)))
    with pytest.raises(ftc.Refused, match=match):
        ftc.plan(d, 9, 1, 0, **tree)



def test_refuses_to_change_values_already_set(tree):
    _set_nulls(tree["configs"])
    p = tree["configs"][512]
    p.write_text(ftc._LINE["epochs"].sub(lambda m: f"{m['ind']}epochs: 3", p.read_text(encoding="utf-8"), count=1),
                 encoding="utf-8")
    with pytest.raises(ftc.Refused, match="already 3"):
        ftc.plan(_paste(), 9, 1, 0, **tree)
    assert ftc.plan(_paste(), 9, 1, 0, overwrite=True, **tree)


# --- doc and CLI -------------------------------------------------------------------------------------

def test_doc_tables_filled_and_cli_writes_only_with_write(tree, tmp_path, monkeypatch, capsys):
    _set_nulls(tree["configs"])
    changes, report = ftc.plan(_paste(), 9, 1, 0, **tree)
    doc = changes[tree["doc"]]
    assert "| D-FINE-S | 512 | 640 | |" not in doc and "| D-FINE-S | 512 | 640 | 3.0 | 5.0 |  | 9.0 | 13.5 | OOM |" in doc
    assert "### 5.4 Training schedule" in doc and "template, awaiting" not in doc
    assert "| D-FINE-M | 512 | 640 | |" in doc                             # rows of models not run stay empty

    real = [*ftc.CONFIGS.values(), ftc.DOC]
    real_before = {p: p.read_text(encoding="utf-8") for p in real}
    monkeypatch.setattr(ftc, "CONFIGS", tree["configs"])
    monkeypatch.setattr(ftc, "DOC", tree["doc"])
    monkeypatch.setattr(ftc, "ARTIFACT_DIR", tree["artifact_dir"])
    j = tmp_path / "paste.json"
    j.write_text(json.dumps(ftc.paste_block(GOOD, "s")))
    before = {p: p.read_text(encoding="utf-8") for p in [*tree["configs"].values(), tree["doc"]]}
    assert ftc.main(["--json", str(j), "--hours-per-session", "9"]) == 0
    out = capsys.readouterr().out
    assert "+  batch_size: 8" in out and "Diff only" in out
    assert all(p.read_text(encoding="utf-8") == t for p, t in before.items())   # nothing saved
    assert ftc.main(["--json", str(j), "--hours-per-session", "9", "--write"]) == 0
    assert yaml.safe_load(tree["configs"][640].read_text())["schedule"]["batch_size"] == 8
    assert list(tree["artifact_dir"].glob("t4_paste_2026-10-11.json"))
    assert ftc.main(["--json", str(tmp_path / "missing.json"), "--hours-per-session", "9"]) == 2
    assert {p: p.read_text(encoding="utf-8") for p in real} == real_before    # the real repo files were not touched
    assert not list((REPO / "artifacts/p3/feasibility").glob("t4_paste_2026-10-11.json"))


def test_cli_requires_the_time_budget(tmp_path):
    with pytest.raises(SystemExit):
        ftc.main(["--json", str(tmp_path / "x.json")])                     # no --hours-per-session: no guess
