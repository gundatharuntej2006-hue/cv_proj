"""P3-06: D16 selection logic on synthetic run folders (no training needed)."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_det import select_size as ss  # noqa: E402
from p3_det.train.config import load_run_config, schedule_hash  # noqa: E402

warnings.filterwarnings("ignore", message="D21 not locked")

GT = {f"tb{i:03d}": {"boxes": [[100 + i, 100, 160 + i, 170]], "subtype": "active"} for i in range(30)}
GT.update({f"h{i:03d}": {"boxes": [], "subtype": None} for i in range(30)})


def _dets(quality: float, seed: int) -> list[dict]:
    """quality = share of TB images that get an exact hit (score 0.9); everyone gets clutter."""
    r = np.random.default_rng(seed)
    out = []
    for iid, g in GT.items():
        boxes = [[300, 300, 340, 340, float(r.uniform(0.01, 0.5))]]
        if g["boxes"] and r.random() < quality:
            boxes.append([*g["boxes"][0], 0.9])
        out.append({"image_id": iid, "boxes": boxes})
    return out


def make_run(root: Path, size: int, quality: float, *, epochs=3, seed=0, prep=None, sched="sha256:x",
             finished=True) -> Path:
    d = root / f"dfine_s_{size}"
    d.mkdir(parents=True)
    prep = prep or {"margin": 0.1, "clahe": False, "provisional": False}
    (d / "run_meta.json").write_text(json.dumps({
        "run_name": f"dfine_s_{size}", "run_dir": d.name, "input_size": size, "seed": seed,
        "schedule_hash": sched, "preprocessing": prep, "config": {"schedule": {"epochs": epochs}}}))
    n_logged = epochs if finished else epochs - 1
    (d / "train_log.csv").write_text("epoch,train_loss,dev_ap50\n" + "".join(f"{e},1.0,0.1\n" for e in range(n_logged)))
    (d / "dev_ground_truth.json").write_text(json.dumps({"images": GT}))
    (d / "dev_detections_last.json").write_text(json.dumps({
        "coords": "full512", "epoch": n_logged - 1, "images": _dets(quality, size)}))
    return d


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "decisions" / "p3_det_input_size.json"
    ss.write_criterion(path, today="2026-10-08")
    return {"path": path, "ev": tmp_path / "ev", "root": tmp_path / "runs"}


def run_compare(env, r512, r640, **kw):
    return ss.compare(r512, r640, path=env["path"], evidence_dir=env["ev"], allow_uncommitted=True,
                      today="2026-11-01", **kw)


def test_criterion_written_before_evidence(env):
    doc = json.loads(env["path"].read_text())
    assert doc["criterion_preregistered"] == ss.CRITERION and doc["preregistered_at"] == "2026-10-08"
    assert doc["value"] == {} and doc["evidence"] == [] and doc["locked"] is False
    assert doc["data_used"] == ["train", "development"] and doc["reviewed_by"] == "p4"
    ss.write_criterion(env["path"])                         # idempotent
    doc["criterion_preregistered"] = "something else"
    env["path"].write_text(json.dumps(doc))
    with pytest.raises(ss.Refused, match="never rewritten"):
        ss.write_criterion(env["path"])


def test_valid_against_c16_standin(env):
    pytest.importorskip("jsonschema")
    sys.path.insert(0, str(REPO / "tests" / "p3_decisions"))
    from c16_standin import validate_decision
    validate_decision(json.loads(env["path"].read_text()))


def test_compare_picks_better_size_with_cis(env):
    doc, code = run_compare(env, make_run(env["root"], 512, 0.4), make_run(env["root"], 640, 0.8))
    v = doc["value"]
    assert code == 0 and v["input_size"] == 640
    assert v["ap50_640"]["value"] > v["ap50_512"]["value"]
    for k in ("ap50_512", "ap50_640"):
        r = v[k]
        assert r["ci_low"] <= r["value"] <= r["ci_high"]
        assert (r["n_images"], r["n_tb_images"], r["n_non_tb_images"], r["n_boxes"]) == (60, 30, 30, 30)
    assert v["ap50_640_minus_512"]["value"] > 0
    assert doc["locked"] is False and v["contingency"]["triggered"] is False
    assert (env["ev"] / "d16_comparison.json").exists()


def test_tie_keeps_512(env):
    r512 = make_run(env["root"], 512, 0.5)
    r640 = make_run(env["root"], 640, 0.5)
    (r640 / "dev_detections_last.json").write_text((r512 / "dev_detections_last.json").read_text())
    doc, _ = run_compare(env, r512, r640)
    assert doc["value"]["input_size"] == 512


def test_contingency_stops(env):
    doc, code = run_compare(env, make_run(env["root"], 512, 0.0), make_run(env["root"], 640, 0.0))
    assert code == 3 and doc["value"]["contingency"]["triggered"] and doc["locked"] is False
    doc2, code2 = ss.compare(env["root"] / "dfine_s_512", env["root"] / "dfine_s_640", path=env["path"],
                             evidence_dir=env["ev"], allow_uncommitted=True, lock=True)
    assert code2 == 3 and doc2["locked"] is False            # --lock never locks under contingency


@pytest.mark.parametrize("kw512, kw640, match", [
    ({"seed": 0}, {"seed": 1}, "seed"),
    ({"sched": "sha256:a"}, {"sched": "sha256:b"}, "schedule_hash"),
    ({}, {"prep": {"margin": 0.05, "clahe": False, "provisional": False}}, "preprocessing"),
    ({}, {"finished": False}, "not finished"),
])
def test_refuses_incomparable_runs(env, kw512, kw640, match):
    with pytest.raises(ss.Refused, match=match):
        run_compare(env, make_run(env["root"], 512, 0.5, **kw512), make_run(env["root"], 640, 0.5, **kw640))


def test_refuses_lock_on_provisional_and_without_commit(env):
    prov = {"margin": 0.1, "clahe": False, "provisional": True}
    r512 = make_run(env["root"], 512, 0.5, prep=prov)
    r640 = make_run(env["root"], 640, 0.6, prep=prov)
    with pytest.raises(ss.Refused, match="PROVISIONAL"):
        run_compare(env, r512, r640, lock=True)
    doc, code = run_compare(env, r512, r640)                 # dry run is allowed, stays unlocked
    assert code == 0 and doc["locked"] is False and doc["value"]["preprocessing"]["provisional"]
    with pytest.raises(ss.Refused, match="not committed"):
        ss.compare(r512, r640, path=env["path"], evidence_dir=env["ev"])


def test_refuses_changed_criterion(env):
    doc = json.loads(env["path"].read_text())
    doc["criterion_preregistered"] += " (edited later)"
    env["path"].write_text(json.dumps(doc))
    with pytest.raises(ss.Refused, match="differs"):
        run_compare(env, make_run(env["root"], 512, 0.5), make_run(env["root"], 640, 0.5))


def test_cli_lock_with_allow_uncommitted_refused(capsys):
    assert ss.main(["compare", "--run512", "a", "--run640", "b", "--lock", "--allow-uncommitted"]) == 2


# --- configs ---------------------------------------------------------------------------------------

def test_512_and_640_configs_identical_except_size():
    c512, c640 = load_run_config("dfine_512.yaml"), load_run_config("dfine_640.yaml")
    assert (c512.input_size, c640.input_size) == (512, 640)
    assert c512.schedule_hash == c640.schedule_hash
    assert {k for k in c512.raw if c512.raw[k] != c640.raw[k]} == {"run_name", "input_size"}
    assert c512.seed == c640.seed and c512.model["init"] == "coco"
    assert c512.data["augment"]["hflip_p"] == 0.0


def test_schedule_hash_detects_changes():
    raw = load_run_config("dfine_512.yaml").raw
    changed = json.loads(json.dumps(raw))
    changed["schedule"]["batch_size"] = 4
    assert schedule_hash(raw) != schedule_hash(changed)


def test_training_refused_while_feasibility_values_are_null():
    from p3_det.train.config import require_trainable
    cfg = load_run_config("dfine_512.yaml")
    if cfg.schedule["epochs"] is not None and cfg.schedule["batch_size"] is not None:
        pytest.skip("feasibility values already filled in")
    with pytest.raises(ValueError, match="p3_dfine_feasibility.md"):
        require_trainable(cfg)
