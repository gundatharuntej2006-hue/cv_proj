"""P3-07: D17 (checkpoint) and D18 (confidence threshold): known answers, flows, git order, schema."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("jsonschema")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "p3_decisions"))

from c16_standin import validate_decision  # noqa: E402
from p3_det import decisions as dec  # noqa: E402
from p3_det import lock_checkpoint as d17  # noqa: E402
from p3_det import lock_threshold as d18  # noqa: E402

warnings.filterwarnings("ignore", message="D21 not locked")


# --- D18 rule: known answers -----------------------------------------------------------------------

def test_threshold_known_answer_strictly_above_the_next_image():
    # 100 non-TB images: 5 may carry a box (5%). Top maxima: five at 0.9, one at 0.5, rest 0.2.
    # t = 0.500 counts six images (0.5 >= 0.5) -> too many; t = 0.501 counts five -> allowed.
    maxima = [0.9] * 5 + [0.5] + [0.2] * 94
    assert d18.lowest_threshold(maxima) == (0.501, 5)


def test_threshold_known_answer_on_grid_value():
    # the sixth-highest maximum is 0.4999 -> t = 0.500 already excludes it
    maxima = [0.9] * 5 + [0.4999] + [0.0] * 94
    assert d18.lowest_threshold(maxima) == (0.5, 5)


def test_threshold_small_n_rounds_down():
    # 30 images: 5% of 30 = 1.5 -> at most 1 image may have a box
    maxima = [0.8, 0.7, 0.3] + [0.0] * 27
    assert d18.lowest_threshold(maxima) == (0.701, 1)


def test_threshold_floor_when_negatives_are_clean():
    assert d18.lowest_threshold([0.0] * 50) == (0.01, 0)


def test_threshold_is_lowest_property():
    r = np.random.default_rng(0)
    for _ in range(200):
        n = int(r.integers(20, 300))
        m = r.uniform(0, 1, n) * (r.random(n) < 0.7)          # ~30% of non-TB images without any box
        t, n = d18.lowest_threshold(m)
        assert n / len(m) <= 0.05 and n == int((m >= t).sum())
        if t > d18.FLOOR:
            assert (m >= round(t - d18.STEP, 3)).sum() / len(m) > 0.05   # one grid step lower breaks it


def test_threshold_refuses_without_non_tb():
    with pytest.raises(dec.Refused):
        d18.lowest_threshold([])


# --- synthetic run folder ----------------------------------------------------------------------------

def make_run(root: Path, *, ap_by_epoch=(0.10, 0.30, 0.30, 0.20), best_epoch=1, input_size=512,
             provisional=False) -> Path:
    rd = root / f"dfine_s_{input_size}_m0.10_clahe0"
    rd.mkdir(parents=True)
    gt = {f"tb{i:03d}": {"boxes": [[100, 100, 150, 160]], "subtype": "active"} for i in range(20)}
    gt.update({f"h{i:03d}": {"boxes": [], "subtype": None} for i in range(100)})
    (rd / "dev_ground_truth.json").write_text(json.dumps({"images": gt}))
    images = []
    for i in range(20):                                     # TB: a hit at 0.8 on 12 of 20 images
        images.append({"image_id": f"tb{i:03d}", "boxes": [[100, 100, 150, 160, 0.8]] if i < 12 else []})
    maxima = [0.9] * 5 + [0.5] + [0.2] * 94                 # D18 answer 0.501 (see known-answer test)
    for i, m in enumerate(maxima):
        images.append({"image_id": f"h{i:03d}", "boxes": [[300, 300, 340, 340, m]]})
    det = {"coords": "full512", "epoch": best_epoch, "images": images}
    (rd / "dev_detections_best.json").write_text(json.dumps(det))
    from p3_det.metrics import ImageGT, ap50, detections_from_c10
    ap_true = ap50(detections_from_c10(det), {k: ImageGT(v["boxes"], v["subtype"]) for k, v in gt.items()},
                   ci=False).value
    ap = list(ap_by_epoch)
    ap[best_epoch] = ap_true                                 # log must agree with the recomputed AP50
    ap = [min(a, ap_true) if e != best_epoch else a for e, a in enumerate(ap)]
    (rd / "train_log.csv").write_text("epoch,train_loss,dev_ap50\n" + "".join(
        f"{e},1.0,{a:.6f}\n" for e, a in enumerate(ap)))
    (rd / "best.json").write_text(json.dumps({"epoch": best_epoch, "metrics": {"dev_ap50": ap_true}}))
    ckpt = rd / "ckpt_best.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    (rd / "checkpoints_sha256.txt").write_text(f"{d17.sha256_file(ckpt)}  ckpt_best.pt\n")
    prep = {"margin": 0.1, "clahe": False, "provisional": provisional}
    (rd / "run_meta.json").write_text(json.dumps({
        "run_dir": rd.name, "input_size": input_size, "seed": 0, "schedule_hash": "sha256:x",
        "preprocessing": prep, "config": {"schedule": {"epochs": len(ap)}}}))
    return rd


@pytest.fixture
def env(tmp_path):
    ddir = tmp_path / "decisions"
    d16 = ddir / "p3_det_input_size.json"
    dec.save(d16, {"locked": True, "value": {"input_size": 512,
                                             "preprocessing": {"margin": 0.1, "clahe": False, "provisional": False}}})
    return {"tmp": tmp_path, "d16": d16, "d17": ddir / "p3_det_checkpoint.json",
            "d18": ddir / "p3_det_conf_threshold.json", "ev": tmp_path / "ev", "run": make_run(tmp_path / "runs")}


def _d17(env, **kw):
    return d17.result(env["run"], path=env["d17"], d16_path=env["d16"], evidence_dir=env["ev"], **kw)


def _d18(env, **kw):
    return d18.result(env["run"], path=env["d18"], d17_path=env["d17"], **kw)


# --- D17 ---------------------------------------------------------------------------------------------

def test_d17_best_epoch_earliest_tie_and_sha(env):
    dec.write_criterion(env["d17"], d17.criterion_template("2026-10-08"))
    doc = _d17(env, dry_run=True, checkpoint=env["run"] / "ckpt_best.pt")
    v = doc["value"]
    assert v["epoch"] == 1 and v["training_length_epochs"] == 2
    assert v["checkpoint"]["sha256"] == d17.sha256_file(env["run"] / "ckpt_best.pt")
    assert v["checkpoint"]["verified_from_file"] is True
    assert v["dev_ap50"]["ci_low"] <= v["dev_ap50"]["value"] <= v["dev_ap50"]["ci_high"]
    assert v["dev_ap50"]["n_images"] == 120
    ev = json.loads((env["ev"] / "d17_evidence.json").read_text())
    assert ev["checkpoint"]["sha256"] == v["checkpoint"]["sha256"]              # SHA in the evidence too
    assert doc["data_used"] == ["train", "development"] and doc["locked"] is False
    validate_decision(doc)


def test_d17_earliest_epoch_on_ties():
    log = [{"epoch": "0", "dev_ap50": "0.2"}, {"epoch": "1", "dev_ap50": "0.4"}, {"epoch": "2", "dev_ap50": "0.4"}]
    assert d17.best_epoch(log) == (1, 0.4)


def test_d17_refusals(env):
    dec.write_criterion(env["d17"], d17.criterion_template())
    with pytest.raises(dec.Refused, match="not committed"):
        _d17(env)
    bad = env["tmp"] / "other.pt"
    bad.write_bytes(b"different")
    with pytest.raises(dec.Refused, match="does not match"):
        _d17(env, dry_run=True, checkpoint=bad)
    (env["run"] / "best.json").write_text(json.dumps({"epoch": 3}))
    with pytest.raises(dec.Refused, match="best.json"):
        _d17(env, dry_run=True)


def test_d17_provisional_cannot_lock(tmp_path):
    rd = make_run(tmp_path / "runs", provisional=True)
    path = tmp_path / "d17.json"
    dec.write_criterion(path, d17.criterion_template())
    with pytest.raises(dec.Refused, match="PROVISIONAL"):
        d17.result(rd, path=path, d16_path=tmp_path / "none.json", evidence_dir=tmp_path / "ev",
                   dry_run=True, lock=True)


# --- D18 ---------------------------------------------------------------------------------------------

def test_d18_dry_run_value_and_reports(env):
    dec.write_criterion(env["d18"], d18.criterion_template("2026-10-08"))
    doc = _d18(env, dry_run=True)
    v = doc["value"]
    assert v["conf_threshold"] == 0.501
    assert v["non_tb_share_with_box"]["n_with_box"] == 5 and v["non_tb_share_with_box"]["n_non_tb_images"] == 100
    assert v["tb_image_level_lesion_sensitivity"]["value"] == pytest.approx(12 / 20)
    assert doc["guide_ack"] is False and doc["locked"] is False and doc["data_used"] == ["development"]
    assert d18.RULE in doc["criterion_preregistered"]
    validate_decision(doc)


def test_d18_guide_ack_required_and_tied_to_value(env):
    dec.write_criterion(env["d18"], d18.criterion_template())
    with pytest.raises(dec.Refused, match="compute the threshold first"):
        d18.ack_guide("2026-11-02", "approved in meeting", path=env["d18"])
    _d18(env, dry_run=True)
    with pytest.raises(dec.Refused, match="cannot lock"):
        _d18(env, dry_run=True, lock=True)                    # dry runs never lock
    doc = d18.ack_guide("2026-11-02", "approved in meeting", path=env["d18"])
    assert doc["guide_ack"] is True and doc["value"]["guide_ack_record"]["conf_threshold_acknowledged"] == 0.501
    assert _d18(env, dry_run=True)["value"]["guide_ack_record"]["date"] == "2026-11-02"   # survives re-run


# --- git order: rule committed before result ---------------------------------------------------------

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    r = tmp_path / "repo"
    (r / "decisions").mkdir(parents=True)
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    return r


def test_full_flow_with_git_order(repo, tmp_path):
    run = make_run(tmp_path / "runs")
    d16p, d17p, d18p = (repo / "decisions" / n for n in
                        ("p3_det_input_size.json", "p3_det_checkpoint.json", "p3_det_conf_threshold.json"))
    dec.save(d16p, {"locked": True, "value": {"input_size": 512, "preprocessing":
                                              {"margin": 0.1, "clahe": False, "provisional": False}}})
    dec.write_criterion(d17p, d17.criterion_template())
    dec.write_criterion(d18p, d18.criterion_template())
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "P3-07: pre-register D17 and D18 rules")
    assert dec.verify_order(d18p)[0]                                      # "OK so far"

    doc17 = d17.result(run, path=d17p, d16_path=d16p, evidence_dir=tmp_path / "ev", lock=True,
                       checkpoint=run / "ckpt_best.pt")
    assert doc17["locked"] and doc17["value"]["criterion_commit"] and len(doc17["git_sha"]) == 40
    _git(repo, "commit", "-qam", "P3-07: lock D17")

    doc18 = d18.result(run, path=d18p, d17_path=d17p)
    assert doc18["value"]["detector"]["checkpoint_sha256"] == doc17["value"]["checkpoint"]["sha256"]
    _git(repo, "commit", "-qam", "P3-07: D18 threshold result")
    with pytest.raises(dec.Refused, match="guide's acknowledgement"):
        d18.result(run, path=d18p, d17_path=d17p, lock=True)              # no guide_ack yet
    d18.ack_guide("2026-11-02", "guide approved by email", path=d18p)
    doc18 = d18.result(run, path=d18p, d17_path=d17p, lock=True)
    assert doc18["locked"] and doc18["guide_ack"]
    _git(repo, "commit", "-qam", "P3-07: lock D18 after guide ack")

    ok, report = dec.verify_order(d18p)
    assert ok and "precedes" in report
    lines = report.splitlines()
    assert lines[0].split()[2] == "rule" and "result" not in lines[0]        # first commit: rule only
    validate_decision(doc17)
    validate_decision(doc18)


def test_verify_order_fails_when_rule_and_result_committed_together(repo, tmp_path):
    run = make_run(tmp_path / "runs")
    d18p = repo / "decisions" / "p3_det_conf_threshold.json"
    dec.write_criterion(d18p, d18.criterion_template())
    d18.result(run, path=d18p, d17_path=tmp_path / "none.json", dry_run=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "everything at once")
    ok, report = dec.verify_order(d18p)
    assert not ok and "FAIL" in report
    assert dec.criterion_commit(d18p, d18.CRITERION) is None


def test_verify_order_without_git(tmp_path):
    p = tmp_path / "x.json"
    dec.save(p, {"criterion_preregistered": "r", "value": {}})
    ok, report = dec.verify_order(p)
    assert not ok and "no git history" in report
