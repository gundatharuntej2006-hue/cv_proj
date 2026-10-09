"""P3-02: checkpoint helper: kill/resume, atomic writes, retention, RNG restore (CPU only)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from p3_ops.cloud import checkpoint as ck  # noqa: E402
from p3_ops.cloud import demo  # noqa: E402
from p3_ops.cloud import (  # noqa: E402
    find_latest_checkpoint, list_checkpoints, read_best, resume_if_exists, save_checkpoint)

EPOCHS = 6


def _epochs_on_disk(d):
    return sorted(e for e, _ in list_checkpoints(d))


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    return demo.train(tmp_path_factory.mktemp("full"), epochs=EPOCHS, seed=0)


@pytest.mark.parametrize("kill_epoch", [0, 1, 3, EPOCHS - 1])
def test_kill_mid_epoch_then_resume_loses_at_most_one_epoch(tmp_path, uninterrupted, kill_epoch):
    with pytest.raises(demo.SimulatedKill):
        demo.train(tmp_path, epochs=EPOCHS, seed=0, kill_at=(kill_epoch, 3))
    # every epoch before the killed one is safely on disk
    latest = find_latest_checkpoint(tmp_path)
    if kill_epoch == 0:
        assert latest is None
    else:
        assert latest.name == ck.checkpoint_name(kill_epoch - 1)

    resumed = demo.train(tmp_path, epochs=EPOCHS, seed=0)
    assert resumed.start_epoch == kill_epoch            # continues from the killed epoch
    assert resumed.epochs_completed == list(range(kill_epoch, EPOCHS))
    assert kill_epoch - resumed.start_epoch <= 1        # lost at most the interrupted epoch
    # RNG, optimiser momentum and scheduler restored: identical to never being killed
    for k, v in uninterrupted.model_state.items():
        assert torch.equal(resumed.model_state[k], v), k
    assert resumed.val_losses == uninterrupted.val_losses[kill_epoch:]


def test_identity_check_has_teeth_without_rng_restore(tmp_path, uninterrupted, monkeypatch):
    # Control: if RNG state were not restored, the resumed run would diverge (dropout, shuffling).
    with pytest.raises(demo.SimulatedKill):
        demo.train(tmp_path, epochs=EPOCHS, seed=0, kill_at=(3, 3))
    real = demo.resume_if_exists
    monkeypatch.setattr(demo, "resume_if_exists", lambda *a, **kw: real(*a, **kw, restore_rng=False))
    resumed = demo.train(tmp_path, epochs=EPOCHS, seed=0)
    assert not all(torch.equal(resumed.model_state[k], v) for k, v in uninterrupted.model_state.items())


def test_two_kills_in_a_row(tmp_path, uninterrupted):
    for epoch in (2, 4):
        with pytest.raises(demo.SimulatedKill):
            demo.train(tmp_path, epochs=EPOCHS, seed=0, kill_at=(epoch, 1))
    final = demo.train(tmp_path, epochs=EPOCHS, seed=0)
    assert final.start_epoch == 4
    for k, v in uninterrupted.model_state.items():
        assert torch.equal(final.model_state[k], v)


def test_partially_written_files_are_ignored(tmp_path):
    demo.train(tmp_path, epochs=3, seed=0)                      # epochs 0..2 saved
    # a temp file left by a kill during torch.save, and a truncated "newest" checkpoint
    (tmp_path / ".ckpt_epoch0003.pt.abc123.tmp").write_bytes(b"\x80\x02garbage")
    good = (tmp_path / ck.checkpoint_name(2)).read_bytes()
    (tmp_path / ck.checkpoint_name(3)).write_bytes(good[: len(good) // 2])
    (tmp_path / ck.checkpoint_name(4)).write_bytes(b"")          # zero-byte file

    model, opt, sched = demo.build(0)
    with pytest.warns(UserWarning, match="skipping unreadable checkpoint"):
        state = resume_if_exists(tmp_path, model, opt, sched)
    assert state.resumed_from.name == ck.checkpoint_name(2)
    assert state.start_epoch == 3
    assert [p.name for p in state.skipped] == [ck.checkpoint_name(3)]


def test_kill_during_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    demo.train(tmp_path, epochs=2, seed=0)
    real_save = torch.save

    def dying_save(obj, f, *a, **kw):
        f.write(b"half a checkpoint")
        raise KeyboardInterrupt("session killed during write")

    model, opt, sched = demo.build(0)
    monkeypatch.setattr(ck.torch, "save", dying_save)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(model, opt, sched, 2, {}, 0, tmp_path)
    monkeypatch.setattr(ck.torch, "save", real_save)

    assert _epochs_on_disk(tmp_path) == [0, 1]
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))
    assert resume_if_exists(tmp_path, *demo.build(0)).start_epoch == 2


def test_keeps_last_two_plus_best(tmp_path):
    model, opt, sched = demo.build(0)
    val = [5.0, 1.0, 3.0, 4.0, 2.5, 6.0]          # best = epoch 1
    for e, v in enumerate(val):
        save_checkpoint(model, opt, sched, e, {"val_loss": v}, 0, tmp_path, is_best=v == min(val[: e + 1]))
    assert _epochs_on_disk(tmp_path) == [4, 5]
    assert read_best(tmp_path) == {"epoch": 1, "metrics": {"val_loss": 1.0}}
    best = ck.load_checkpoint(tmp_path / ck.BEST_NAME)
    assert best["epoch"] == 1 and best["seed"] == 0


def test_checkpoint_contents_load_with_weights_only(tmp_path):
    model, opt, sched = demo.build(0)
    p = save_checkpoint(model, opt, sched, 0, {"val_loss": 0.5, "ap50": 0.3}, 42, tmp_path,
                        extra={"note": "x"})
    c = torch.load(p, map_location="cpu", weights_only=True)
    assert c["seed"] == 42 and c["epoch"] == 0 and c["metrics"]["ap50"] == 0.3
    assert set(c["rng"]) == {"python", "numpy", "torch", "cuda"}
    assert c["extra"] == {"note": "x"} and c["format_version"] == ck.FORMAT_VERSION


def test_rng_state_round_trip():
    state = ck.get_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    ck.set_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == a


def test_optimizer_scheduler_and_extra_restored(tmp_path):
    model, opt, sched = demo.build(0)
    for _ in range(3):
        opt.step()
        sched.step()
    counter = torch.nn.Module()
    counter.register_buffer("n", torch.tensor(7))
    save_checkpoint(model, opt, sched, 4, {}, 0, tmp_path, extra={"counter": counter})

    model2, opt2, sched2 = demo.build(0)
    counter2 = torch.nn.Module()
    counter2.register_buffer("n", torch.tensor(0))
    state = resume_if_exists(tmp_path, model2, opt2, sched2, extra={"counter": counter2})
    assert state.start_epoch == 5 and state.last_epoch == 4 and state.seed == 0
    assert sched2.last_epoch == sched.last_epoch
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
    assert int(counter2.n) == 7


def test_fresh_start_when_empty_or_missing(tmp_path):
    state = resume_if_exists(tmp_path / "does_not_exist", *demo.build(0))
    assert state.start_epoch == 0 and state.resumed_from is None and state.best is None


def test_seed_mismatch_is_refused(tmp_path):
    demo.train(tmp_path, epochs=1, seed=0)
    with pytest.raises(ValueError, match="seed"):
        demo.train(tmp_path, epochs=2, seed=1)
