"""P3-02: environment detection and Drive sync plumbing (no real Drive, no rclone needed)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
pytest.importorskip("torch")

from p3_ops.cloud import sync  # noqa: E402
from p3_ops.cloud.sync import CheckpointStore, detect_environment  # noqa: E402

ENV_VARS = ["TB_CKPT_ENV", "TB_CKPT_ROOT", "TB_RCLONE_REMOTE", "TB_CKPT_SEED_DIR",
            "COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(sync, "KAGGLE_WORKING", tmp_path / "no_kaggle_here")


@pytest.mark.parametrize("var, env", [("COLAB_RELEASE_TAG", "colab"), ("KAGGLE_KERNEL_RUN_TYPE", "kaggle")])
def test_detect_environment(monkeypatch, var, env):
    assert detect_environment() == "local"
    monkeypatch.setenv(var, "x")
    assert detect_environment() == env
    monkeypatch.setenv("TB_CKPT_ENV", "local")
    assert detect_environment() == "local"


def test_local_store_uses_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TB_CKPT_ROOT", str(tmp_path))
    s = CheckpointStore.for_run("run1")
    assert s.env == "local" and s.dir == tmp_path / "run1" and s.dir.is_dir() and s.remote is None
    s.push()
    s.pull()  # both no-ops


def test_colab_store_on_mounted_drive(monkeypatch, tmp_path):
    (tmp_path / "MyDrive").mkdir()
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    monkeypatch.setattr(sync, "COLAB_MOUNT", tmp_path)
    monkeypatch.setattr(sync, "mount_colab_drive", lambda: tmp_path / "MyDrive")
    s = CheckpointStore.for_run("p2_effb3")
    assert s.env == "colab" and s.dir == tmp_path / "MyDrive" / "tb_checkpoints" / "p2_effb3"


def test_kaggle_without_rclone_warns_and_stays_local(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setenv("TB_CKPT_ROOT", str(tmp_path))
    monkeypatch.setattr(sync, "setup_rclone_from_kaggle_secret", lambda: False)
    monkeypatch.setattr(sync, "_rclone_conf_has", lambda remote: False)
    s = CheckpointStore.for_run("p3_dfine")
    assert s.env == "kaggle" and s.remote is None
    assert "NOT copied to Drive" in capsys.readouterr().out


def test_kaggle_with_rclone_pushes_and_pulls(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("TB_CKPT_ROOT", str(tmp_path))
    monkeypatch.setenv("TB_RCLONE_REMOTE", "gdrive:tb_checkpoints/")
    monkeypatch.setattr(sync.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(sync.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    s = CheckpointStore.for_run("p3_dfine")
    assert s.remote == "gdrive:tb_checkpoints/p3_dfine"
    s.pull()
    s.push()
    assert calls[0] == ["rclone", "copy", "gdrive:tb_checkpoints/p3_dfine", str(s.dir)]
    assert calls[1][:4] == ["rclone", "sync", str(s.dir), "gdrive:tb_checkpoints/p3_dfine"]


def test_kaggle_uses_existing_rclone_conf(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setenv("TB_CKPT_ROOT", str(tmp_path / "ck"))
    monkeypatch.setattr(sync.Path, "home", lambda: tmp_path)
    conf = tmp_path / ".config" / "rclone" / "rclone.conf"
    conf.parent.mkdir(parents=True)
    conf.write_text("[gdrive]\ntype = drive\n")
    monkeypatch.setattr(sync, "setup_rclone_from_kaggle_secret", lambda: False)
    monkeypatch.setattr(sync.shutil, "which", lambda name: "/usr/bin/rclone")
    assert CheckpointStore.for_run("r").remote == "gdrive:tb_checkpoints/r"


def test_rclone_failure_never_raises(monkeypatch, capsys):
    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr="quota exceeded")
    monkeypatch.setattr(sync.subprocess, "run", boom)
    CheckpointStore(dir=Path("."), env="kaggle", remote="gdrive:x").push()
    assert "rclone sync failed" in capsys.readouterr().out


def test_pull_from_seed_dir(monkeypatch, tmp_path):
    seed = tmp_path / "prev_output"
    seed.mkdir()
    (seed / "ckpt_epoch0003.pt").write_bytes(b"x")
    monkeypatch.setenv("TB_CKPT_SEED_DIR", str(seed))
    s = CheckpointStore(dir=tmp_path / "work", env="kaggle")
    s.dir.mkdir()
    s.pull()
    assert (s.dir / "ckpt_epoch0003.pt").read_bytes() == b"x"


def test_demo_cli_with_explicit_dir(tmp_path, capsys):
    from p3_ops.cloud import demo
    assert demo.main(["--ckpt-dir", str(tmp_path), "--epochs", "4", "--kill-at-epoch", "2"]) == 1
    assert "SIMULATED KILL" in capsys.readouterr().out
    assert demo.main(["--ckpt-dir", str(tmp_path), "--epochs", "4"]) == 0
    assert "resumed from ckpt_epoch0001.pt" in capsys.readouterr().out
