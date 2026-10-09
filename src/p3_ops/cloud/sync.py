"""P3-02: where checkpoints live, chosen from the environment, and how they reach Google Drive.

    colab   Google Drive is mounted at /content/drive; checkpoints are written straight to
            MyDrive/<root>/<run> (atomic rename on the Drive mount), so push/pull are no-ops.
    kaggle  Kaggle cannot mount Drive. Checkpoints are written to /kaggle/working/checkpoints/<run>
            and, when an rclone remote is configured (Kaggle secret RCLONE_CONF, see the runbook),
            mirrored to Drive after every save with ``rclone sync`` and pulled back at start.
            Without rclone they survive only inside the session, or in the notebook's output
            when the version is saved ("Save & Run All"); attach that output as an input and set
            TB_CKPT_SEED_DIR to resume from it.
    local   ~/tb_checkpoints/<run> (or TB_CKPT_ROOT/<run>); no sync.

Environment overrides: TB_CKPT_ENV (colab|kaggle|local), TB_CKPT_ROOT, TB_RCLONE_REMOTE
(e.g. "gdrive:tb_checkpoints"), TB_CKPT_SEED_DIR (read-only directory to copy checkpoints from).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DRIVE_ROOT_NAME = "tb_checkpoints"
COLAB_MOUNT = Path("/content/drive")
KAGGLE_WORKING = Path("/kaggle/working")
RCLONE_SECRET = "RCLONE_CONF"


def detect_environment() -> str:
    forced = os.environ.get("TB_CKPT_ENV")
    if forced:
        return forced
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_WORKING.is_dir():
        return "kaggle"
    return "local"


def mount_colab_drive(mount_point: Path = COLAB_MOUNT) -> Path:
    """Mount Google Drive on Colab (asks for consent the first time) and return MyDrive."""
    if not (mount_point / "MyDrive").is_dir():
        from google.colab import drive  # only importable on Colab
        drive.mount(str(mount_point))
    return mount_point / "MyDrive"


def setup_rclone_from_kaggle_secret(secret_name: str = RCLONE_SECRET) -> bool:
    """Write rclone.conf from a Kaggle secret. Returns False when not on Kaggle or unset."""
    try:
        from kaggle_secrets import UserSecretsClient
        conf = UserSecretsClient().get_secret(secret_name)
    except Exception:
        return False
    if not conf:
        return False
    path = Path.home() / ".config" / "rclone" / "rclone.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(conf)
    path.chmod(0o600)
    return True


@dataclass
class CheckpointStore:
    """A local checkpoint directory plus an optional rclone remote it mirrors to."""
    dir: Path
    env: str
    remote: str | None = None   # rclone path, e.g. "gdrive:tb_checkpoints/p3_dfine_s_512"

    @classmethod
    def for_run(cls, run_name: str, env: str | None = None) -> "CheckpointStore":
        env = env or detect_environment()
        root = os.environ.get("TB_CKPT_ROOT")
        remote = None
        if env == "colab":
            base = Path(root) if root else mount_colab_drive() / DRIVE_ROOT_NAME
        elif env == "kaggle":
            base = Path(root) if root else KAGGLE_WORKING / "checkpoints"
            rbase = os.environ.get("TB_RCLONE_REMOTE")
            if rbase is None and (setup_rclone_from_kaggle_secret() or _rclone_conf_has("gdrive")):
                rbase = f"gdrive:{DRIVE_ROOT_NAME}"
            if rbase and shutil.which("rclone"):
                remote = f"{rbase.rstrip('/')}/{run_name}"
            else:
                print("[p3_ops.cloud] WARNING: no rclone remote; checkpoints are NOT copied to Drive. "
                      "See docs/p3_kaggle_runbook.md section 5.")
        else:
            base = Path(root) if root else Path.home() / DRIVE_ROOT_NAME
        store = cls(dir=base / run_name, env=env, remote=remote)
        store.dir.mkdir(parents=True, exist_ok=True)
        return store

    def pull(self) -> None:
        """Bring existing checkpoints into ``dir`` before resuming (remote and/or seed dir)."""
        seed = os.environ.get("TB_CKPT_SEED_DIR")
        if seed and Path(seed).is_dir():
            for p in Path(seed).glob("*"):
                if p.is_file() and not (self.dir / p.name).exists():
                    shutil.copy2(p, self.dir / p.name)
        if self.remote:
            _rclone("copy", self.remote, str(self.dir))

    def push(self) -> None:
        """Mirror ``dir`` to the remote (deletions included, so pruning is mirrored). No-op without one."""
        if self.remote:
            _rclone("sync", str(self.dir), self.remote, "--exclude", ".*.tmp")


def _rclone_conf_has(remote: str) -> bool:
    conf = Path.home() / ".config" / "rclone" / "rclone.conf"
    try:
        return f"[{remote}]" in conf.read_text()
    except OSError:
        return False


def _rclone(*args: str) -> None:
    try:
        subprocess.run(["rclone", *args], check=True, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as e:
        # A failed upload must never kill training; the local copy is still there.
        detail = getattr(e, "stderr", "") or e
        print(f"[p3_ops.cloud] WARNING: rclone {args[0]} failed: {detail}")
