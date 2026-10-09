"""P3-02: crash-safe checkpoint-to-Drive helper for Kaggle/Colab training (used by P2-12, P2-17, P3-06).

    from p3_ops.cloud import CheckpointStore, resume_if_exists, save_checkpoint

    store = CheckpointStore.for_run("p3_dfine_s_512")   # picks Colab Drive / Kaggle+rclone / local
    store.pull()
    state = resume_if_exists(store.dir, model, optimizer, scheduler)
    for epoch in range(state.start_epoch, epochs):
        ...
        save_checkpoint(model, optimizer, scheduler, epoch, metrics, seed, store.dir, is_best=...)
        store.push()
"""
from .checkpoint import (
    ResumeState,
    find_latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    read_best,
    resume_if_exists,
    save_checkpoint,
)
from .sync import CheckpointStore, detect_environment, mount_colab_drive, setup_rclone_from_kaggle_secret

__all__ = [
    "CheckpointStore", "ResumeState", "detect_environment", "find_latest_checkpoint",
    "list_checkpoints", "load_checkpoint", "mount_colab_drive", "prune_checkpoints", "read_best",
    "resume_if_exists", "save_checkpoint", "setup_rclone_from_kaggle_secret",
]
