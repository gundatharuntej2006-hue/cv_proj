# P3-02 · Kaggle / Colab runbook: GPU training that survives a killed session

**Who this is for:** anyone on the team training on a free cloud GPU (P2-12 EfficientNet-B3, P2-17 RAD-DINO fine-tune, P3-06 D-FINE).
**Goal:** if the session dies, you lose **at most one epoch**. The next session picks up where the last one stopped.

The helper is `src/p3_ops/cloud` (owner P3). You don't need to understand it to follow this runbook.

> Kaggle's interface and quotas change from time to time. If a menu name below doesn't match what you see, look for the closest equivalent. When the numbers in section 8 disagree with what Kaggle shows you, trust Kaggle.

---

## 0. One-time setup (about 20 minutes)

### 0.1 Kaggle account
1. Sign up at https://www.kaggle.com.
2. **Verify your phone number** (profile picture → *Settings* → *Phone verification*). Without it, Kaggle won't give you a GPU or internet access in notebooks.

### 0.2 Google Drive access for Kaggle (rclone)
Kaggle can't "mount" Google Drive the way Colab can. We copy checkpoints to Drive with **rclone**, a command-line file copier, after every epoch. You set it up once on your laptop:

1. Install rclone: https://rclone.org/downloads/ (on Windows, unzip it and keep `rclone.exe` somewhere handy).
2. In a terminal, run `rclone config` and answer:
   - `n` (new remote) → name: **`gdrive`**
   - storage: **`drive`** (Google Drive)
   - client_id / client_secret: press Enter (leave empty)
   - scope: **`drive.file`** (rclone can only see files it created itself, which is the safest choice)
   - leave the rest at the defaults, then say **yes** to "auto config". A browser opens; log in with your Google account and allow access.
   - `q` to quit.
3. Test it: `rclone mkdir gdrive:tb_checkpoints`, then `rclone lsd gdrive:` should list `tb_checkpoints`.
4. Run `rclone config show gdrive` and copy **all** of the output, starting at the `[gdrive]` line.

⚠️ That text contains a login token for your Drive. Treat it like a password: never paste it into a notebook cell, a commit or a chat. If it ever leaks, revoke it at https://myaccount.google.com/permissions ("rclone").

### 0.3 Kaggle secrets
In any Kaggle notebook: top menu **Add-ons → Secrets → Add a new secret**.

| Label | Value |
|---|---|
| `RCLONE_CONF` | the whole `rclone config show gdrive` output from step 0.2.4 |
| `GITHUB_TOKEN` | only if the team repo is private: a GitHub *fine-grained* token with **read-only "Contents"** access to that one repo (GitHub → Settings → Developer settings → Personal access tokens) |

Secrets are attached per notebook. In each new notebook, open *Add-ons → Secrets* and tick the ones it needs.

---

## 1. Create a Kaggle notebook
1. https://www.kaggle.com/code → **+ New Notebook**.
2. Rename it (click the title), e.g. `p3-06-dfine-s-512`.
3. Right-hand panel (**Session options**; if you don't see it, click the `>|` icon):
   - **Accelerator → GPU T4 x2** (or *GPU T4*). Our code uses one GPU.
   - **Internet → On**.
   - **Persistence → Files only** if offered. This keeps `/kaggle/working` between interactive restarts, which is a useful extra safety net.
4. Data: **+ Add Input** → search for the dataset you need (e.g. TBX11K, or the team's private processed dataset). Inputs appear read-only under `/kaggle/input/<name>/`.
   - Datasets you upload yourself must be **Private**. Never upload anything with patient identifiers. Never upload sealed splits (`internal_test`, `challenge_test`); training uses train and development only.

## 2. Add the repo (first cell)
```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

REPO = "github.com/<org>/cv_proj.git"      # TODO: team repo
BRANCH = "main"                            # or your p3/... branch
try:
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = f"https://x-access-token:{token}@{REPO}"
except Exception:
    url = f"https://{REPO}"                # public repo
subprocess.run(["git", "clone", "-q", "--recurse-submodules", "-b", BRANCH, url, "/kaggle/working/cv_proj"], check=True)
os.chdir("/kaggle/working/cv_proj")
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
```
(The token is never printed. Don't add `print(url)`.)

## 3. Install requirements (second cell)
Kaggle already has CUDA PyTorch. Do **not** reinstall torch; install only the extras:
```python
!pip install -q -r requirements/p3.txt      # P2 uses requirements/p2.txt
import sys; sys.path.insert(0, "src")
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
```
If you see `CUDA not available`, the accelerator is off: go back to step 1.3.

## 4. Connect Drive

### 4a. Kaggle (rclone)
```python
!wget -q https://downloads.rclone.org/rclone-current-linux-amd64.zip -O /tmp/rclone.zip \
  && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/
from p3_ops.cloud import setup_rclone_from_kaggle_secret
print("rclone configured:", setup_rclone_from_kaggle_secret())   # reads the RCLONE_CONF secret
!rclone lsd gdrive:tb_checkpoints
```
`CheckpointStore` finds the secret by itself and mirrors every checkpoint to `gdrive:tb_checkpoints/<run>`. To use a different folder, set `os.environ["TB_RCLONE_REMOTE"] = "gdrive:some/other/folder"` before creating the store.

### 4b. Colab (alternative)
Runtime → *Change runtime type* → **T4 GPU**. Then:
```python
from p3_ops.cloud import CheckpointStore
store = CheckpointStore.for_run("p3_dfine_s_512")   # mounts Drive (a consent popup appears) and saves to MyDrive/tb_checkpoints/
```
On Colab, checkpoints are written straight to Drive, so no rclone is needed.

## 5. Run a job
Every training script uses the helper in the same way:
```python
from p3_ops.cloud import CheckpointStore, resume_if_exists, save_checkpoint

store = CheckpointStore.for_run("p3_dfine_s_512")   # Kaggle: /kaggle/working/checkpoints/<run> + Drive mirror
store.pull()                                         # fetch checkpoints from Drive (needed after a dead session)
state = resume_if_exists(store.dir, model, optimizer, scheduler, extra={"scaler": scaler})
for epoch in range(state.start_epoch, EPOCHS):
    train_one_epoch(...)
    metrics = evaluate_on_development(...)
    save_checkpoint(model, optimizer, scheduler, epoch, metrics, SEED, store.dir,
                    is_best=..., extra={"scaler": scaler})
    store.push()                                     # copy to Drive; failures only print a warning
```
- Use the **same run name** every time you restart the same experiment. A new run name starts from scratch.
- The run's seed is saved in every checkpoint. Resuming with a different seed is refused.
- Only the **last 2 epochs + the best** are kept, locally and on Drive.

**Try it with the toy demo first (about 1 minute):**
```python
!PYTHONPATH=src python -m p3_ops.cloud.demo --run demo_check --epochs 4
```

### Long jobs: "Save & Run All"
An interactive session stops when you close the tab or go idle. For runs longer than about an hour, click **Save Version → Save & Run All (Commit)**. Kaggle then runs the whole notebook in the background (up to the session limit, see section 8), and you can close the browser. Follow progress under *Your Work → the notebook → Versions → Logs*.

## 6. When the session dies
Symptoms: "Session stopped", "Kernel died", a quota or time limit, or a frozen tab.

1. **Don't panic, and don't change the run name.**
2. Check Drive: `rclone ls gdrive:tb_checkpoints/<run>` (laptop or a new notebook) should list `ckpt_epochNNNN.pt` files.
3. Start a new session (*Run → Restart & clear*, or open the notebook again → *Edit*). Re-run sections 2–4.
4. Re-run the training cell **unchanged**. The first line printed should say `resumed from ckpt_epochNNNN.pt`, and training continues at epoch NNNN+1.
5. If nothing was mirrored (rclone was not set up), check `/kaggle/working/checkpoints/<run>` (still there if *Persistence* was on), or the **output** of a saved version. To use a saved version's output: *+ Add Input → Your Work → notebook output*, then
   `os.environ["TB_CKPT_SEED_DIR"] = "/kaggle/input/<that-output>/checkpoints/<run>"` before `store.pull()`.
6. A half-written checkpoint is skipped automatically: you'll see the warning `skipping unreadable checkpoint`, and the previous epoch is used instead.

## 7. Done with a run
- The best weights are `ckpt_best.pt` (with `best.json`) on Drive. Record their **SHA-256** in the decision file (D12 / D17), e.g. `sha256sum ckpt_best.pt`.
- Large checkpoints stay on Drive. Never commit `.pt` checkpoints to git except small heads tracked through LFS (README §3).

## 8. GPU quota and limits (check the live numbers on Kaggle)
- **Weekly GPU quota:** Kaggle gives a fixed number of GPU hours per week, historically around **30 h/week**. Your remaining hours are shown in the notebook's session panel and on your profile (*Settings / Quotas*). The counter resets once a week; hover it to see when.
- **Session limit:** one session can run for at most about **12 h**, after which it is stopped. Plan epochs so that a checkpoint is written well within that.
- **Idle interactive sessions** are stopped after a period of inactivity. Use *Save & Run All* for anything long.
- Time with the accelerator switched on counts against the quota **even when idle**. Turn the GPU off (or stop the session) when you're just reading or editing.
- `/kaggle/working` holds about 20 GB of output; `/kaggle/temp` is not saved.
- Team plan: P3 ≤ 10 GB of Drive for detector checkpoints (README §3 storage budget). Keep `keep_last=2`.
- **Colab (free):** GPU availability isn't guaranteed, sessions are shorter, and idle sessions are disconnected sooner. Use it as a backup.

## 9. Troubleshooting
| Problem | Fix |
|---|---|
| No GPU option, or Internet greyed out | Verify your phone (0.1) |
| `rclone configured: False` | Secret `RCLONE_CONF` is not attached to this notebook (*Add-ons → Secrets*, tick it) |
| `WARNING: no rclone remote; checkpoints are NOT copied to Drive` | As above, or rclone isn't installed: re-run cell 4a |
| `rclone sync failed: ... quota` | Drive is full: delete old runs in `tb_checkpoints/` |
| `checkpoint seed X != requested seed Y` | You changed the seed of an existing run. Use the original seed or a new run name |
| `CUDA out of memory` | Lower the batch size (see `docs/p3_dfine_feasibility.md` for the T4 limits) |
| Clone asks for a password | Private repo without a `GITHUB_TOKEN` secret (0.3) |

---

## 10. Deliberate-kill test log (P3-02 acceptance: a killed session resumes losing ≤ 1 epoch)

**Procedure: notebook `notebooks/p3/kill_resume_test.ipynb`** (Kaggle, GPU optional, about 20 minutes)

Before you start: build the code bundle on your laptop (`python -m p3_ops.kaggle_bundle`, which writes
`artifacts/p3/kaggle/p3_code_bundle.zip`) and upload it as a **private** Kaggle dataset. Do sections 0.2-0.3 once,
so the `RCLONE_CONF` secret exists.

1. Kaggle → *+ New Notebook* → *File → Import Notebook* → upload `notebooks/p3/kill_resume_test.ipynb`.
   Session options: Accelerator **GPU T4** (or None), Internet **On**. Then *+ Add Input* → your bundle
   dataset, and *Add-ons → Secrets* → tick **RCLONE_CONF**.
2. Run **Cells 1-3**. Cell 3 must print `rclone configured: True`.
3. **Part A (automatic):** run **Cell 4**. It kills its own training process half-way through epoch 4, deletes
   the local checkpoints and resumes from Drive. It prints a one-row table.
4. **Part B (the I3 acceptance test):** run **Cell 5**. When the output shows `epoch   4` (or later), stop the
   whole session: top-right power button → **Stop session**. Write down the last `epoch N` number you saw.
5. Open the notebook again (*Edit*). Run **Cells 1-3**, set `LAST_PRINTED_B = N` in Cell 1 and re-run Cell 1,
   then run **Cell 6** (resume; it lists the checkpoints on Drive first) and **Cell 7**. Do not run Cell 5 again.
6. Copy everything Cell 7 prints (the table and the `KILLTEST_RESULT` lines) and paste it back to Claude Code,
   or log it yourself: save it to a file and run
   `python -m p3_ops.kill_test_log --paste <file>` (shows the rows), then the same command with `--write`
   (appends them to the table below). Only a Part B row from Kaggle is written as a "Kaggle" row; Part A is a
   supporting row.

Manual alternative (no notebook): sections 1-4a, then
`!PYTHONPATH=src python -m p3_ops.cloud.demo --run killtest_<date> --epochs 20 --epoch-seconds 30`, stop the
session after `epoch   4`, reopen, re-run sections 2-4a and the same command, and check the first line
(`start at epoch …`).

**Pass rule:** epochs lost ≤ 1, where epochs lost = (last epoch printed before the kill + 2) − (epoch the restart begins at). So the restart must begin right after the last completed epoch, and only the interrupted epoch is re-run. Checkpoints must be on Drive, and the run must finish to epoch 19. (Before 2026-10-10 the wording was "resumed epoch ≥ last completed epoch", which would have allowed 2 lost epochs.)

| Date | Platform / GPU | Run name | Last epoch printed before kill | Killed how | Checkpoints on Drive after kill | First line after restart | Epochs lost | Pass? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | Kaggle / T4 | | | Stop session | | | | | |
| | Colab / T4 (optional) | | | Runtime → Disconnect and delete runtime | | | | | |

CPU simulation (automated, `tests/p3_ops/`): kills at epochs 0, 1, 3 and 5 plus a double kill. Each time the run resumes at the killed epoch, and the final weights are bit-identical to a run that was never killed. Passed on 2026-10-07.
