# P3-01 · D-FINE setup, licences and T4 feasibility

Status: **template, awaiting Kaggle T4 numbers**. D20 (`decisions/p3_licences.json`) is `locked=false`.

## 1. D-FINE pin

| Item | Value |
|---|---|
| Repository | https://github.com/Peterande/D-FINE (official) |
| Path | `third_party/D-FINE` (git submodule) |
| Pinned commit | `956d1709314c2c6a4df6f34de232054578a7449f` (2026-08-19, "Update README with license note for Objects365") |
| Why this commit | Upstream has no release tags, so we pin the tip of `master` on 2026-10-07. It includes `03f5cb3` (2026-07-09), which replaces the unsafe `yaml.Loader` with `yaml.SafeLoader` (CWE-502); the only later change is a README licence note. Since 2025-07-09 (`d669475`) only three files changed: `src/core/yaml_utils.py` (the SafeLoader fix), a TensorRT C++ example and the README. Model and training code are untouched. |
| Check | `make p3-setup` fails if the submodule HEAD differs from the pin in `mk/p3.mk` |

## 2. Licences (read from the actual LICENSE files, 2026-10-07)

| Component | Version inspected | Licence (file) | SHA-256 of LICENSE | Verdict |
|---|---|---|---|---|
| D-FINE | commit `956d170` | Apache-2.0 (`LICENSE`, Apache License Version 2.0) | `1eb85fc9…a2a761ae9c8c6` | accepted |
| pytorch-grad-cam | PyPI `grad-cam==1.5.7` (sdist), cross-checked with GitHub `7043934` | MIT, © 2021 Jacob Gildenblat | `daaa1016…e668abc7403` (sdist, CRLF) | accepted |
| RT-DETRv2 (extension) | `lyuwenyu/RT-DETR` commit `29320b6`, `rtdetrv2_pytorch/` | Apache-2.0 (root `LICENSE`; no separate licence in subdir) | `1eb85fc9…a2a761ae9c8c6` | accepted |

Full hashes are in D20. Notes:
- **Weights ≠ code.** The D-FINE README states that Objects365-trained checkpoints (`*_obj365`, `*_obj2coco`) may fall under Objects365 dataset terms. The default custom configs only download ImageNet-pretrained PP-HGNetV2 backbones. Which initial weights we fine-tune from is decided in P3-06 and its terms will be recorded there.
- D-FINE's `requirements.txt` omits `matplotlib` (imported in `src/solver/validator.py`), and torchvision's `CocoDetection` needs `pycocotools`. Both are in `requirements/p3.txt`.

## 3. Feasibility method

Script: `notebooks/p3/p3_01_dfine_feasibility.py` (Kaggle wrapper: `notebooks/p3/p3_01_dfine_feasibility.ipynb`, or `make p3-feasibility`).

- **Data:** synthetic COCO, 320 images of 512 × 512, single category `{id: 1, name: "tb_lesion"}`, about 60% with 1–4 boxes and the rest negatives (`src/p3_det/synthetic_coco.py`). No real or sealed data.
- **Model:** official `dfine_hgnetv2_{s,m}_custom.yml` with the dataset, input size, batch and AMP overridden. `num_classes = 2` because C06 uses `category_id = 1` and D-FINE uses the id as the label index. P3-04 decides whether to remap to 0; the effect on memory is negligible.
- **Peak memory / batch:** for each batch in {2, 4, 6, 8, 12, 16, 20, 24, 32, 48}, run 3 AMP training steps (forward, loss, backward, optimiser, EMA) at the **largest multi-scale size** D-FINE trains at (`int(1.25·base/32)·32`: 512 → 640 px, 640 → 800 px). Record `torch.cuda.max_memory_reserved` and `max_memory_allocated`. Stop at the first OOM.
- **Seconds per epoch:** at ONE common batch for both sizes (the smaller of the two chosen batches, because D16 needs identical schedules), run 2 real D-FINE epochs (`train_one_epoch`: augmentation, multi-scale, EMA). The second epoch is timed, and its peak memory is recorded as a cross-check. s/epoch for the real training set = s/image × n_train. One evaluation pass (EMA weights + postprocessor, as P3-06 runs on development every epoch) is timed too and extrapolated to n_dev.
- **Paste-back:** the notebook's last cell prints one JSON block; `python -m p3_det.fill_training_config --json <file> --hours-per-session <h>` checks the rule, picks the common `batch_size`, proposes the same `epochs` for both sizes, shows the diff and (with `--write`) fills the configs and section 5.
- **Seed:** 0, via `tbcore.seed.set_all` (fallback seeding while tbcore is missing; the JSON records which was used).
- **Units:** GB = 10⁹ bytes. A T4 reports about 15.6 GB of total memory.

## 4. Acceptance rule (pre-registered)

> For each input size, the **chosen batch** is the largest batch size whose peak **reserved** GPU memory is **≤ 14 GB** at the largest multi-scale training size, with AMP on a single Kaggle T4. The real-epoch peak at that batch must also be ≤ 14 GB. If it is not, step down to the next smaller batch.

This only fixes the batch size per input size for P3-06. The 512-vs-640 choice itself is D16, made on development AP@0.50.

## 5. Results (Kaggle T4, fill in from the run)

Run info: date `____`, platform `kaggle`, GPU `Tesla T4` (`__._` GB), torch `____` / CUDA `____`, D-FINE commit `956d170`, seed `0`, seed util `____`, n_mock `320`, n_train (extrapolation) `____`, artifact `artifacts/p3/feasibility/dfine_t4_________.{json,csv}`.

### 5.1 Peak reserved memory per batch at the largest multi-scale size (GB; OOM = out of memory)

| Model | Input | Train-scale max | bs 2 | 4 | 6 | 8 | 12 | 16 | 20 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D-FINE-S | 512 | 640 | | | | | | | | | | |
| D-FINE-S | 640 | 800 | | | | | | | | | | |
| D-FINE-M | 512 | 640 | | | | | | | | | | |
| D-FINE-M | 640 | 800 | | | | | | | | | | |

### 5.2 Summary

| Model | Input | Largest workable batch | Chosen batch (≤ 14 GB) | Peak reserved @ chosen (GB) | Real-epoch peak (GB) | ≤ 14 GB? | s/epoch (320 mock) | s/image | s/epoch extrapolated (n_train) |
|---|---|---|---|---|---|---|---|---|---|
| D-FINE-S | 512 | | | | | | | | |
| D-FINE-S | 640 | | | | | | | | |
| D-FINE-M | 512 | | | | | | | | |
| D-FINE-M | 640 | | | | | | | | |

### 5.3 Outcome

- Chosen batch at 512: `__` (model `_`); at 640: `__` (model `_`). Both ≤ 14 GB: `yes / no`.
- Notes (dataloader bottleneck, warnings, OOM behaviour): ____

## 6. CPU smoke run (plumbing only, not a feasibility result)

`make p3-feasibility-smoke` (D-FINE-N, batch 2, CPU, no AMP, no pretrained weights) completed for 512 and 640 on 2026-10-07 (Windows, Python 3.14, torch 2.14.1+cpu). Memory figures are not available on CPU.
