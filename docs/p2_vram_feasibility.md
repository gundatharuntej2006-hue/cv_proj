# P2-01 VRAM feasibility on a 6 GB laptop

**Task:** P2-01 · **Owner:** Person 2 · **Date:** 2026-09-29 · **Verdict: GO**

Acceptance criterion from the work plan: peak `max_memory_allocated` at or below 5.5 GB for the embedding and Grad-CAM paths, plus P2-03's requirement that 11,200 images embed in under 60 minutes on the laptop.

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM reported by the driver | 6.0 GB |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| Python | 3.13.2 |

## Model actually loaded

| Item | Value |
|---|---|
| Model | `microsoft/rad-dino` |
| Pinned revision | `110cbc18d5133582e320b43d53bf5c44e410c936` |
| Licence (model card) | MIT, LICENSE file starts "MIT License" |
| Input size | 518 x 518 pixels |
| Patch size | 14 |
| Tokens per image | 1370 |
| Embedding width | 768 |
| Transformer blocks | 12 |
| Parameters | 86.6 M |

The embedding width of 768 confirms contract C07, which declares `emb float16 [N, 768]`.

## 1. Embedding extraction, fp16 (the P2-03 path)

| Batch | Peak VRAM (GB) | Images/second | Minutes for 11,200 images | Within budget |
|---|---|---|---|---|
| 1 | 0.21 | 39.1 | 4.8 | yes |
| 2 | 0.24 | 36.7 | 5.1 | yes |
| 4 | 0.30 | 35.4 | 5.3 | yes |
| 8 | 0.42 | 39.7 | 4.7 | yes |
| 16 | 0.67 | 39.4 | 4.7 | yes |

Frozen RAD-DINO in half precision is far cheaper than budgeted. At batch 16 the peak is 0.67 GB, about 12 percent of the 5.5 GB allowance, and one full pass over TBX11K takes about 5 minutes against the 60-minute criterion.

**Consequence for P2-06.** The ten preprocessing variants (full image plus four crop margins, each with CLAHE off and on) cost roughly 47 minutes of GPU time in total for the training split, so the variant sweep is cheap and does not need Kaggle.

## 2. Grad-CAM path, forward and backward at batch 1

Peak VRAM **0.84 GB**, well inside the budget, and the gradient of the last block's activations was captured, which is what Person 3's Grad-CAM needs from `classifier_module()` (task P3-09).

One implementation note for P3-09 and D19: the backbone is frozen, so its parameters have `requires_grad=False` and the activations carry no gradient unless the graph starts earlier. Set `requires_grad=True` on the input tensor. Without that, `retain_grad()` raises and the heatmap is silently unavailable.

## 3. EfficientNet-B3 training step at 512 pixels (reference only)

| Batch | Peak VRAM (GB) | Seconds/step | Images/second | Within budget |
|---|---|---|---|---|
| 4 | 1.87 | 0.186 | 21.6 | yes |
| 8 | 3.53 | 0.231 | 34.7 | yes |
| 16 | 6.89 | 2.734 | 5.9 | no |

**Important measurement artefact.** Batch 16 did not raise an out-of-memory error, yet its peak exceeds both the budget and the card's physical memory. On Windows the driver silently spills into shared system memory instead of failing, and the cost shows up as speed: throughput collapses from 35 to 6 images per second. A run completing is therefore not evidence that it fitted. Everyone on the team should check peak memory, not just whether the script survived.

Usable laptop batch size for the baseline is **8**. The real baseline still trains on a Kaggle T4 as planned (P2-12); this table only sizes local debugging runs.

## Verdict

| Check | Result |
|---|---|
| Embedding path within 5.5 GB | PASS (best batch 16, 0.67 GB) |
| 11,200 images under 60 minutes | PASS (4.7 minutes) |
| Grad-CAM path within 5.5 GB | PASS |
| **Overall** | **GO** |

The frozen-backbone design in the methodology is comfortable on this hardware. The binding constraint is not the foundation model; it is detector training, which stays on Kaggle.

## How to reproduce

```
py -3 scripts/p2_01_vram_feasibility.py     # writes artifacts/p2/p2_01_vram_feasibility.json
py -3 scripts/p2_01_write_report.py         # writes this file and decisions/p2_licences.json
```

Raw measurements: `artifacts/p2/p2_01_vram_feasibility.json`.
