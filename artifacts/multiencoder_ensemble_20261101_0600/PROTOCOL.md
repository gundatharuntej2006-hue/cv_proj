# Protocol — multi-encoder regional ensemble

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory.** TBX11K calibration is the primary evaluation split; Vietnam TUNE is
exposed and reported for context only. Nothing is adopted into production.

## 1. Hypothesis

> Every head so far sits on one frozen encoder (RAD-DINO, `110cbc18…`). Two other
> public chest-X-ray foundation encoders exist with different pre-training data and
> objectives: **RAD-DINO-MAIRA-2** (same architecture, continued self-supervised training
> on a larger CXR corpus) and **XraySigLIP** (ViT-B/16 at 512 px, image–text contrastive
> on radiology reports). Encoders trained differently make different errors; a regional
> head on each, averaged, should discriminate better than any one of them, **in either
> direction**. This is the cheapest capacity increase available that keeps the
> model fully frozen, the head linear, and attribution exact.

## 2. Encoders — frozen, pinned

| key | model | revision | tokens | regional map |
|---|---|---|---|---|
| **D** | `microsoft/rad-dino` | `110cbc18…` | 37×37 patch-14 at 518 px | existing cache `regional_tbx11k_lungs_g12_v1` / `…calibration…` / `regional_vietnam_lungs_g12_v1` |
| **D2** | `microsoft/rad-dino-maira-2` | `a6132efb…` | 37×37 patch-14 at 518 px | adaptive-avg-pool 37→12, L2 per region |
| **S** | `StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli` (vision tower only) | `e2ad64ee…` | 32×32 patch-16 at 512 px | adaptive-avg-pool 32→12, L2 per region |

Inputs are the **same lungs-masked 512 px PNGs** the RAD-DINO cache was built from
(`data/processed/v1/tbx11k_lungs512`, `vietnam_img512_lungs`), each encoder's own
pinned preprocessor applied on top. Only TBX11K train / development / calibration and
Vietnam TUNE are embedded. Prefix tokens (CLS, registers) are dropped before pooling.

## 3. Arms — one fitting path

Per encoder: `Linear(d→1)` per region, max over 144 regions, `tb_label` target,
TBX11K train healthy + TB (3,241), AdamW 3e-3 / batch 256 / wd 1e-4 / ≤120 epochs, early
stopping on TBX11K development AUROC (patience 15), logit adjustment, seeds
17 / 29 / 43 / 61 / 79, seed-mean logits. Identical to production.

| arm | score |
|---|---|
| **D** | RAD-DINO head — must reproduce 0.9498 on calibration vs active target |
| **D2**, **S** | single-encoder heads, same trainer |
| **E3** | mean of the three heads' logits, each standardised by its **development** mean / sd |
| **E_DS**, **E_DD2**, **E_D2S** | the three pairwise means, same standardisation |
| **C3** | one `Linear(2304→1)` head on the concatenated regional features, same trainer |

Standardisation constants come from development only. No weights, temperatures or
mixing coefficients are fitted anywhere else.

## 4. Evaluation

**Primary:** paired ΔAUROC on **TBX11K calibration** (899; 81 active-TB positives),
**E3 minus D**, scored against `active_tb_label`, paired bootstrap 4,000 resamples
seed 0, identical indices. Minimum meaningful effect **+0.02**, interval excluding zero.
Half-width > 0.05 → INCONCLUSIVE.

**Secondary (reported, not decisive):** every other arm minus D on calibration; all arms
on Vietnam TUNE (Xpert reference, exposed); sensitivity at the development-fitted
0.90-specificity threshold; AUPRC; per-encoder encode time and peak GPU memory, because
a three-encoder product costs three times the compute and that must be weighed.

**Decision rule.** ACCEPT only if the primary contrast meets its bar. If a secondary arm
beats the bar and E3 does not, that is reported as a hypothesis for a new pre-registered
run, not as a result.

## 5. Partitions

TBX11K `internal_test` (901) and Vietnam REPORT (1,356) are not read, not embedded. No
per-case output. No Vietnam patient identifier is written to any table.
