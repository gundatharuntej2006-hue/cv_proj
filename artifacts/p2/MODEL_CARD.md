# Model card — TB screening regional head, bundles v2 (served) and v3 (research candidate)

**Date:** 2026-11-01 · **Owner:** P2 (Atul Dhull), BMSIT&M BAI506 · **Status:** research system; **not clinically validated; not a medical device**

## Intended use

Ranking a batch of frontal chest radiographs by tuberculosis suspicion for human review in a screening programme, with the region that produced each score shown on the image. Output is a **screening score (logit)** with flags at two development-fitted operating points. It is **not** a probability (measured: sigmoid of the logit has ECE 0.49, calibration intercept −5.3, slope 0.30 on the Vietnam screening cohort; audit A1), not a diagnosis, and not a replacement for microbiological confirmation. Out of scope: lateral views, paediatric films, non-chest images (the quality gate does not detect them), differential diagnosis of non-TB lung disease.

## Models

| | bundle v2 (`artifacts/p2/bundle_v2_regional`) | bundle v3 (`artifacts/p2/bundle_v3_candidate`) |
|---|---|---|
| status | **served default** | **research candidate, not promoted** |
| encoder, frozen | `microsoft/rad-dino` @ `110cbc18…` (86.6 M) | `microsoft/rad-dino-maira-2` @ `a6132efb…` |
| features | 512 px, PSPNet lung mask (outside → image mean), 518 px, 37×37 tokens → 12×12 × 768-d, L2 per region | same |
| head | Linear(768→1) per region, max over 144, 5 seeds averaged (3,845 params) | same, sha `7f61a082…` |
| training target | `tb_label` (active + latent + mixed TB vs healthy), TBX11K train 3,241 | `active_tb_label` (latent relabelled negative), 3,241 |
| thresholds | logits at 0.70 / 0.90 specificity on TBX11K development: −3.3368 / −0.9052 | −4.0317 / −2.3512 |
| fusion (optional) | LR on standardised logit + 12 fields, fitted on Vietnam TUNE (exposed); +13 params | same; scanner columns are site-specific |

## Performance (all TBX11K numbers on a provisional split; all Vietnam numbers on an exposed cohort)

| cohort | v2 | v3 | comparator (Lunit 3.9.0.1) |
|---|---|---|---|
| TBX11K development (659) | 0.9368 | 0.9321 (vs active) | — |
| TBX11K calibration (899, vs active) | 0.9498 | 0.9466 | — |
| Vietnam REPORT (1,356) | **0.7242 [0.6843, 0.7618]** | not scored (sealed since R11) | 0.7862 |
| Vietnam TUNE raw (1,352) | 0.7237 | 0.7585 [0.722, 0.796] | 0.7780 |
| Vietnam TUNE fused, nested CV | 0.7809 | **0.7941** | 0.7778 |
| previous-TB subgroup, TUNE fused | 0.7122 | 0.7349 | 0.7144 |
| sens @ 70 % spec, TUNE fused | 0.729 | 0.741 | 0.708 |
| AUPRC, TUNE fused | 0.276 | 0.300 | 0.306 (raw 0.328) |
| localisation, 49 dev images with boxes, top-1 region-cell hit | 0.959 [0.898, 1.0] | — | — |

Paired v3-fused − comparator-raw on TUNE: **+0.0170 [−0.0193, +0.0527]** — no superiority, equivalence or non-inferiority claim is supported. Without the scanner term: 0.7803 vs 0.7780.

## Probability calibration (audit A1, Vietnam TUNE, exposed, out-of-fold, 10 equal-mass bins)

| output | ECE [95 % CI] | intercept | slope | Brier skill vs prevalence |
|---|---|---|---|---|
| image logit through a sigmoid (v2) | 0.487 [0.464, 0.510] | −5.27 | 0.30 | −3.6 |
| fused v2 | 0.024 [0.018, 0.044] | 0.00 | 0.94 | +0.09 |
| fused v3 | 0.016 [0.013, 0.035] | 0.00 | 0.94 | +0.11 |
| comparator raw score | 0.439 [0.419, 0.459] | −3.41 | 0.68 | −2.6 |

The fused output is probability-shaped **only where its 13 coefficients were fitted** (this cohort's prevalence and three scanners); per-scanner ECE 0.015–0.045. No temperature, Platt or isotonic map is shipped. Never apply a sigmoid to the screening score.

## Known failure modes

- **Previous tuberculosis:** AUROC falls from 0.7720 to 0.6392 (v2, construction A); the comparator also falls. Healed-disease appearance raises the score.
- **Scanner-associated offset:** the v3 score sits ≈ 1.3 logits higher on one of three Vietnam scanners at similar prevalence; pooled ranking and referral burden are affected (50 % referred on that scanner at a fixed rule). A per-scanner adjustment recovers ≈ 0.02 AUROC and needs either labels or a representative unlabelled sample; the label-free version fails under population shift (R25) and is shipped fail-closed, experimental.
- **Negative class:** trained against healthy controls; a screening queue's negatives include chronic lung disease, which the model scores as abnormal.
- **Distribution shift is not detected** by the v1 kNN gate (passes 99.2 % of the screening cohort).
- **Thresholds do not transfer:** the development 0.90-specificity logit gives ≈ 0.35 specificity on Vietnam; local threshold recalibration (≈ 25 labelled images) is required at any new site.

## Training data and provenance

TBX11K core train/val lists (CC BY 4.0), provisional split C01 (definitive split pending from P1); one image per individual, no patient identifier; 158 cross-split near-duplicate pairs in the official splits; label defect `tb1199` outstanding. Fusion coefficients: Vietnam TUNE (Dryad 10.5061/dryad.5tb2rbpcv). Not used for any weight or threshold: TBX11K `internal_test`, Vietnam REPORT.

## Evaluation discipline

25 pre-registered runs with hashed protocols, reproduction gates, paired bootstrap intervals; index in `artifacts/P2_PROJECT_RECORD_20261101.md`. The only evaluation that can validate either bundle is the pre-registered head-to-head on an untouched cohort (`artifacts/headtohead_prereg_v1/`), which has not yet run.

## Compute

Encoder 0.030 s/image fp16, peak 0.69 GB; full pipeline median 0.13 s/image on an RTX 4050 laptop GPU after a ~40 s model load; ≈ 2.1 s/image on CPU.

## Licences

RAD-DINO and MAIRA-2: see the model cards on Hugging Face (revisions pinned above). TBX11K: CC BY 4.0. Vietnam cohort: Dryad terms. torchxrayvision: Apache-2.0.
