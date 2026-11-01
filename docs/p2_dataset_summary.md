# Dataset summary and outcomes

**Owner:** Person 2 (classification + gate) · **Date:** 2026-10-16 · **Status:** all numbers
provisional — the split is not Person 1's final C01, and `internal_test` has been read only
in a labelled rehearsal. 13 GB of image data on disk.

---

## 1. What we train on

### TBX11K — core trainval, 8,400 images

Self-collected by top hospitals in China; **TB confirmed by sputum culture**, not by image
review; one image per unique individual; lesion boxes drawn by a 5–10-year radiologist and
reviewed by a 10+-year radiologist. This is the whole reason the dataset is valuable — TB
cannot be confirmed from an image, so culture-confirmed labels are rare.

| category | images | |
|---|---|---|
| healthy | 3,800 | |
| sick, non-TB | 3,800 | |
| **TB-positive** | **800** | 631 active · 139 latent · 30 mixed |
| **total** | **8,400** | |

Our split (seed 20260921), and the only TB images a model may learn from:

| split | TB+ | total | role |
|---|---|---|---|
| train | **541** | 5,941 | the only data the head sees |
| development | 59 | 659 | early stopping, variant selection |
| calibration | 99 | 899 | thresholds |
| internal_test | 101 | 901 | **sealed**, one-shot |

Note the imbalance: **541 TB against 5,400 non-TB, roughly 1:10** — which is why D10
(logit-adjusted loss) exists. And 59 TB cases in development means sensitivity can only move
in steps of 1/59, which is why that metric keeps saturating at 1.0000.

**Licence conflict, unresolved:** the COCO annotation files say CC BY-NC-SA 2.0; the project
website says CC BY 4.0. The non-commercial clause is the stricter reading and should be
assumed.

### Three findings about this dataset

1. **The official splits contain cross-split duplicates** — 158 near-duplicate pairs at
   cosine ≥ 0.95 across the official boundaries, 52 of them train→val. Any published number
   trained on the official train split and evaluated on official val is inflated by an
   unknown amount.
2. **The archive ships 576 images that are not part of the benchmark** — 400 Shenzhen/
   Montgomery and 176 DA/DB, under `imgs/extra/`, referenced **only** by the `all_*.txt`
   lists. Train on those lists and external data silently enters training. We use the core
   lists, so ours did not: verified at max cosine 0.8389 against all 8,400 training
   embeddings, zero pairs above 0.90.
3. **12.1% of the official test split is not Chinese hospital data** — 400 of its 3,302
   images are Shenzhen/Montgomery copies, distributed anonymised as `unknown_####.png`. Their
   labels are therefore partially recoverable from the public NLM originals, which is a
   benchmark-integrity issue worth documenting and not exploiting.

### TBX11K official test split — 3,302 images, unused

Labels are withheld by the authors (standard benchmark practice, like ImageNet's test set).
Unusable for training or evaluation, which is why we re-split trainval instead. Still useful
as *unlabelled* data for the OOD bank or distribution-shift measurement.

---

## 2. What we test on — external, never trained on

Each of these carries **its own controls from the same hospital and programme**. That matters:
a test set whose positives and negatives come from different sources can be scored highly by
recognising the acquisition pipeline rather than the disease. The widely used Kaggle
"Tuberculosis (TB) Chest X-ray Database" fails exactly this test — positives from
NLM/Belarus/NIAID, negatives from the RSNA pneumonia challenge — so we do not use it.

| dataset | TB+ | total | source | status |
|---|---|---|---|---|
| **Shenzhen** | 336 | 662 | NLM / Shenzhen No. 3 People's Hospital, China | **evaluated** |
| **Montgomery** | 58 | 138 | NLM / Montgomery County, Maryland, USA | **evaluated** |
| DA + DB | ~74 | 176 | NITRD, New Delhi, India | on disk, not yet used |

### External outcomes

| | n | TB | AUC | 95% CI |
|---|---|---|---|---|
| TBX11K development (internal) | 659 | 59 | 0.9950 | [0.9883, 0.9993] |
| **Shenzhen** | 662 | 336 | **0.9382** | [0.9189, 0.9552] |
| **Montgomery** | 138 | 58 | **0.8183** | [0.7308, 0.8986] |

Neither external interval overlaps the internal one — the degradation is real. At a
**transferred threshold**, with no local recalibration:

| site | sensitivity | specificity |
|---|---|---|
| Shenzhen | 0.8690 | 0.8742 |
| **Montgomery** | **0.6724** | 0.9500 |

**At Montgomery the system would miss one TB case in three.** That, not the AUC, is the
number for the limitations section. Re-fitting the threshold on local data recovers Shenzhen
to 0.9345 sensitivity at 70% specificity; it barely helps Montgomery, whose ceiling is set by
its AUC.

---

## 3. Out-of-distribution pools — for the gate, not the classifier

| pool | images | source | purpose |
|---|---|---|---|
| CheXpert Plus frontals | 150 | US academic hospital | cross-site control |
| CheXpert Plus laterals | 501 | same | alien input (wrong view) |
| Non-medical | 247 | Wikimedia Commons | alien input (not a radiograph) |

**Gate outcome at the corrected threshold (0.70):** 100% of both alien pools rejected, 0.2%
of in-distribution images falsely rejected, 4.9% across the two NLM external sites.

Site distance turns out to be a measurable spectrum that predicts everything:

| set | median OOD score | rejected |
|---|---|---|
| TBX11K | 0.3637 | 0.2% |
| Shenzhen | 0.5331 | 1.2% |
| Montgomery | — | 22.5% |
| CheXpert frontals | 0.7787 | 87.3% |
| lateral films | 0.8778 | 100% |
| non-medical | 0.8851 | 100% |

The ordering matches the AUC ordering. **A new site's distance is measurable in advance from
unlabelled images alone**, so an installer can tell whether the gate will work before
collecting a single label.

---

## 4. Headline outcomes

**Good:**
- Internal development AUC **0.9949**; sealed rehearsal on held-out test **0.9920**.
- External AUC **0.9382** on Shenzhen — real evidence of transfer.
- **RAD-DINO is justified by measurement**: ΔAUC +0.0409 [0.0140, 0.0727] over a
  DenseNet-121 trained on 800k chest X-rays. Off-the-shelf zero-shot floor is 0.8026.
- Gate catches 100% of aliens at a 0.2% false-flag cost.
- Immune to horizontal flip, rotation, JPEG, blur and gamma.

**The problem, and it is the most important result:**

**The outer 12% of the image predicts TB at AUC 0.9094** — 82.7% of the full image's
above-chance AUC, from a frame containing shoulders, image edge, collimation and background,
and no lung tissue.

| what the model can see | dev AUC | above-chance AUC retained |
|---|---|---|
| everything | 0.9949 | 100% |
| lungs only | 0.9889 | 98.8% |
| everything except lungs | 0.9532 | 91.6% |
| lungs dilated 48px, erased | 0.9493 | 90.8% |
| **outer 12% frame only** | **0.9094** | **82.7%** |

Not a masking artifact: dilating the erased region to remove the apices costs only 0.0039,
and the border variant needs no masking assumption at all. Two further lines agree — flipping
the image changes nothing (+0.0001) though a chest X-ray is strongly asymmetric, and
lungs-only is significantly *worse* than the full image. **Most of this classifier's
performance is not lung pathology.** TBX11K collected TB cases and controls as separate
groups, so anything differing between those acquisition groups is free signal.

**Other limitations found:**
- **Noise breaks the operating point, not the AUC.** Specificity falls 95% → 26% at noise
  σ=20 while AUC only drops to 0.966 and sensitivity rises. Robustness must be reported at a
  fixed threshold.
- **Active and latent TB are indistinguishable** to the model (AUC 0.55, chance), and 21% of
  its TB calls are healed old disease sent for urgent workup.
- **The REFER band is empty** — 0 of 901 cases — because probabilities are strongly bimodal.
- **The benchmark is close to saturated**: sensitivity at 70% specificity is 1.0000 for all
  ten preprocessing variants and both head types; an 18-dimensional interpretable feature
  vector reaches AUC 0.9024.

---

## 5. What is missing, and what it would cost

| want | why | cost |
|---|---|---|
| **Repeat the shortcut probe on Shenzhen** | Shenzhen also grouped its cases, so the 0.9382 may be partly non-pathological too. Until this is run, the external number cannot be cited as evidence of pathological generalisation. | ~20 minutes. **Highest priority.** |
| DA + DB evaluated | third geography; makes "distance predicts transfer" much stronger with three sites than two | ~10 minutes, already on disk |
| **VinDr-CXR** | 643 TB cases, Vietnam, radiologist consensus with 3–5 readers. The only legitimate route to substantially more labelled TB data. | PhysioNet credentialing + CITI course: hours of effort, days to weeks of waiting |
| TB Portals / Belarus, PadChest | more geographies | data use agreements, weeks |

**Not an option: labelling anything ourselves.** TB cannot be confirmed from an image — the
labels we rely on are culture results. And pseudo-labelling with our own model would encode
the border shortcut into the labels, where no later experiment could detect it.

---

**Sources for every number above:** `docs/p2_external_validation.md`,
`p2_shortcut_probe.md`, `p2_robustness.md`, `p2_latent_vs_active_tb.md`,
`p2_backbone_baseline.md`, `p2_ood_threshold_is_miscalibrated.md`,
`p2_tbx11k_composition_corrected.md`, `p2_duplicate_finding.md`,
`p2_d08_d09_recommendation.md`. Artifacts under `artifacts/p2/`.
