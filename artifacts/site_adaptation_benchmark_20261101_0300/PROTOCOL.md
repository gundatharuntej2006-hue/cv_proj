# Protocol — efficiency and site-adaptation benchmark

**Hashed before any measurement.** The script re-hashes this file and refuses to run if it
changed. **Exploratory.** Nothing here is a superiority claim, a clinical claim, or
independent validation.

## 1. Questions

1. What does it cost, in wall-clock time and hardware, to screen an image and to adapt the
   classifier to a new site — on GPU and on CPU?
2. Does **refitting the 769-parameter head** on local labelled images improve held-out
   performance at a fixed operating point, **over refitting only the threshold**?
3. How many local labelled images does either take?

## 2. Model and recipe — production, unchanged

Frozen RAD-DINO ViT-B/14 (`110cbc18…`), 12×12 lungs-masked regional features, one shared
`Linear(768→1)` per region, max over regions. AdamW 3e-3 / batch 256 / wd 1e-4 / ≤120
epochs, early stopping on **TBX11K development** (patience 15), logit adjustment.
`p2_32_mil_sweep.train` and `.score` imported unmodified. **Head seed 17** throughout.

## 3. Sites

| site | n | TB | prevalence | role |
|---|---|---|---|---|
| **Shenzhen** | 662 | 336 | 0.508 | primary |
| Montgomery | 138 | 58 | 0.420 | secondary — too small for anything but a direction |

Both are **curated case-control sets at ~50% prevalence, both previously used by this
project for external validation.** They can show whether adaptation moves held-out
performance; they **cannot** establish generalisation to screening populations.

## 4. Split — patient-disjoint, fixed before fitting

Each site is split once, **50 / 50, stratified by label, seed 20261101**, into an
**adaptation pool** and an **evaluation set**. One image per patient in both sets, so
image-level and patient-level disjointness coincide. The evaluation set is never used to
fit a head, a threshold, or to select anything.

## 5. Arms, at each local sample size n

Local images are drawn from the adaptation pool only, stratified, **3 independent draws**
per n (seeds 0, 1, 2).

| arm | head | threshold | what it represents |
|---|---|---|---|
| **A0** | TBX11K head, frozen | TBX11K-development threshold | install and use, no adaptation |
| **A1** | TBX11K head, frozen | **refit on the n local images** | threshold-only recalibration — what current deployment guidance requires |
| **A2** | **refit on TBX11K train + n local, pooled** | refit on the n local images | head adaptation — the proposed differentiator |

n ∈ {0, 10, 25, 50, 100, 200, all} for Shenzhen (all = 331); {0, 10, 25, 50, all} for
Montgomery (all = 69). n = 0 is A0 by definition.

**Threshold rule, fixed:** the smallest score reaching **≥ 0.70 specificity** on the images
used to fit it (WHO screening operating point). For A0 that set is TBX11K development.
No threshold is ever chosen on an evaluation set.

Local images enter A2 **unweighted**. Upweighting is a hyperparameter that could only be
tuned on the evaluation site, so it is not done.

## 6. Endpoints — all on the held-out evaluation set

Primary: **sensitivity and specificity at the arm's own threshold**, reported together,
never one without the other. Also AUROC, referral rate, and — for reference only — the
oracle sensitivity at exactly 0.70 specificity on the evaluation set itself (an upper
bound no deployment could reach).

Uncertainty: percentile bootstrap, 2,000 resamples of the evaluation set, seed 20261101,
plus the spread across the 3 draws.

**Decision language.** "Head refit helps" requires A2's sensitivity *and* specificity to
both be at or above A1's at the same n, with the A2−A1 paired bootstrap interval on
sensitivity excluding zero. Anything else is reported as "no clear benefit" or "harm".

## 7. Timing

Measured, not extrapolated, except where stated:

| quantity | GPU | CPU |
|---|---|---|
| encoder load | measured | measured |
| encode throughput (s/image, images/s) | all 800 NLM images | **32 images**, per-image rate reported |
| head fit, 3,241 TBX11K images | measured | measured |
| **total adaptation** for n local = encode(n) + fit(3,241 + n) | measured per n | encode extrapolated from the 32-image rate; fit measured |
| peak GPU memory | measured | — |

CPU runs use the same code with CUDA disabled. fp32 on CPU will not reproduce the fp16
GPU embeddings bit-for-bit; CPU timing is reported for cost only, and the adaptation
curve uses the GPU embeddings throughout.

## 8. Constraints

No training of the backbone. No change to any production file. No access to Vietnam
REPORT or TBX11K `internal_test`. No per-patient output. Evaluation-set images are never
opened for fitting or selection. Shenzhen/Montgomery exposure is disclosed in every table.
