# Protocol — box-supervised regional head

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory.** TBX11K calibration is the primary evaluation split; Vietnam TUNE is
exposed and reported for context only. Nothing is adopted into production.

## 1. Hypothesis

> TBX11K annotates every TB lesion with a box labelled `ActiveTuberculosis` (1) or
> `ObsoletePulmonaryTuberculosis` (2). Nobody has used that as supervision: the project,
> and the literature on this dataset, collapse it to an image label. Supervising the
> 12×12 regional head at the **region** level — active-lesion regions positive,
> obsolete-only and healthy regions negative — should improve active-TB discrimination
> over image-level MIL with the same target, **in either direction**.

Motivation from the record: relabelling obsolete TB as negative at image level was worth
+0.0203 on Vietnam (R12). Region-level supervision is the same signal, applied where the
lesion is rather than to the whole image.

## 2. Arms — one fitting path, three supervision regimes

| arm | image-level target | region loss | notes |
|---|---|---|---|
| **A** | `tb_label` | none | production; must reproduce 0.9498 on calibration vs active target |
| **B1** | `active_tb_label` | none | R12 arm; must reproduce 0.9460 |
| **R** | `active_tb_label` | **yes, λ = 1** | the candidate |

Everything else identical: frozen RAD-DINO 12×12 lungs-masked regions, `Linear(768→1)`
per region, max aggregation, AdamW 3e-3 / batch 256 / wd 1e-4 / ≤120 epochs, early
stopping on TBX11K development image-level AUROC (patience 15), logit adjustment, seeds
17 / 29 / 43 / 61 / 79, seed-mean scores. Eligible training rows: `healthy` + the three TB
categories (3,241 images), as in production.

**Built-in correctness gate:** arm R's trainer with λ = 0 must reproduce arm B1's
calibration AUROC to within 0.0005, or the run halts.

## 3. Region labels (train only)

For each training image, each of the 144 regions gets one of {+1, 0, masked}:

| image | region overlaps an active box (cat 1) | overlaps only obsolete boxes (cat 2) | overlaps no box |
|---|---|---|---|
| healthy | — | — | **0** |
| TB (any subtype) | **+1** | **0** | **masked** (TB image, location unknown) |

Category 3 does not occur in trainval (verified in R12). "Overlaps" = the box's pixel
footprint intersects the region's pixel footprint by ≥ 1 patch after the exact
preprocessing chain: box (original coords) → 512 (assert originals are 512) → 518
(×518/512) → 14-px patches → 37→12 adaptive-pool bins
`[(0,4),(3,7),(6,10),(9,13),(12,16),(15,19),(18,22),(21,25),(24,28),(27,31),(30,34),(33,37)]`.

Region loss: BCE with logits over the labelled regions, mean over labelled regions in the
batch, added to the image-level MIL loss with **λ = 1, fixed, not tuned**.

## 4. Evaluation

**Primary:** paired ΔAUROC on **TBX11K calibration** (899; 81 active-TB positives), arm R
minus arm B1, scored against `active_tb_label`, paired bootstrap 4,000 resamples seed 0,
identical indices. Minimum meaningful effect **+0.02**, interval excluding zero.
Half-width > 0.05 → INCONCLUSIVE.

**Secondary:** R − A; AUPRC with prevalence; sensitivity at the development-fitted
threshold (≥ 0.90 specificity on dev) with specificity; Vietnam TUNE exploratory (all
three arms, no threshold fitted on TUNE); per-seed spread.

**Localisation (exploratory, dev TB images with ≥ 1 active box, n ≈ 49):** for each arm,
does the arg-max region hit an active box? Top-1 and top-3 hit rate with image-level
bootstrap CIs; baselines = uniform random region, centre region (6,6), and random region
inside the lung mask. Reported for A, B1 and R so the effect of supervision on
localisation is visible.

## 5. Constraints

No access to Vietnam REPORT or TBX11K `internal_test`. No production file modified.
Training and evaluation rows are disjoint by image id (TBX11K has no patient id; stated).
Localisation on development is in-sample for early stopping, not for the box loss — the
boxes of development images are never used in training. Everything labelled exploratory.
