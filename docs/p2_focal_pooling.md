# Focal pooling: global average pooling was discarding the signal

**Owner:** Person 2 · **Date:** 2026-10-20 · **Status:** exploratory, provisional split ·
**Best screening result so far**

## The idea

Every result until now used RAD-DINO's **globally pooled** output: one 768-dimensional
vector summarising the whole radiograph. Tuberculosis is focal. A small apical cavity
occupies a handful of the 1,369 patch tokens, and average pooling dilutes it into 99.6%
normal lung.

That predicts precisely the failure pattern we measured — adequate on florid disease in
curated case-control sets, poor on subtle disease in a screening cohort, where lesions are
smaller because cases are found earlier.

The standard remedy is multiple-instance learning: score each region with a shared linear
head and aggregate so that one strongly abnormal region can carry the image.

**Implementation.** The 37×37 patch grid is average-pooled to **6×6 regions** (each about
86 px of the 518 px input, comfortably smaller than a typical lesion). That is 1/38th the
storage of keeping all tokens — 478 MB for 9,308 images rather than ~7 GB.

## Result

Identical linear scorer, identical seeds, identical training data (lungs-only masking,
healthy-only negatives). Only the aggregator changes:

| aggregator | TBX11K development | **Vietnam** | 95% CI |
|---|---|---|---|
| **mean** (equivalent to global pooling — the control) | 0.7764 | 0.6456 | [0.6125, 0.6785] |
| **max** | 0.9043 | **0.7058** | **[0.6768, 0.7333]** |
| top-3 mean | 0.9092 | 0.6928 | [0.6641, 0.7207] |
| logsumexp | 0.8718 | 0.6600 | [0.6289, 0.6897] |

Against the production global model on the same patients, paired:
**ΔAUC +0.0314, 95% CI [+0.0070, +0.0567] — significant.**

| | Vietnam AUC |
|---|---|
| production global model | 0.6744 |
| **regional max-MIL** | **0.7058** |
| commercial comparator (Lunit INSIGHT) | 0.7827 |

**The gap to the commercial system narrows from 0.108 to 0.077.** This is the largest single
improvement found, and unlike the negative-composition result it has a mechanism that was
predicted in advance rather than discovered by search.

## Why the control matters

The `mean` aggregator is mathematically what global pooling does, so it is the clean
internal control: `max` beating `mean` by 0.0602 under identical training cannot be an
artifact of the training setup, because both share it.

Note separately that this MIL training loop is **weaker** than the production one — its mean
aggregator reaches 0.7764 on TBX11K development against the production model's 0.9889, since
it is a simpler full-batch Adam loop rather than the tuned `heads.train_seeds` path. So the
regional model achieves **better external transfer while fitting TBX11K worse** (0.9043
against 0.9889).

That pattern is now consistent across four separate findings in this project: whatever makes
a model fit TBX11K more closely makes it transfer less well. It is the same shape as the
border shortcut, the healthy-only negatives result, and the flat learning curve.

## What this does not do

**It does not make the system deployable.** 0.7058 against a commercial product's 0.7827,
on a cohort where that product is itself only moderate, is still short. The headline finding
is unchanged: benchmark 0.99 becomes screening 0.71 at best.

**It is not a tuned result.** The region grid (6×6), the aggregator family, and the optimiser
were set once and not searched. A proper sweep would likely find a little more, and the
production training path would likely add more still — the MIL loop here underfits.

## Recommended next steps, in order

1. **Port the MIL head into the production training path** (`heads.train_seeds`), since the
   current loop underfits by about 0.08 AUC on development. Cheap, and likely the largest
   remaining easy gain.
2. **Finer grids.** 6×6 was chosen for storage, not measured. 9×9 or 12×12 may localise
   better; the full 37×37 is affordable for a single configuration.
3. **Re-run every external site** under max-MIL — only Vietnam has been scored so far.
4. **Then revisit tiling at higher input resolution**, which addresses the same dilution
   problem from the other direction and is now better motivated.

## Caveats

- Exploratory; provisional split; `internal_test` untouched. Only Vietnam scored externally.
- Max pooling is sensitive to a single outlying region, which is a plausible failure mode on
  images with artefacts; top-3 is the safer choice at 0.6928 and the difference between them
  is within the intervals.
- Region embeddings are L2-normalised per region, matching the global convention.
- Changing the aggregator changes the score scale, so calibration and thresholds would have
  to be refitted; only AUCs are compared here.

Evidence: `artifacts/p2/regional/regional_mil.json`, via `scripts/p2_31_regional_mil.py`.
