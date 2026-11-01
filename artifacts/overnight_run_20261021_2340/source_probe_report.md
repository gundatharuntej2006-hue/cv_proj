# Source-prediction probe: can the frozen features tell which machine took the picture?

**Read-only diagnostic. Nothing here was fed back into any model, feature or threshold.**

Metrics: `source_probe_metrics.json` · Script: `scripts/source_probe.py`

## Why this probe and not another

The project's central finding is that performance collapses between a curated benchmark and
a real screening cohort, and that interventions improving the benchmark fit make transfer
worse. One candidate mechanism is that the frozen representation carries acquisition
information. This probe tests whether that information is *there*. It deliberately does not
test whether the tuberculosis head *uses* it, because the two are different questions and
only the first is answerable without fitting something new.

## Leakage control

| control | what was done |
|---|---|
| partition | fitted on the Vietnam **TUNE** half only, 1,352 cases. The REPORT half is not opened by the script. |
| target | `rad_system`, not tuberculosis. No outcome label enters the probe. |
| patient grouping | Vietnam is one image per patient (2,708 unique `pid` for 2,708 rows, asserted in code), so stratified 5-fold CV *is* patient-grouped. |
| feedback | the probe's output is not used to select features, a classifier, a threshold or a configuration. |

## Result

Multinomial logistic regression on standardised 768-d globally pooled frozen RAD-DINO
embeddings, C = 0.1, 5-fold stratified CV, seed 17.

| quantity | value |
|---|---|
| **balanced accuracy** | **0.9737** |
| macro F1 | 0.9772 |
| majority-class accuracy | 0.4919 |
| balanced accuracy, scanner labels permuted (10 repeats) | 0.3381 mean · 0.3515 max |

Class counts in TUNE: rad_system 1 → 204, 2 → 665, 3 → 483.

Confusion matrix, rows true 1/2/3, columns predicted 1/2/3:

```
[[195,   6,   3]
 [  2, 657,   6]
 [  1,  10, 472]]
```

Twenty-eight of 1,352 images are assigned to the wrong machine. A linear read-out of a
representation that was never trained on this cohort identifies the acquisition system
almost perfectly.

## Is scanner just a stand-in for disease?

No, not in this cohort. Scanner and Xpert result are close to independent:

| scanner | n | TB | prevalence |
|---|---|---|---|
| 1 | 408 | 44 | 0.1078 |
| 2 | 1,332 | 151 | 0.1134 |
| 3 | 968 | 85 | 0.0878 |

χ² = 4.051, df = 2, **p = 0.132**, Cramér's V = **0.039**.

So the representation encodes, with near-perfect fidelity, a variable that carries almost no
information about the outcome. That is the definition of an available nuisance direction.

## What this does and does not establish

**Established.** The frozen RAD-DINO representation retains acquisition-source information
at a level a linear model recovers almost perfectly, on a cohort the backbone never saw.
Any method that claims source robustness has a real and strong signal to work against, and
the project now has a measured baseline for it.

**Not established, and explicitly disclaimed.** This does **not** show the tuberculosis
classifier exploits that information. A representation can encode a nuisance variable that
the downstream head ignores entirely. Demonstrating *use* would require an intervention —
removing the source subspace and measuring what happens to TB discrimination — which is
pre-registered in `PRIMARY_EXPERIMENT_PREREGISTRATION.md` §4 and was not run (see that
document for the blocker).

Three scanners in one country is also not a sample of scanners in general. Nothing here
supports a population-level claim about acquisition hardware.

## The companion observation that is operationally more important

The probe shows source is *encoded*. The score-distribution audit
(`score_distribution_report.md`) shows source is *acted on* at the operating point: the
median score of a **healthy** person differs by 3.07 logits between rad_system 2 and
rad_system 3, which is larger than the entire interquartile range within either group. That
second finding does not depend on this probe and is the one with an immediate fix.
