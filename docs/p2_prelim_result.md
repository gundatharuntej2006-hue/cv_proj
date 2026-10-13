# Preliminary TB classification result (frozen RAD-DINO + linear head)

**Owner:** Person 2 · **Date:** 2026-10-13 · **Status:** preliminary, in-distribution only
**Reproduce:** the embeddings + `p2_cls.heads`; numbers in `artifacts/p2/scores/p2_prelim_tbx11k_result.json`

## The number

Frozen RAD-DINO embeddings (full-image variant) + a linear binary head, 5 seeds,
early-stopped on TBX11K val:

| metric | value |
|---|---|
| val AUC (TB vs not-TB) | **0.9947** |
| sensitivity at 70% specificity | **1.000** (Wilson CI 0.981–1.000, n_pos = 200) |
| seed-to-seed AUC spread | 0.0000 |

This validates the architecture: even with the backbone **frozen** and only a linear head
trained, RAD-DINO separates TB on TBX11K almost perfectly. It is consistent with published
TBX11K classification numbers (~0.95–0.99 AUC).

## Why this number must not be over-read

Three reasons it is an **upper bound that will not survive deployment**, all of which the
project's own findings predict:

1. **It is in-distribution.** Train and val are both TBX11K — the same source, scanner and
   preprocessing. We have already shown (docs/p2_ood_gate_finding.md) that TBX11K occupies
   an artificially tight region of embedding space (self-distance 0.39 vs a real site's
   0.78). A near-perfect in-distribution number is exactly what you expect from a
   homogeneous benchmark and says little about a new hospital.

2. **Shortcut risk.** A frozen foundation model separating TB at 0.99 from a single source
   is the classic setup for source/scanner shortcut learning — the model may be reading the
   acquisition signature, not the pathology. This is the same failure mode seen elsewhere
   (bovine-breed models scoring ~98% off the backdrop). The only honest test is
   source-held-out, which needs a genuinely independent site (TB Portals), not yet in hand.

3. **The zero seed spread is a tell, not a triumph.** Five seeds giving identical AUC means
   the linear head on frozen embeddings converges to one solution — the signal is trivially
   linearly separable in this embedding space. Trivial separability in-distribution is
   precisely what fails to transfer.

## Leakage impact (quantified)

Removing the 50 val images that duplicate a train image (the train→val leak) changes val
AUC from **0.9947 to 0.9944 — an inflation of 0.0002.** So on this metric the leakage is
negligible, *because the task is near-saturated in-distribution*: when a model scores 0.99,
removing 50 of 1,800 images cannot move it much.

This does **not** make the duplicates unimportant. They matter for reproducibility and
principle (you should never evaluate on training images), and they would matter more on a
harder metric or a weaker model. But it is honest to report that the measured AUC inflation
here is tiny, rather than imply the leak invalidates the benchmark's headline number.

## What this means for the project

- The frozen-RAD-DINO design is sound; the architecture is not the risk.
- The risk is entirely cross-site generalisation, which is the paper's thesis and which we
  cannot measure until an independent TB-labelled site arrives (TB Portals).
- Caveats on this specific run: it uses the **full-image** variant, not the primary lung-crop
  variant (D08/D09 not yet run on real data), and **stand-in preprocessing**, not Person 1's
  C04. Re-run on the primary crop variant from C04 before quoting in the report.
