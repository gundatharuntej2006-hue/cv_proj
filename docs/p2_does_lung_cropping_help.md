# Does lung cropping help? Measured answer: no, on either axis

**Owner:** Person 2 · **Date:** 2026-10-14 · **Answers:** methodology Section 14's
"Does lung cropping help?" · **Status:** provisional split, real C04/C05 crops

## Summary

The pipeline spends a PSPNet segmentation forward pass per image, a sanity check, a
fallback rule and a whole decision (D09) on cropping to the lung box. Measured on real
crops, that machinery buys **nothing**:

- **Accuracy, in-distribution:** cropping is neutral at best and significantly worse at
  worst. The plain full image has the highest AUC of every variant tested.
- **Cross-site transfer:** cropping changes the site effect by nothing at all.

## 1. Accuracy: paired bootstrap against the full image

Frozen RAD-DINO + linear head, 5 seeds, development split, 2,000 paired bootstrap
resamples. ΔAUC is variant minus `full_clahe0`; a CI crossing zero means indistinguishable.

| variant | ΔAUC vs full image | 95% CI | verdict |
|---|---|---|---|
| crop_m0.00_clahe0 (tight) | −0.0019 | [−0.0045, −0.0000] | **significantly worse** |
| crop_m0.10_clahe0 | −0.0009 | [−0.0027, +0.0003] | no difference |
| crop_m0.15_clahe0 | −0.0006 | [−0.0019, +0.0003] | no difference |
| crop_m0.10_clahe1 (CLAHE) | −0.0057 | [−0.0144, −0.0005] | **significantly worse** |

No variant beats the uncropped image. The tight crop and CLAHE both measurably hurt.

Effect sizes are tiny throughout (|ΔAUC| < 0.006), so the honest phrasing is **"cropping
provides no measurable benefit"**, not "cropping is harmful".

## 2. Transfer: does cropping remove the site signature?

A reasonable hypothesis, given that a kNN gate calibrated on TBX11K rejects ~99% of a new
hospital's normal chest X-rays: the site signature might live in the borders, collimation
edges and burned-in markers, all of which a lung crop discards.

Tested by segmenting and cropping 150 CheXpert frontals the same way (150/150 passed the
sanity check) and re-measuring:

| | site separability (AUROC) | new-site false-rejection | median distance to bank |
|---|---|---|---|
| uncropped full image | 0.9986 | 99.3% | 0.769 |
| cropped to lungs, m0.10 | 0.9987 | 99.3% | 0.769 |

**Identical.** The hypothesis is wrong.

That is the more informative outcome: the site signature is **not** a peripheral artifact.
It survives cropping to pure lung field, so it is in how the scanner renders the tissue
itself — the reconstruction and post-processing, not the frame around it. This explains why
none of the obvious corrections work: mean-removal (rank-1), whitening, and now cropping
all leave the gate's cross-site failure untouched (see `p2_ood_gate_finding.md`).

## 3. What this means for the project

**Recommendation: do not justify the lung-segmentation stage on classification grounds.**
Two defensible options:

1. **Keep it, report honestly.** State in Section 14 that cropping was tested and gave no
   measurable benefit, and that it is retained for the detector and for interpretability
   (lesion boxes and heatmaps are easier to read on a lung crop). This is the conservative
   choice and preserves P3's pipeline.
2. **Simplify.** Drop cropping from the classification path and use the full image, which
   is at least as accurate and removes a segmentation dependency, a sanity-check failure
   mode and a decision (D09) from the critical path.

Either way the *reported* reason must match the measurement. The methodology currently
assumes cropping helps; it does not.

**Untested here:** whether cropping helps the **detector** (P3). Lesion localisation works
in crop coordinates and may well benefit, and the comparison above says nothing about it.
That is a separate experiment and should not be assumed either way.

## Caveats

- Provisional split (P1's C01 not yet available); 59 TB cases in development, so the
  bootstrap CIs are wide and only large effects would be detected. The conclusion is
  "no benefit detected", not "proven identical".
- One external site (CheXpert) for the transfer test.
- In-distribution TB classification only.

Evidence: `artifacts/p2/dryrun/real_d08_d09_sweep.json`,
`artifacts/p2/dryrun/crop_vs_full_bootstrap.json`.
