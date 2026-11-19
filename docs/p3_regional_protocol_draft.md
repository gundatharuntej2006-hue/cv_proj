# Regional explainability migration - draft for P2/P4/guide review

Status: draft only. This does not amend or lock D19 and contains no measured result.

The original work plan specifies a cropped CLS-based classifier with calibrated probabilities.
P2's later service uses lungs-only full-frame features, a 12x12 regional head and development
logit operating points. The original Grad-CAM protocol cannot be claimed unchanged.

## Proposed reproducible target

- Use the pinned encoder and image processor from the actual served v2/v3 bundle.
- Prepare the same full512 lungs-only image as `pipeline_v2.lungs_only`; do not apply the
  detector's crop to classifier input. Detector preprocessing still follows D08/D09 separately.
- Reproduce the served region pooling, per-region normalisation and feature precision.
- Explain the **mean of each seed's maximum region logit**. This generally differs from the
  maximum of the seed-mean displayed region map. Report any per-seed argmax disagreement.
- Select and document the inspected transformer target layer, token layout and full512 mapping.
  `p3_xai.regional` supplies a differentiable adapter, not validation of that choice.
- Verify actual development-image scores against the served bundle before interpreting a CAM.
- Apply an agreed randomisation sanity rule to the real classifier on 20 seeded development
  images (10 TB/10 non-TB); report signed and absolute correlations and constant-map counts.
- Measure peak VRAM on the RTX 4050. Compare development localisation with all original
  annotations and report n/CIs. Keep the 40-case guide review.

## Reporting/contract boundary

- The v2 PDF is a separate research report format. Its score is a logit; it is not a C14
  calibrated-probability report and does not invent TB/REFER/NOT_TB categories.
- The detector supplies independent lesion evidence and additional-review flags to v2.
  It never changes P2's logits, operating-point flags or fusion coefficients.
- Predicted-box/displayed-cell geometric agreement is explicitly exploratory. It is distinct
  from ground-truth lesion accuracy and P2's published patch/cell overlap experiment.
- Keep R3's original `make_pdf`/`log_case` C14/C15 interfaces for P4's calibrated pipeline.
  Any change to those shared contracts follows the team's actual review process.

Record the agreed target, preprocessing, layer, sanity rule, evidence paths and protocol version
in the reviewed decision record. Resolve conflicting legacy D19 history first; preserve the
existing historical evidence and dates.
