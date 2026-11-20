# P3 Detector Preprocessing Protocol

**Status: draft for team review, 20 November 2026.** P1 and P2 agree that
the detector needs a separate protocol. Kunal confirmed the selection rule
and comparison order for P4. The bootstrap settings, verified input handoff,
and versioned decision record are pending.

## Scope

This protocol governs the D-FINE lesion detector.
[D08](../decisions/p2_clahe.json) and [D09](../decisions/p2_crop_margin.json)
concern classification. Atul's [provisional classifier sweep](https://github.com/gundatharuntej2006-hue/cv_proj/blob/atul/docs/p2_d08_d09_recommendation.md)
describes the previous split. Atul's proposed records for the C01 v0 rerun
specify CLAHE off and a 0.00 lung-crop margin, with both records unlocked
and pending review. A 0.00 margin still uses the lung ROI. The frozen classifier
bundles use full images, and changing their serving preprocessing is a separate
decision. The detector requires its own evidence.

## Detector comparison

Use one D-FINE-S configuration at 512 for this study. The 512 setting is fixed
for the preprocessing comparison. D16 will compare 512 and 640 afterward.

| Candidate | View | CLAHE |
| --- | --- | --- |
| Baseline | Full 512 × 512 image | Off |
| Full-image CLAHE | Full 512 × 512 image | On |
| Lung crop | Lung ROI with margin 0.10 | Off |
| Lung crop with CLAHE | Lung ROI with margin 0.10 | On |

The 0.10 study margin is the existing provisional H4 setting. P1 should
confirm how the CLAHE image is generated and the ROI fallback policy.
For a crop candidate, apply the same P1 geometry and fallback
rule consistently across train and development. Record the CLAHE settings and
whether it is applied before crop generation.

Keep the same train/development split, starting weights, input size, seed,
optimizer, epochs, batch size, augmentation, and model evaluation settings.
Verify the exact C01 split file against P1's published SHA-256 before every
run. The detector runner holds the verified split in memory for both dataset
loaders. The attached P2 sweep is a reference for split verification and
paired comparisons. Its classifier head and scoring rules stay with P2.
Select the training schedule from measured T4 feasibility. Record the eligible
training counts and exclusions in each arm, including any positive images whose
boxes are outside the crop or too small after clipping.

## Selection rule agreed with P4

1. Score the final-epoch EMA model for every candidate on the same reviewed
   development images. Map detections back to full 512 × 512 coordinates and
   score against all original development boxes, including out-of-crop lesions.
2. Start with **full image without CLAHE** as the current choice. Visit
   **full image with CLAHE**, **0.10 lung crop without CLAHE**, and
   **0.10 lung crop with CLAHE** in that order. Compare each candidate with
   the current choice. If a candidate replaces it, use that candidate as
   the reference for the next comparison.
3. Replace the current choice only when **both** conditions hold for
   `AP50(candidate) - AP50(current)`:

   - The unrounded point estimate is **at least 0.01 absolute AP50**.
   - The lower bound of the paired 95% confidence interval is **strictly above zero**.

   Otherwise, keep the current, simpler choice. An interval whose lower
   endpoint equals zero does not qualify.
4. In each paired bootstrap replicate, resample development image IDs with
   replacement within TB and non-TB strata. Use the same resampled IDs for
   both candidates. Recompute both AP50 values and subtract them. Take the
   2.5th and 97.5th percentiles of those differences. P4 proposes 2,000
   replicates and seed 42, subject to D21 approval.
5. Record all four AP50 values, the three comparisons and their intervals,
   development image and lesion counts, eligible training counts, the bootstrap
   count and seed, and each promotion decision.
6. Record the reviewed setting with the protocol commit, input checksums, run
   metadata, and result artifacts. Apply it unchanged to both D16 sizes.

After all candidates finish, generate the paired evidence report with the
D21-approved bootstrap parameters. Supply exactly three `--candidate` flags.

```sh
python -m p3_det.compare_preprocessing --baseline <full-off-run> --candidate <full-on-run> --candidate <crop-off-run> --candidate <crop-on-run> --resamples <count> --seed <seed> --out <new-report.json>
```

The report checks that the split, manifest, original development
annotations, detector configuration, and initialization are comparable before
calculating AP50 and paired intervals in the fixed order. It records the final
choice for review. Formal approval remains with the decision owners.

Only the train and reviewed development splits are used for this study.
C01 v0 results remain provisional. Confirm how the study will be repeated on
or reconciled to C01 v1 before using its outcome for final training.

## Data handoff and sign-off

- **P1:** Share the private path and checksums for the C01 v0 split and C04 v0
  manifest and 8,400 non-CLAHE full images. Confirm which original train and
  development lesion annotations are complete, and provide the standardized
  full-image box files. Once the settings are frozen, provide the CLAHE image
  variant and the C05/C06 crop inputs for the agreed margin.
- **P4 and P3:** Confirm the bootstrap count and seed through D21, the reviewed
  split version, and the location of a separate P3-owned decision record.
  Agree on a versioned update for D16, whose existing criterion points to
  D08/D09, and keep the earlier decision history available for audit.
- **P3:** Verify the published C01 and C04 hashes and check the coverage and
  full-image geometry of the original annotations before running comparisons.
  Run the agreed comparison and share
  the evidence for review. The [provisional runner](p3_completion_status.md#detector-preprocessing-study)
  supports both views. Final training remains gated on approval and integration
  of the reviewed protocol.
