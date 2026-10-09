# Lesion detection (D-FINE)

*Owner: P3. Reviewer: P4. Every number cites its source in square brackets; "TBD: waiting on ..." marks results
that do not exist yet.*

## Why D-FINE

The detector localises TB lesions on the lung crop and supplies the evidence for the link rule, which can move
a NOT TB case to REFER (methodology section 9.3). We use D-FINE, a recent DETR-family detector that predicts box
edges as refined probability distributions instead of single numbers. It is released under the Apache-2.0
licence, verified from the licence file at the pinned commit `956d170` [decisions/p3_licences.json]. That
replaces the copyleft detector named in the proposal [README.md, section 8]. The comparison detector RT-DETRv2 (an extension) is also
Apache-2.0 [decisions/p3_licences.json].

## Task set-up

- **Single class.** All radiologist boxes form one class, `tb_lesion`. The active / latent / combined labels are
  kept only for the stratified analysis (methodology section 7).
- **Training data.** The training split has about 540 TB images, all with radiologist boxes, out of about 5,940
  training images [docs/methodology.md, section 1.2]. Non-TB images are included as negatives with no boxes.
  TB images without boxes ("uncertain TB") are excluded, because they contain disease but have no box to learn
  from [src/p3_det/data/dataset.py]. Exact counts after deduplication: TBD: waiting on
  decisions/p1_splits.json (D01) and the run's `train_data` record in run_meta.json.
- **Input.** Each image is cropped to the lung crop (margin D09, CLAHE D08) and resized to the input size. Boxes
  are shifted, clipped and dropped if fully outside the crop. Boxes map back to full-image 512 coordinates
  exactly; through the pixels the round-trip error is under 1 px (tested) [tests/p3_det/test_det_data.py].
- **Augmentation.** Mild scale and shift plus brightness and contrast only. Augmentation never removes a box. No
  flips, because a mirrored chest X-ray is anatomically implausible [src/p3_det/data/augment.py].
- **Pretrained start.** D-FINE-S starts from the official COCO-only checkpoint, loaded where parameter shapes
  match. The COCO class head does not fit one class and is re-initialised [src/p3_det/configs/dfine_512.yaml].
  The SHA-256 of the starting weights is recorded in each run's run_meta.json: TBD: waiting on
  artifacts/p3/det/runs/*/run_meta.json.
- **Training.** Kaggle T4 GPU with mixed precision (AMP), checkpointing to Drive every epoch so a killed session
  loses at most one epoch [docs/p3_kaggle_runbook.md]. Both input sizes use the identical configuration and the
  same seed, 0 [src/p3_det/configs/dfine_512.yaml; src/p3_det/configs/dfine_640.yaml]. A test enforces that the
  two configs differ only in input size [tests/p3_det/test_select_size.py].
  - Batch size and peak memory per input size: TBD: waiting on docs/p3_dfine_feasibility.md (P3-01 T4 run).
  - Number of epochs: TBD: waiting on the final config and decisions/p3_det_checkpoint.json (D17).
- **Preprocessing.** Provisional runs (margin 0.10, CLAHE off) [src/p3_det/data/config.py] were only used to start development. D16–D18
  rest on runs that used the locked D08/D09 preprocessing [decisions/p3_det_input_size.json, criterion].

## Input size: 512 vs 640 (D16)

TBX11K images are 512 × 512 [docs/methodology.md, section 7], so 640 cannot add detail; it only enlarges small lesions relative to the feature
maps (methodology section 7). The rule was pre-registered before the comparison
[decisions/p3_det_input_size.json]:
- compare development AP at IoU 0.50 at each run's final epoch, using the EMA weights (the moving-average
  weights D-FINE keeps alongside training);
- the higher value wins, and an exact tie keeps 512 [decisions/p3_det_input_size.json, criterion].

| Input size | Development AP50 | 95% CI | n images (TB / non-TB) | n boxes | Source |
|---|---|---|---|---|---|
| 512 | TBD | TBD | TBD | TBD | TBD: waiting on decisions/p3_det_input_size.json (D16) |
| 640 | TBD | TBD | TBD | TBD | TBD: waiting on decisions/p3_det_input_size.json (D16) |
| 640 − 512 (paired) | TBD | TBD | TBD | | TBD: waiting on artifacts/p3/det/select_size/d16_comparison.json |

Chosen input size: TBD: waiting on decisions/p3_det_input_size.json (D16).

AP50 is computed as in pycocotools. Our implementation matches pycocotools to within 1e-3 on three test
fixtures [tests/p3_det/test_det_metrics.py]. It is evaluated in full-image 512 coordinates against the original
boxes (D21). CIs are percentile bootstrap over images, stratified TB / non-TB [src/p3_det/metrics/stats.py].

## Training length and confidence threshold (D17, D18)

- **Checkpoint (D17).** The epoch with the highest development AP50; the earliest wins on ties. The checkpoint is
  identified by its SHA-256 [decisions/p3_det_checkpoint.json, criterion].
  - Chosen epoch, development AP50 with its CI, and checkpoint SHA-256: TBD: waiting on
    decisions/p3_det_checkpoint.json (D17).
- **Confidence threshold (D18), pre-registered rule:** "the lowest confidence threshold such that at most 5% of
  development non-TB images have any detection box above it" [decisions/p3_det_conf_threshold.json]. It is searched on a 0.001 grid with `score >= t`
  [decisions/p3_det_conf_threshold.json, criterion]. The guide's acknowledgement is required before locking.
  - Threshold, share of non-TB images with a box (Wilson CI, n) and TB image-level lesion sensitivity at the
    threshold: TBD: waiting on decisions/p3_det_conf_threshold.json (D18, with guide_ack).

## Results on the sealed internal test (from P4's tables)

Detection was run once on the internal test, under P4's gate token (P3-13). The tables are produced by P4.

| Measure | Value | 95% CI | n (TB / non-TB images, boxes) | Source |
|---|---|---|---|---|
| AP at IoU 0.50 | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection.csv |
| Image-level lesion sensitivity (at D18) | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection.csv |
| FROC lesion sensitivity @ 1 FP/image | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection.csv |
| FROC lesion sensitivity @ 2 FP/image | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection.csv |
| FROC lesion sensitivity @ 4 FP/image | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection.csv |

**FROC definition.** Lesion sensitivity is the best value reachable while averaging at most k wrong boxes per
image, counted over all images (TB and non-TB), with no interpolation. A second box on an already-found lesion
counts as a false positive [src/p3_det/metrics/detection.py].

### Subtype-stratified results (secondary)

Each subtype row uses that subtype's TB images plus all non-TB images, so false positives on healthy and
sick-non-TB images still count [src/p3_det/metrics/detection.py].

| Subtype | AP50 (95% CI) | Image-level sensitivity (95% CI) | FROC @ 1 / 2 / 4 FP | n TB images | Source |
|---|---|---|---|---|---|
| Active TB | TBD | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection_subtype.csv |
| Latent TB | TBD | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection_subtype.csv |
| Active + latent | TBD | TBD | TBD | TBD | TBD: waiting on artifacts/p4/tables/t12_4_detection_subtype.csv |

## Latent TB caveat

**The latent and combined results are not reliable.** The whole TBX11K dataset contains only 212 latent-TB
images and 54 images with both active and latent findings [docs/methodology.md, section 1.1 table]. Our sealed
internal test is half of the official validation split [docs/methodology.md, section 1.2], and that whole split
holds only 36 latent and 7 combined images [docs/methodology.md, section 1.1 table]. With so few images the
confidence intervals are very wide. Our metrics
code labels every latent row "not reliable (very small n)" and any subtype row with fewer than 30 TB images as
"small n" [src/p3_det/metrics/detection.py]. These rows are reported for completeness only; no conclusion about
latent TB should be drawn from them.
