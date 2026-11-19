# P3 data dependency update - 18 November 2026

Source: Tharun/P1's message supplied by the user on this date. These are reported delivery
statuses, not an independent audit of P1's machine or repository. This is the TB project,
not the separate Pathya contract review.

## P1 delivery status

| Input | P1's reported status | Dependency |
|---|---|---|
| Definitive C01 v1 | Not produced | C01 v0 (P1-04), then Atul's C17 near-duplicate candidates, then P1-07 duplicate resolution |
| C04 processed 512x512 images | Not produced | P1-05 ingest (P1's next task), then P1-06a image store |
| C05 lung ROI table | Not produced | Final split, segmentation and sanity checks, P1-08 through P1-10 |
| C06 cropped COCO annotations | Not produced | P1-10 and the corresponding crop geometry |
| Original dataset | Available to P1 | 8,400 official train/validation images; 3,000 challenge images have no public labels |
| Original lesion annotations | Available to P1 | TBX11K_train/val/trainval and only_tb COCO files, plus 800 XML files |

Original annotations are useful source material, but they are not the project's C01/C04/C05/C06
handoff. They must not be renamed or labelled as definitive contract outputs. No download link
or local path for those originals was included in the message.

## What P3 can continue

- Continue synthetic adapter, geometry, augmentation, metrics, checkpoint/resume, runtime,
  PDF/logging and integration tests. Existing synthetic tests remain software evidence.
- Run T4 feasibility on synthetic COCO once a cloud GPU session is available; this does not
  require P1's finished real-data contracts.
- Once the official **training-only** images and matching original annotations are shared,
  inspect their actual schema and prepare an isolated provisional data path under H4:
  margin 0.10, CLAHE off. Do not fabricate lung ROIs or a definitive patient/split manifest.
- Preserve original full-image boxes, including boxes outside a crop, and subtype metadata
  through any conversion. Converting COCO/XML requires the actual files and verified geometry.

## Provisional boundaries

- Do not train on official validation images: they will feed future calibration/internal test.
- Do not use challenge-test images for training or tuning; their public labels are unavailable.
- Provisional work on official training data is exploratory and may overlap a future
  development partition. It is not final development performance or locked selection evidence.
- Keep provisional inputs, decisions and run directories separate from final artifacts.
  Do not overwrite or force D16-D20 results/locks to make readiness checks pass.
- Final detector training/selection, C10/C11 delivery and accepted metrics require the
  definitive C01 and actual C04/C05/C06 with the reviewed locked preprocessing. Re-run both
  detector sizes as required by H4; do not relabel a throwaway checkpoint as the final model.
- Decision-history resolution, the revised regional D19 review and hardware measurements
  remain separate prerequisites; this message does not resolve them.

## Next handoff to request

Ask P1 for the official training-only images and matching COCO JSON (or a local path) now,
and C01 v0 when published. Request the definitive outputs as they become available instead
of requesting a link to files P1 has confirmed do not yet exist. GPU access is still needed
for T4 feasibility/training and RTX 4050 acceptance measurements.

P3's current completion status remains: implementation progressed and tested, final real-data
acceptance blocked upstream. No new training, evaluation or decision lock was performed in
response to this status message.
