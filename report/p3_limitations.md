# Limitations and honest negatives (P3 components)

*Owner: P3. Reviewer: P4.*

- **Few boxed training images.** The detector learns from about 540 TB training images with radiologist boxes,
  out of about 5,940 training images [docs/methodology.md, section 1.2]. Lesion detection is therefore
  data-poor. Model selection on development rests on only about 60 TB images [docs/methodology.md, section 1.2],
  so development CIs are wide.
- **Latent TB.** Only 212 latent and 54 combined active-and-latent images exist in the whole dataset
  [docs/methodology.md, section 1.1 table]. Subtype results for these classes are not reliable and are flagged as
  such [src/p3_det/metrics/detection.py].
- **Detector contingency.** If the chosen detector's development AP50 is below 0.05, D16 is not locked, tuning
  stops and the contingency is raised with P4 [decisions/p3_det_input_size.json, criterion]. The contingency
  means: report the negative result with its CI, disable the link rule, and use Grad-CAM as the main
  localisation evidence. Whether it was triggered: TBD: waiting on decisions/p3_det_input_size.json
  (value.contingency.triggered). If it was, we state it plainly here, with the AP50 and its CI. This rule is not
  written in the methodology or README; it was agreed in the P3-06 task and must be confirmed by P4.
- **Incomplete boxes.** "All original boxes" (D21) cannot yet be supplied for boxes lying fully outside the lung
  crop. The box file used drops them, so detection metrics may slightly over-state performance on those images
  [docs/p3_contract_notes.md, item 11]. TBD: waiting on P1's uncropped box file or confirmation that the gap is
  closed.
- **Explanations are not findings.** Localisation agreement is measured against boxes from one radiologist that
  are not exhaustive. High agreement does not prove clinical correctness, and low agreement does not prove error.
- **Sanity check scope.** The pre-registered randomisation check uses the mean absolute Spearman correlation
  [decisions/p3_gradcam_config.json, criterion], because a signed rule would pass an inverted heatmap. It tests
  dependence on the weights, not clinical correctness.
- **Non-determinism.** Runs use a fixed seed, recorded per run, but GPU kernels are not bit-for-bit deterministic,
  so retraining may give slightly different numbers.
- **Latency on our hardware only.** E6 was measured on one laptop GPU and one CPU-only set-up; other hardware will
  differ.
