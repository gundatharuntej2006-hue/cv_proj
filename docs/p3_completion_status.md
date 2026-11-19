# Role 3 completion status — 2026-11-18

P1 has confirmed that definitive C01/C04/C05/C06 do not yet exist. See the
[18 November dependency update](p3_p1_dependency_update_20261118.md) for the delivery sequence
and the permitted provisional-development scope under H4.

The latest implementation changes, verification and outstanding acceptance inputs are in
[the 2026-11-17 progress record](p3_implementation_progress_20261117.md).
The October final verdict and blocker lists are historical.
Implementation fixes do not constitute trained-model results or clinical validation.
Verification evidence is in [the fix verification record](p3_fix_verification_20261103.md).

## Priority order and remaining acceptance evidence

| Priority | Work | Implemented here | Still required |
|---|---|---|---|
| 1 | Training prerequisites | Read-only `python -m p3_ops.preflight`; reproducible pinned checkout via `python -m p3_det.setup --fetch`; existing feasibility/config-fill/training tools retained | P1 C01/C04/C05/C06 files and original full-image boxes; measured Kaggle T4 feasibility; actual 512/640 training runs |
| 2 | Detector selection | Existing D16/D17/D18 pre-registered rules retained; real loader still verifies locks and checkpoint SHA | Measured AP50 comparison, checkpoint, threshold results, guide acknowledgement, owner locks |
| 3 | Runtime integration | Fixed full512 input, separate box/score arrays, locked threshold, missing-detector handling, saved heatmaps and case-specific overlays in P4 orchestration | Real checkpoint and real model/data smoke run; team review of shared P4 changes |
| 4 | P2 v2 bridge | Opt-in lesion evidence and overlays, additive referral flag, existing P1 preprocessing functions reused, no classifier-score changes | Run with Atul's merged checkout and frozen bundle; verify D08/D09 input equivalence on real images |
| 5 | Explainability | Frozen encoders now support input gradients; differentiable regional v2/v3 adapter reproduces mean-of-seed-max score algebra and float16 feature rounding in tests | Revised D19 approval for regional model, actual score reproduction, randomisation sanity, VRAM, localisation tables, guide review |
| 6 | Shared contracts | Adapter supports actual P4 names; C10/C15 validated by P4, C11 by strict producer validator with explicit provenance | Real dataset/model exports |
| 7 | PDF/logging | C14 Pydantic objects supported; current shared disclaimer used; unavailable detector described honestly; synthetic legacy exporter produces an actual PDF | Real end-to-end case reports/logs and dashboard visual review |
| 8 | Sealed exports and E6 | Guarded one-shot C10/C11 runner, permanent token claims, failed-run record; nested Pydantic timings supported; unconfigured/mock default E6 refused | All 23 owner decisions locked, real P2 provider, existing P4 token, authorised one-shot runs, full GPU/CPU measurements |
| 9 | Final handover | Report text no longer implies an unperformed sealed evaluation or guide session | Real report numbers, signed guide minutes, Kaggle kill/resume record, demo recording |

**No D16–D20 lock or guide acknowledgement was fabricated. No sealed evaluation was run.**

**Additional research blocker:** D16–D19 have conflicting older locked records in the team history.
See [the decision-history review](p3_decision_history_conflict.md). Result tools now refuse that conflict;
it must be resolved explicitly before new results can be recorded as pre-registered evidence.

## Current environment

- P3 environment reports PyTorch 2.14.1+cpu and CUDA unavailable. `nvidia-smi` is not on PATH.
- Data splits, processed manifest, lung ROI and original box files are absent from this checkout.
- D08/D09 files are marked locked; their evidence and definitive split still require team confirmation.
- Training configurations still intentionally contain null epochs/batch size until measured T4 evidence is supplied.
- D-FINE local checkout was verified at `956d1709314c2c6a4df6f34de232054578a7449f` with no tracked modifications.
- The existing D-FINE checkout is preserved and is now registered as a pinned submodule.
  The gitlink and `.gitmodules` are staged; source changes have not been committed/pushed.
- Atul's current branch was fetched for interface inspection; it was not merged into Ritika's branch.
- The missing frozen methodology/proposal sources were restored byte-for-byte from local baseline `main`
  so existing report citations resolve. Their content was not revised.

## Run the remaining work

Use a Python environment with base and P3 requirements installed and `PYTHONPATH=src`.

1. Run `python -m p3_ops.preflight`. This is read-only and returns 2 while prerequisites are missing.
2. Supply P1's definitive C01, C04, C05, C06 and uncropped full-image annotation files in their contract paths.
3. On Kaggle T4, run `notebooks/p3/p3_01_dfine_feasibility.ipynb`, then the deliberate-kill/resume notebook.
4. Save the feasibility JSON; use `python -m p3_det.fill_training_config --json <results.json> --hours-per-session <budget>` to review proposed settings. Add `--write` only with actual measured data.
5. Train both configurations through `python -m p3_det.train --config dfine_512.yaml` and `dfine_640.yaml` under identical schedules. Follow the existing selection commands in the runbook; retain their pre-registration checks.
6. Select D16/D17/D18, obtain guide acknowledgement, and have owners lock the decisions. Verify the downloaded checkpoint SHA. Export development/calibration C10.
7. Integrate with Atul's checkout using `python -m p3_ops.serve_v2 --p2-root <checkout>`. The launcher requires a real locked detector and reuses the P1 ROI/CLAHE functions and P2 segmenter. Optional `--input-provider module:function` must return `(prepared_full512_uint8, locked_crop_box)`.
8. Agree the regional explainability protocol with P2/P4/guide before replacing the original CLS-based D19. `p3_xai.regional` is an implementation adapter, not an approved protocol change. Do not report its synthetic score tests as real-model reproduction.
9. After owner approval, reviewed code/decision commits and all gate prerequisites, use `python -m p3_ops.sealed_run --split internal_test --model-id <registered-model>`. This consumes an existing P4 token; it never issues one. A failure permanently reserves that attempt. Challenge export requires the prior P3 internal run to be complete. The runner currently supports TBX11K C01 and the original R3 crop-based explanation path; regional-model sealed explanation needs the approved D19 integration first.
10. Configure all real R4 stages, then run E6 on 103 development inputs (3 warm-up, 100 timed). The default unconfigured orchestrator is deliberately rejected by the E6 driver.

The supplied work plan assigns the backup demo recording to P1, not P3. Existing demo-script
drafts are preserved; they are not counted as a missing P3-17 task.

## Scientific reporting

- P2's 95.9% region-cell overlap on 49 development images is not D-FINE accuracy or Grad-CAM accuracy.
- P2 v2 returns logits and operating-point flags. Its detector bridge adds an independent review flag, not a calibrated TB/NOT_TB output.
- The image score is the mean of each seed's maximum region logit. In general, it is not the maximum of the seed-mean region map. A displayed mean-map argmax must not be described as explaining the complete ensemble exactly when seed argmax locations disagree.
- The published 95.9% result uses a specific preprocessing/patch-overlap definition. Plain geometric overlap between predicted detector boxes and a display cell is a different exploratory measure.
- Keep detector boxes separate from classifier training; the P2 box-supervision experiment was rejected.
- The internal test had a previous P2 rehearsal exposure and Vietnam REPORT is exposed. A new runner cannot turn either into an untouched cohort.
- RT-DETRv2 comparison and lesion segmentation remain optional extensions after the core work.

## Review scope

The user's request to fix all Role 3 gaps includes the necessary P3 integration changes in
`src/p4_pipeline/orchestrator.py`, the legacy demo regression, and isolated P4 gate/C16 validation tests.
The gate tests now use temporary synthetic decisions and assert refusal of unlocked decisions; no real
decision locks were changed. These shared changes need Kunal's
review before merging. No files were committed, pushed, published, or sent to teammates.
