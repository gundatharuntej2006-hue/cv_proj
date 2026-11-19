# P3 implementation and remaining acceptance - 2026-11-17

Scope: Person 3, local `ritika` checkout. Existing user changes were preserved.
This record separates implemented software from measured research acceptance.
No detector training result, decision lock, guide acknowledgement or sealed run was fabricated.

## Completed in this implementation pass

- Registered the existing clean D-FINE checkout as a git submodule at
  `956d1709314c2c6a4df6f34de232054578a7449f`. `.gitmodules` and the gitlink are staged;
  no source commit or push was made.
- Added geometric agreement between thresholded predicted lesion boxes and P2's displayed
  12x12 seed-mean argmax region. Positive-area overlap excludes edge-only contact. The result
  distinguishes zero boxes from an unavailable detector, and explicitly does not claim lesion
  accuracy, exact ensemble attribution or reproduction of P2's published 95.9% measurement.
- Added lesion confidence labels and moved the overlay caption outside the image, preserving
  lesions near the bottom edge.
- Added in-memory v2 PDF reports, cached downloads and screen download links. A cached report
  preserves the original analysis/fusion selection without rerunning inference. Opaque per-case
  keys prevent identical images analysed under different clinical inputs from overwriting reports.
  The bounded cache covers a complete 200-image batch and writes no uploads/reports to disk.
- The v2 PDF reports the screening logit, operating-point flags, detector provenance, referral
  evidence and exploratory overlap. It imports the shared disclaimer/scope text and does not
  populate C14 probability fields with logits. Rejected/error cases remain unscored and receive
  manual-review PDFs even if no classifier bundle is installed. Original filenames, raw error
  text, questionnaire answers and upload metadata are excluded from report text.
- Added explicit `--detector-disabled`, `--bundle` and `--preload` launch options. Disabled
  detectors produce unavailable evidence, never a fabricated zero count or a referral.
- Fixed development detector evaluation to use all original boxes, including lesions completely
  outside the crop. Locked evaluation refuses cropped-only ground truth. Training still uses
  cropped targets; provisional synthetic tests remain clearly labelled.
- Fixed sealed-export subtype lookup (`category` in C01). Challenge exports no longer require
  withheld ground truth; their localisation table is empty and marked not evaluable. Completed
  exports record SHA-256 for predictions, localisation tables and heatmaps.
  The loaded explanation model must match the requested registered model before token consumption;
  the inspected explanation specification is retained in the audit record.
  `--full-model-id` includes the registered full-image comparator in the same one-shot transaction;
  all models/specifications are validated before consuming the one token.
- Added `python -m p3_det.verify_runtime`: actual locked-model/C10 comparison on 50 seeded
  development images with three warm-ups, median/p95 latency and an explicit GPU <=500 ms verdict.
  Mock/unlocked modes, incomplete samples and source/model mismatches are refused.
- Repaired the 50-sample visual-review script and the training notebook's broken checksum cell.
  Training notebook defaults now identify the actual repository/branch, prefer a supplied local
  code snapshot, verify its file hashes, enforce required data/locked preprocessing/measured
  schedule, and stop after failed commands. Checkpoints are hashed in chunks and the summary
  selects the exact configuration's run instead of the lexically last run folder.
- Kaggle code bundles now include required preprocessing/P3 decision snapshots and a SHA-256
  manifest, including the source revision and dirty-worktree status. No datasets, checkpoints,
  credentials or third-party checkout are included.
- Added stage-specific readiness checks (`setup`, `training`, `runtime`, `all`) and `--data-root`
  support so training readiness does not incorrectly require already-selected D16-D18 results.

## Verification

- First focused API/PDF integration run: **19 passed**.
- Final focused notebook/bundle, full-image evaluation, runtime comparison, PDF routes and token
  safeguards: **28 passed**. Later end-to-end sealed-export tests are recorded in the final run.
- End-to-end synthetic internal/challenge export, PDF and notebook checks: **26 passed**.
- Final loaded-model identity and sealed-export checks: **9 passed**.
- Corrective regression including both broad-run failures and their surrounding modules:
  **31 passed**. The two issues were the obsolete code-only archive expectation and a new direct
  tbcore import; the archive test now verifies every snapshot hash and version access now goes
  through the shared adapter. These overlapping counts must not be summed.
- Fresh C10/C15 mock files passed P4's actual validators. C11 passed the strict producer validator;
  P4 does not currently supply a C11 validator.
- Parsed **71 P3 Python files** and every P3 notebook code cell: no syntax errors.
- Rendered and visually inspected the synthetic v2 PDF: one A4 page, readable images/text,
  complete disclaimer/scope text, no clipping. This is layout evidence, not real-model evidence.
- Setup readiness: pinned registered submodule and real execution mode both READY.
- Broad CPU regression: **500 passed, 2 failed, 57 skipped, 2 GPU tests deselected**.
  Both failures were the two issues corrected above; their complete surrounding modules passed
  the 31-test corrective run. The broad run took 48 minutes and was not repeated from scratch.
  Final comparator-export and adapter rerun: **16 passed**, including both crop/full models
  for internal and challenge exports under one token.

## Required inputs and decisions still absent

| Requirement | Actual status | Who/input needed |
|---|---|---|
| C01 definitive split | Missing locally | P1 |
| C04 manifest and processed images | Missing locally | P1 |
| C05 lung ROI table | Missing locally | P1 |
| C06 at locked margin + original development boxes | Missing locally | P1 |
| T4 feasibility, both-size training, cloud kill/resume | Not performed | Kaggle/Colab T4 plus private P1 dataset and Drive setup |
| Real selected checkpoint, D16-D18 results and locks | Absent | Training/selection and D18 guide acknowledgement |
| D16-D19 history disposition | Unresolved | P4/team research review of conflicting earlier locked records |
| D19 for the regional classifier | Unapproved/unmeasured | Agreed protocol, actual bundle, score reproduction, sanity and VRAM evidence |
| D20 final review/lock | Unlocked | Licence/weight terms confirmation and owner review |
| Actual C10/C11/C15 and mixed-batch PDFs | Not delivered | Real configured providers and data; P4/P1 integration |
| RTX 4050 latency/VRAM and CPU E6 | Not measured | Configured pipeline and the required hardware |
| Guide's 40-case review and 50-sample real-data visual review | Not performed | Actual development outputs and human review |
| Sealed exports | Not performed | Reviewed committed code, all gate prerequisites and an existing P4 token |
| Report measured tables and P4 review | Pending | Actual artifacts with n and confidence intervals |
| GitHub CI/merge acceptance | Not established by local tests | Reviewed commits/PR and CI execution |

The read-only hardware inventory lists **Intel Iris Plus Graphics 640** and no NVIDIA GPU.
The current Python environment is CPU-only. No RTX 4050/T4 acceptance measurement can be made here.
No local Kaggle credential file, API token, username/key configuration or CLI was found.
No matching real P1 contracts or trained checkpoint were found in the project/application folders
checked during this pass. Supplying `TB_DATA_ROOT` is supported; the data need not be copied into git.

## Run sequence once inputs arrive

Use base plus P3 requirements and `PYTHONPATH=src`.

1. `python -m p3_ops.preflight --stage setup`
2. Resolve the documented decision-history conflict before accepting new pre-registered results.
3. Set `TB_DATA_ROOT` to P1's contract root; run `python -m p3_ops.preflight --stage training`.
4. Upload `artifacts/p3/kaggle/p3_code_bundle.zip` as a private Kaggle input. Run feasibility and
   cloud kill/resume notebooks; retain their actual JSON/logs. Fill schedules with the existing
   `p3_det.fill_training_config` tool using measured feasibility results.
5. Train both configurations and apply the existing D16/D17/D18 selection commands from the
   runbook. Complete licence and guide reviews; do not force locks to pass readiness checks.
6. Export development/calibration C10, then run `python -m p3_det.verify_runtime --device cuda`.
7. Launch with `python -m p3_ops.serve_v2 --p2-root PATH_TO_REVIEWED_P2_CHECKOUT --preload`.
   Use `/v2/ui`; each analysed/rejected case receives a report link. PDF requests are
   `POST /v2/report` (upload) and `GET /v2/report/{report_key}` (cached original result).
   The standard C14 `p3_ops.api.make_pdf`/`log_case` interfaces remain unchanged.
8. Complete the approved D19 real-model checks, development localisation tables and guide review.
9. Have P4 configure the real R4 providers and complete integration review. Only then run the
   authorised sealed exports and the E6 GPU/CPU commands from the original runbook.
   Include `--full-model-id REGISTERED_FULL_MODEL` on the one sealed export invocation when the
   crop/full comparison is required; do not try to consume a second token for that comparator.
10. Fill the report with measured numbers and artifact citations, obtain P4 review, then prepare
    reviewed commits/PRs. The supplied work plan assigns the backup demo video to P1; it is not
    an additional P3-17 acceptance task.

The optional RT-DETRv2 and box-supervised segmentation extensions are still deferred until core
acceptance; implementing or running them now would not resolve the absent core evidence.
