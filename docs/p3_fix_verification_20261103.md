# Role 3 fix verification — 2026-11-03

## Results

- Initial broad regression (CPU, excluding the separate large training smoke module): **481 passed,
  5 failed, 57 skipped, 2 GPU tests deselected**. The five failures were missing methodology citations,
  one obsolete unreadable C14 fixture, two gate tests assuming all repository decisions were locked,
  and a C16 test incorrectly requiring every valid decision to be locked.
- Corrective rerun covering all five failures and the changed decision-selection guard: **114 passed**.
  This includes a test proving the gate refuses an unlocked D16. Research files were not locked to make tests pass.
- Final focused v2 bridge / regional score / C14 PDF-log / sealed-token checks: **20 passed**.
- Selected audit checks A1/A2/D1/D2c: **4 passed**. Ownership now uses a related team baseline and records
  the explicitly authorised shared changes; restored frozen documents are checked byte-for-byte.
- C10 and C15 contract generation passed actual P4 validators. C11 passed its strict producer validator;
  P4 currently provides no C11 validator, and this is reported explicitly.
- A generated synthetic PDF was rendered and visually reviewed: one A4 page, no clipped images, result
  text, lesion section, or shared disclaimer/scope statement.
- Changed Python modules passed compilation; tracked changes passed `git diff --check`.

The targeted reruns overlap the broad run; these counts must not be added to claim a larger independent
validation sample. Unit tests use synthetic fixtures and do not establish real model performance.

## Deliberately unresolved

- Audit B4 remains a real **FAIL**: earlier locked D16–D19 records conflict with later pre-registration.
  See `docs/p3_decision_history_conflict.md`. The result-writing tools now refuse this condition.
- No real P3 training, checkpoint selection, locality metric, guide review, sealed evaluation, GPU/CPU E6
  measurement, or demo recording was created. No research lock or acknowledgement was invented.
- The CPU D-FINE training smoke module was excluded from the broad final run because this drive had
  limited space. Existing checkpoint/inference, decision-selection and geometry tests were included.
- The full suite was not rerun from scratch after the corrective changes; all five failed cases and their
  relevant surrounding modules were rerun successfully, followed by the final focused checks.
- The v2 service launch and regional Grad-CAM reproduction still need the real bundle, processed data,
  approved protocol, and hardware. Algebra/adapter tests are not a substitute.

## Environment and cleanup

P3 Python environment: Python 3.14, PyTorch 2.14.1+cpu, CUDA unavailable. Missing pandas/Pydantic/pydicom
dependencies were installed into a task-local ignored directory for verification; requirements were updated
for the P3 integration. CPU test threads were limited to two. Large task-created temporary test folders
were removed after a full-disk fixture failure; no user datasets or pre-existing D-FINE files were removed.

No commits, pushes, publications, or teammate messages were made.
