# P4 disposition of legacy P3 decisions — 2026-11-20

This is P4's governance assessment for team review. It does not edit P3-owned records, approve a new protocol, or authorize sealed evaluation.

## History and disposition

Commit `675e052` (2026-10-06) introduced D16–D20 as locked results with the shared placeholder `git_sha: c0ffee1`. D16–D19 cite result evidence absent from `main`, and some `locked_at` dates are later than the commit itself. These records cannot establish completed review, measured results, or prospective registration. Preserve the commit as history; mark its D16–D19 values **withdrawn as unverified scaffold claims**, without deleting or rewriting it.

Ritika's subsequent P3 commits `dbccc8d` (D16), `e7ba134` (D17), `b958521` (D18), and `01ddf68` (D19) contain unlocked rule-only records. They are useful protocol drafts, but they were written after the conflicting `675e052` result claims. They cannot retroactively validate those claims or be described as preregistered before the legacy values. Keep them unlocked and label the conflict in any future report.

D20 is a separate licence decision. The legacy D20 lock is unsupported, but Ritika's current unlocked D20 contains a later pinned-file verification and licence hashes (updated in `50a5332`). Review its actual licence evidence and weight terms independently; do not withdraw that later work merely because the old lock was invalid.

## Starting point for real P3 results

Use `5b94a1b` (2026-11-19, provisional D-FINE preprocessing separated from P2 decisions) as the **implementation baseline only**, not a locked decision or a claim of preregistration. The real experiment starts at a new, team-reviewed, versioned P3 protocol commit made **before** eligible model comparisons. Record that commit SHA and protocol version in the run manifest and later decision evidence. It must fix:

- the C01 v0 file and full SHA-256, C04 v0 manifest hash, eligible train/development IDs, and verified original annotation coverage and box integrity;
- the four detector preprocessing candidates (full/crop 0.10 × CLAHE off/on), fixed D-FINE 512 configuration and training schedule, development AP50 metric, and exact-tie order (full image, then CLAHE off);
- the separate detector preprocessing decision ID/path agreed by the team, and the relationship to D16's historical 512-versus-640 question;
- the P2 model/bundle version used for D19, its target score and input preprocessing, and a separate reviewed D19 protocol before any CAM result is claimed;
- where run outputs, hashes, reviewer sign-offs and any guide acknowledgement will be recorded.

Until that protocol commit exists and inputs are verified, report new runs as exploratory. Do not reuse the old AP50 0.421, checkpoint, confidence threshold or Grad-CAM setting as measured P3 results. D16–D19 remain unlocked; D20 remains unlocked until its own evidence review. P3 owns the replacement records, and a new decision ID or shared contract change needs the team's change-control approval.

## P4 confirmation of detector preprocessing selection rule

P4 accepts the proposed **0.01 absolute development AP50 margin and paired 95% bootstrap CI as joint requirements** for promoting a more complex preprocessing candidate. A lead alone, or a CI that includes zero, is insufficient. The simpler incumbent remains selected in either case. This is a prospective selection rule, not a result or a claim of statistical superiority.

For a reproducible paired CI, keep the same eligible development image IDs, original boxes, detector evaluation code and fixed training schedule for all four candidates. For each of 2,000 replicates with a recorded seed (proposed `42`, subject to D21 approval), resample development **images** with replacement within TB and non-TB strata; apply the same sampled IDs and multiplicities to both candidates, recompute each AP50, and save the paired difference `AP50(challenger) - AP50(incumbent)`. Use the 2.5th and 97.5th percentiles. Promotion requires a point difference `>= 0.01` **and** a lower CI endpoint `> 0`.

Proposed simplicity order, from first incumbent to last challenger: **full image + CLAHE off; full image + CLAHE on; 0.10 lung crop + CLAHE off; 0.10 lung crop + CLAHE on**. Process candidates in that order, comparing each challenger against the currently selected incumbent. Promote only when both conditions above hold. Record all four AP50 values, pairwise differences actually tested, CIs, sample sizes, bootstrap seed, and the final selection. This order must be copied into and approved in the P3-owned versioned protocol before comparisons start; it is not inferred after results.

## Review of P3 update `1ee6322` (23 November)

P4 confirms the four candidates, fixed D-FINE-S 512 configuration for the preprocessing study, development AP50 selection, joint 0.01 margin and positive paired-CI requirement, and the simplicity order in `docs/p3_detector_preprocessing_protocol.md`. The preprocessing choice is made first; D16's distinct 512-versus-640 comparison follows on that selected preprocessing. The old D16–D19 record history remains visible and is not validated by this confirmation.

For a separate P3-owned C16 decision, P4 proposes `decisions/p3_detector_preprocessing.json` with ID **D24**. This is a proposed location, not a locked record. P3 should own its draft and include the approved protocol commit/version, C01 and C04 checksums, original-box coverage, four run IDs, paired comparison output, and reviewer fields. Because the present registry and gate cover D01–D23, adding D24 to the registry and gate is a governance change requiring team approval before it becomes an authoritative lock. D16 should later reference the approved D24/protocol version and be updated through the same versioned review process; do not replace its historical criterion silently.

The real run remains blocked on a concrete input mismatch: the P1 handoff describes `data/processed/v0/manifest.parquet`, while `p3_det.data.config.DataPaths.manifest` and `p3_det.compare_preprocessing.SHARED_INPUTS` at `1ee6322` require `data/processed/v1/manifest.parquet`. P3 must parameterize and hash the actual C04 v0 manifest (or receive an approved C04 v1) consistently in training and comparison metadata. C01 v0 is also not yet in the fetched remote, and P1 has not confirmed original annotation coverage and integrity. The protocol remains draft until those inputs and the bootstrap seed/count are approved.

## PR #4 review requested from P3

Ritika should check that the gate rejects the placeholder D16–D20 records and duplicate decision IDs, and that the audit describes P3 history accurately. Please identify any P3 evidence that exists outside this repository with an accessible path and hash, and confirm or correct the starting protocol boundary above. A green CI run tests code behavior; it does not certify the historical decisions.
