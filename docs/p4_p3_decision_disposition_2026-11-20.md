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

## PR #4 review requested from P3

Ritika should check that the gate rejects the placeholder D16–D20 records and duplicate decision IDs, and that the audit describes P3 history accurately. Please identify any P3 evidence that exists outside this repository with an accessible path and hash, and confirm or correct the starting protocol boundary above. A green CI run tests code behavior; it does not certify the historical decisions.
