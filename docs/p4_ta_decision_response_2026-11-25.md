# P4 response to TA-01–TA-36 — 2026-11-25

**Status:** P4 recommendations for team and guide review. This document does not lock D21 or D24, approve any P1/P2/P3 record, or authorize detector comparisons or sealed evaluation.

## P4-owned decisions

| Item | P4 recommendation | Concern or dependency |
| --- | --- | --- |
| TA-24 | B=2,000 for each prospective paired comparison; report the number of finite replicates and Monte Carlo uncertainty for interval endpoints. | Guide review before D21 lock. Earlier P2 runs keep their original settings. |
| TA-25 | Master seed 42 with the exact SHA-256-derived per-analysis seed and shared canonical CSV draw file specified in the D21 addendum. All three ordered candidate comparisons use the same file and report its SHA-256. | P3's reader currently expects `value.bootstrap.{B, seed}` and defaults to 0; align the interface and remove the fallback before real runs. The D-FINE training seed is separate. |
| TA-26 | Percentile interval for the paired AP50 difference, matching the prospective detector rule. | Record the method and its limitations; do not silently switch to BCa after seeing results. |
| TA-27 | Explicit `ci_level=0.95`, independent of D21's conformal `alpha=0.10`. | Guide review before lock. |
| TA-28 | A non-finite point estimate or replicate invalidates the comparison; record affected replicate IDs, fix the input or metric, and rerun the complete preregistered analysis. | Do not quietly drop replicates or choose a threshold after seeing results. |
| TA-33 | If the team approves a separate D24, allow only `train` and `development` in its `data_used`, as for D16/D17. | Extend the decision registry and gate together after TA-31; until then D24 is a P3 draft outside the locked registry. |
| TA-36 | Put file SHA-256, canonical manifest content hash, and pixel digest definitions in `tbcore`, with versioned definitions and cross-workstream tests against C04 v0 values. | P1 and P3 must agree on the canonicalization and supply the accessible C04 manifest/images before a reference-value test can run. |

## P4 position on shared decisions

| Items | P4 position and dependency |
| --- | --- |
| TA-01–05 | Support H1 alone as the interim duplicate rule; byte-identical pairs may be batch-approved with a retained contact sheet and independent approver; keep excluded rows with `split=excluded_duplicate` and `dup_of`; exclude confirmed within-split duplicates from calibration/internal-test metrics at minimum. P1 owns final ledger and C01 v1. |
| TA-07 | Keep C17/C01 v1 scoped to 8,400 C01 v0 images, document the omitted official test images, and require a separate duplicate check before challenge submission. |
| TA-08–11 | P2's `tau=0.95`, `>=` D06 candidate is the evidence-backed proposal, but remains unlocked pending P1-07 and withdrawal of the competing placeholder. Commit C17 v1 with its sidecar. P1 should retain seed 20260921 for reproducibility and replace, rather than validate, placeholder D01/D02 after review. |
| TA-17, 19–20, 23 | Support a verified licence record and retaining all original full-image boxes for evaluation. P1 must specify sealed C06 production, clipping and visibility behavior before P3 uses these outputs. |
| TA-29–32 | Support a separate P3-owned D24 draft and D24-before-D16 order if the team selects TA-31(a). The previously discussed four variants and AP50 promotion rule remain proposals until P1 confirms input feasibility and P3 records them. Full image/CLAHE-on and crop/CLAHE-on need an available CLAHE store. Runtime cost and R1 interface must be measured. |
| TA-34–35 | Support a committed P1 release index at `data/releases/<id>.json`. For non-image inputs, record pixel digest as `not_applicable` and a separate `built_from` C04 pixel digest where applicable. P1/P3 should agree on the schema before changing readers. |

## Immediate corrections and blockers

- Atul's `8013b42` D06 draft has empty `reviewed_by` and `data_used=["images_only_all_splits"]`; those earlier P4 field objections are resolved. P1-07 review and the competing D06 record remain open.
- D21's future P4 ECE uses 10 equal-width probability bins. P2's earlier A1 used 10 equal-mass bins. Do not merge or relabel those results. Earlier P2 D08/D09/A1 bootstrap seeds also remain as originally recorded.
- The existing `p4_eval.bootstrap_ci` is a hard-coded stub, and `p4_eval.comparisons` plus the report contain illustrative numbers. They cannot be used as measured or sealed evidence. Replace the report only from traceable run artifacts after approvals.
- P1's C01 v0 file/hash, accessible C04 manifest and images, original-box coverage, D21 interface alignment, and a shared manifest-hash definition are prerequisites to P3 real-data comparisons. The current P3 path assumption `processed/v1` conflicts with P1's `processed/v0` handoff.
