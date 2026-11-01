# Vietnam REPORT exposure audit

**Read-only.** No scoring, training, inference or evaluation was run. No tracked file,
model weight, cache, prediction or prior artifact directory was modified.
**HEAD `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged.**

---

## Conclusion, stated plainly

> **Vietnam REPORT is NOT pristine. It has been scored in at least six distinct analyses
> across five run directories, and at least four of those results influenced a decision.
> It cannot serve as an untouched final external test set, and no result obtained from it
> may be described as independent external validation.**

> **No model score, weight or threshold changed in this audit, and none changed in the
> CLEAR concept audit that preceded it. Vietnam screening AUROC remains 0.7242.**

A one-time, narrowly preregistered test of a *new* intervention whose design was formed on
TUNE only may still be worth running — but it would be a **hypothesis-specific exploratory
test on an already-exposed cohort**, and must be labelled as such.

## The four claims the brief asked to be kept apart

| claim | verdict | basis |
|---|---|---|
| **REPORT was excluded from the CLEAR concept audit** | **TRUE** | `run_manifest.json` records `report_partition_used: false`; the upload was filtered to the TUNE allow-list before leaving the local machine; preflight scanned attached inputs and archive members and found nothing forbidden. Event E9 |
| **REPORT has not been used in prior model development** | **FALSE** | Six scoring events, E1–E7. Three of them (E2, E3, E4) evaluated candidate interventions; E5 evaluated generalization arms; E6 characterised the previous-TB pattern |
| **REPORT can serve as an untouched final external test set** | **FALSE, and permanently so** | Exposure cannot be undone. A prior run already recorded this: *"REPORT has now been scored in at least four separate runs"* — `realworld_generalization_20261021_1015/00_STATE_AND_EVIDENCE_AUDIT.md:50` |
| **REPORT might still support a one-time, narrowly preregistered test of a TUNE-derived intervention** | **CONDITIONALLY TRUE** | Useful, but exploratory and hypothesis-specific. See §4 for a circularity that weakens even this |

## 1. The exposure events

Full ledger: `tables/report_exposure_ledger.csv` (10 rows, including the two negative
controls E8 and E9 and one non-Vietnam flag E10).

| # | date | what was done on REPORT | result | influenced a decision? |
|---|---|---|---|---|
| **E1** | 2026-10-20 | headline evaluation of the selected configuration | **AUROC 0.7242** [0.6843, 0.7618], n=1,356, 141 TB; selection optimism −0.0005 | no — the configuration was selected on TUNE |
| **E2** | 2026-10-21 | **preregistered** arm: INLP scanner-subspace removal | Δ **−0.0002**, CI [−0.0009, +0.0004] → **REJECT** | **yes** — the arm was dropped |
| **E3** | 2026-10-21 | **exploratory** arm, not preregistered: per-scanner offset correction | 0.7242 → **0.7448**, Δ **+0.0206**, CI [−0.0030, +0.0436], p = 0.1575 | **yes** — became the design target for later power planning |
| **E4** | 2026-10-21 | scanner-specific thresholds (fitted on TUNE) evaluated on REPORT | per-scanner sens/spec; **scanner 3 sensitivity 0.674 → 0.488** | **yes** — informed the recommendation against per-scanner thresholds |
| **E5** | 2026-10-21 | generalization arms, endpoints computed per half | primary **+0.0074**, CI [−0.0123, +0.0264]; arm L +0.029 | **yes** — verdict INCONCLUSIVE; arm L not promoted |
| **E6** | 2026-10-21 | previous-TB subgroup characterisation | computed on the **whole cohort, 2,708** — TUNE and REPORT pooled | **yes** — this is the finding the concept hypothesis rests on |
| E7 | 2026-10-21 | reproduction of 0.7242 from stored scores | reproduces | no — integrity check |
| E8 | 2026-10-21 | DA/DB run quotes 0.7242 | — | no — quotation only |
| **E9** | 2026-10-23 | **CLEAR concept audit** | **REPORT not accessed** | no |

E2–E4 are a single sealed pass, deliberately structured as one read covering three arms
(`FROZEN_MANIFEST.md:11-12`: *"Reading REPORT twice is …"*). That was good practice at the
time and is recorded as such. It remains three evaluations' worth of exposure.

## 2. Specific items the brief asked to be verified

**Regional-pooling comparison.** Selection happened on TUNE: `mil_sweep.json` carries
`selected_on`, and contains no REPORT reference (verified by string search). The REPORT
read (E1) came after, and `selection_optimism = −0.0005` is recorded in `mil_final.json`.
The script's own note — *"Configuration selected on TUNE only, and this script was written
before the sweep results were seen"* — is consistent with the artifacts. **No selection
contamination found at E1.**

**Scanner analyses.** `offset_vs_ranking.json` carries `"report_touched": false` and is a
TUNE-side analysis (pooled AUROC 0.7237 = the TUNE figure). The REPORT application is
separate and is recorded in `CALIBRATION_AND_ERROR_ANALYSIS.md:66`: *"The same offsets,
estimated on TUNE, applied unchanged to REPORT in the sealed pass."* **Offsets were fitted
on TUNE, not REPORT.**

**Threshold calibration.** `thresholds_tune.json` holds the fitted thresholds;
`threshold_report_metrics.csv` holds their REPORT evaluation. **Thresholds were fitted on
TUNE and evaluated on REPORT**, not tuned on REPORT.

**The +0.0206 offset result.** Confirmed in `acceptance_verdict.json` under
`offset_correction_exploratory`, with its own note: *"not part of the preregistration;
exploratory"*. CI crosses zero. It nevertheless propagated: `power_planning.json` and
`NEXT_SCREENING_COHORT_PROTOCOL.md` both use `target_delta_auroc = 0.0206` to size a
future cohort. **An exploratory REPORT result is currently acting as a design parameter.**

**Aggregate REPORT metrics used to choose a method.** Yes — E2 (reject), E4 (reject
per-scanner thresholds), E5 (inconclusive, arm L withheld). In each case the decision was
*against* adopting something, which is the less damaging direction, but it is still
selection informed by the evaluation cohort.

## 3. What remains unexposed

| partition | status |
|---|---|
| **TBX11K `internal_test` (901)** | recorded everywhere as never read — **but see the flag below and `UNRESOLVED_ITEMS.md`** |
| everything else | exposed; see `realworld_generalization_20261021_1015/tables/cohort_exposure_registry.csv` |

**Flag (E10), raised not concluded.** `artifacts/p2/sealed/sealed_result_rehearsal1.json`,
dated 2026-10-14, records a scored pass with **n = 901, n_tb = 101, AUC 0.992**, seal
`rehearsal1`, model `radino_full_clahe0_bin`, token issuer `person4`. **901 is the size
this project records for TBX11K `internal_test`.** Every provenance document asserts
internal_test was never read, and each of those assertions traces back to a single
*note* in `artifacts/p2/external/vietnam_eval.json` — *"internal_test untouched"* — rather
than to evidence.

I did **not** open the split, read its labels or score it, so I cannot confirm the
partition from the file itself. Two things are nonetheless true: the model involved
(`radino_full_clahe0_bin`) is an earlier variant, not the current regional-MIL
configuration; and the project's single claimed-pristine partition now rests on a note
that an artifact appears to contradict. **This needs P1/P4 confirmation before anyone
plans an internal_test evaluation.**

## 4. The circularity that matters most for what comes next

The concept hypothesis is: *the classifier's false positives among TB-negatives are driven
by a chronic-appearance axis, and removing it should improve discrimination.*

The CLEAR measurement behind it is clean — TUNE only (E9). But the **previous-TB
false-positive pattern that motivates it was characterised on all 2,708 participants,
TUNE and REPORT pooled** (E6). And the nearest prior intervention, offset correction,
already has a REPORT point estimate of +0.0206 (E3) that is shaping how a successor would
be powered.

So a REPORT test of a concept-informed intervention would not be testing a hypothesis
formed in ignorance of REPORT. It would be a **second, better-specified attempt at an
effect whose first estimate came from the same cohort**. That is worth doing with eyes
open; it is not independent validation, and the write-up must not imply it is.

## 5. Audit method and limits

Evidence came from: run-directory reports and manifests, JSON/CSV metric artifacts,
`scripts/`, and string searches across the repository for the split file, the `report`
partition key, and the specific figures `0.7242`, `0.7448` and `0.0206`.

**Git history was not searched** — this working copy has 7 modified tracked files pending
from earlier documentation tasks and the audit was kept to artifacts on disk. If a REPORT
read happened and left no artifact, this ledger would miss it. Nothing suggests one did.

Confidence is marked per row. E8 is `inferred` (quotation, not access). E10 is `inferred`
on the partition identity. Everything else is `verified` against a named file.
