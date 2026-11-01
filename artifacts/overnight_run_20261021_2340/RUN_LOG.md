# Overnight run log — 2026-10-21

**Run directory:** `artifacts/overnight_run_20261021_2340/`
**HEAD at start and at end:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` (branch `master`)
**Working tree at start:** clean apart from this new directory
**Working tree at end:** clean apart from this new directory
**Nothing was committed, pushed, deployed, published or deleted. No file outside this
directory was created or modified.**

---

## Resources, and the one constraint that shaped the run

| | |
|---|---|
| GPU | RTX 4050 Laptop, 6.4 GB, 5.3 GB free (80% cap = 5.2 GB) — **never used; nothing required it** |
| RAM | 16.4 GB total, **0.56–0.8 GB available throughout** |
| Disk | 175 GB free on D: |
| Python | 3.13.2 · numpy 2.2.6 · scikit-learn 1.9.0 · scipy 1.17.1 · pandas 3.0.3 · torch 2.6.0+cu124 |

**Available RAM under 1 GB was the binding constraint.** The regional feature caches are
1.35 GB (TBX11K), 555 MB (Vietnam) and 164 MB (NLM). Nothing that required loading them was
attempted, in line with the runbook's instruction to respect available compute. This did not
block the audit work, because every Vietnam analysis turned out to be possible from a
committed case-level score file (see Phase 1). It did block the one experiment that would
have needed them, recorded in `PRIMARY_EXPERIMENT_PREREGISTRATION.md` §4.

---

## Phase 0 — run record · COMPLETE

Recorded HEAD, branch, `git status --short`, the last eight commits and the repository
layout before touching anything. Created this directory and hashed 26 input files —
labels, splits, case-level predictions, result JSONs, the deployed bundle, the three
regional caches and the core pipeline sources — into `manifest.json`, together with the
environment, resource caps, declared split roles and the feature specification.

No uncommitted user changes existed, so none needed preserving.

## Phase 1 — verify the existing experimental state · COMPLETE

**OOD gate.** `src/p2_cls/pipeline.py:150` resolves the policy as
`ood_policy or meta.get("ood_policy") or "advisory"`, and an unrecognised value raises
rather than falling back silently. The default is advisory: a flagged case is still
classified, still referred, and carries `review_required` as an *additive* flag. The 11
tests in `tests/p2_cls/test_ood_cannot_suppress.py` all pass.

**Grid shapes read from the npz headers, not from file names.** All three regional caches
are exactly `(n, 144, 768)` float16 — 12×12 as claimed, everywhere the regional
representation is used. The 18×18 caches exist alongside them and were not touched.

| cache | shape |
|---|---|
| `regional_tbx11k_lungs_g12_v1.npz` | (6600, 144, 768) float16 |
| `regional_vietnam_lungs_g12_v1.npz` | (2708, 144, 768) float16 |
| `regional_nlm_lungs_g12_v1.npz` | (800, 144, 768) float16 |

**Which file corresponds to which arm — established from source, not from names.** The
important one: `artifacts/p2/audit/vietnam_case_level_gate_audit.csv` holds a column called
`classifier_score`, and the name does not say which model produced it. Reading
`scripts/p2_35_gate_operational_audit.py` lines 72–101 establishes that it is the
**focal-pooling score**: regional 12×12 cache, `repool(..., 12, 12)`, aggregator `max`,
seeds 17/29/43/61/79, head trained on TBX11K train. Recomputing from the file gives TUNE
0.7237 and REPORT 0.7242, matching `mil_final.json` exactly, which corroborates the reading.

**This was the pivotal discovery of the run.** It meant every Vietnam analysis — scanner
stratification, error analysis, score distributions — could be done from a 2,708-row CSV
with no refitting and no multi-gigabyte load, under a RAM budget that would otherwise have
blocked all of it.

**Split boundaries and prior exposure** are tabulated in `split_provenance.csv` with the
evidence for each classification.

## Phase 2 — scanner, label and data coverage · COMPLETE

`scanner_class_counts.csv`, `data_quality.csv`, `split_provenance.csv`.

**The finding that determined the rest of the run:**
`artifacts/p2/labels_tbx11k_trainval_v1.csv` has ten columns and **none is a scanner,
device, institution or acquisition-source field**. The only cohort in the project with a
genuine source variable is Vietnam (`rad_system` ∈ {1,2,3}) — and Vietnam is an evaluation
cohort.

**Data quality: clean.** Zero duplicate IDs in any of the four datasets. Zero missing or
malformed labels. Zero label mismatches and zero scanner mismatches between the Dryad
metadata and the case-level score file across all 2,708 shared keys. The TUNE/REPORT split
is well balanced by scanner (204/204, 665/667, 483/485).

**Source probe** — `source_probe_report.md`, `source_probe_metrics.json`. Fitted on Vietnam
TUNE only; REPORT never opened. Balanced accuracy **0.9737** predicting `rad_system` from the
frozen 768-d embedding, against 0.3381 under label permutation and 0.4919 for the majority
class. Scanner is nearly independent of TB status in the same cohort (χ² = 4.051, p = 0.132,
Cramér's V = 0.039). Stated explicitly in the report: this shows the information is
*available*, not that the TB head *uses* it.

## Phase 3 — reproduce baselines from committed artifacts · COMPLETE

`baseline_reproduction.csv`, 22 rows, recomputed with `sklearn.metrics` and 4,000-resample
bootstrap intervals. No fitting, no threshold tuning.

**Every reported figure that could be checked reproduces exactly — 8 of 8.**

| figure | reported | recomputed | verdict |
|---|---|---|---|
| Vietnam TUNE | 0.7237 | 0.7237 | MATCH |
| Vietnam REPORT | 0.7242 | 0.7242 | MATCH |
| Lunit 3.9.0.1, Vietnam | 0.7827 | 0.7827 | MATCH |
| Shenzhen A / B / C | 0.9411 / 0.8471 / 0.9690 | identical | MATCH ×3 |
| Montgomery A / B / C | 0.9289 / 0.8772 / 0.9690 | identical | MATCH ×3 |

The 18-configuration pooling sweep was **not** repeated.

## Phase 4 — primary experiment · BLOCKED, documented, not improvised

Full reasoning in `PRIMARY_EXPERIMENT_PREREGISTRATION.md`. In short: a source-balanced
objective needs source labels on the training data; TBX11K has none; the only cohort with
source labels is an evaluation cohort whose partitions have declared roles. The runbook's
fallback — source-aware CV within permitted training/development data — fails for the same
reason. Per the runbook's own stop rule, **fitting was skipped and no surrogate source
variable was invented.**

Two read-only analyses were run in its place (the probe, and the scanner-stratified baseline
any future source-robust head must beat: worst-scanner AUROC **0.7000**). A permissible
design for later is pre-registered in §4 of that document, before any corresponding result
exists.

## Phase 5 — additional analyses · COMPLETE

`error_analysis.csv`, `score_distribution_report.md`, `score_distribution_metrics.json`,
`artifact_integrity.json`, `PRIOR_ART_STATUS.md`, `REPORT_CONSISTENCY_REVIEW.md`.

**The run's most useful new result is in `score_distribution_report.md`:** at one shared
referral threshold (70% specificity on TUNE, the development half), referral rate is **50.3%
on rad_system 3 against 24.6% on rad_system 2** — roughly twofold, in the *opposite*
direction to prevalence (0.088 against 0.113). Specificity 0.520 against 0.795. The median
**healthy** person differs by 3.07 logits between those two machines, more than the
interquartile range within either. Ranking varies far less (AUROC 0.700–0.782), so **most of
the disparity is a calibration offset, and per-source thresholding removes the referral-load
gap with no change to the model.**

**Integrity: clean.** All score arrays finite, all values unique, both classes present in
every subset, all three arms pairwise distinct, cohort counts match the NLM published
figures (Shenzhen 336/662, Montgomery 58/138).

**Compute-efficiency audit: not claimable.** The full test suite ran in 196 s and then 137 s
on the same unchanged code, under varying background load. With timing that unstable, no
speedup or regression can be claimed, and none is. Recorded instead: the regional caches
occupy 6.85 GB, the whole run directory 0.16 MB, and no feature extraction was needed.

## Phase 6 — tests and reproducibility · COMPLETE

`test_results.txt` holds the full captured output.

| suite | result |
|---|---|
| full production suite | **228 passed, 1 skipped**, exit 0 |
| OOD non-suppression | **11 passed**, exit 0 |
| new batch-level OOD guards | **5 passed**, exit 0 |
| `py_compile` on every script and test written tonight | exit 0 |

One genuine coverage gap was found and filled non-destructively in
`tests/test_ood_batch_non_suppression.py` inside this run directory: the OOD non-suppression
guarantee was thoroughly tested for `run_stages` but never for `run_batch`, which swallows
per-case exceptions so the worklist always completes — exactly the path where a dropped case
would go unnoticed. The file is **not** in `tests/`, because this run must not modify the
production suite. Promote it if the team wants it enforced in CI.

No dependency was installed. No pre-existing failure was found.

---

## Two in-run corrections, recorded rather than hidden

1. `scripts/source_probe.py` first passed `multi_class="multinomial"` to
   `LogisticRegression`; scikit-learn 1.9.0 has removed that argument. It was dropped —
   multinomial is the default for a multi-class target in that version, so the intended
   behaviour is unchanged.
2. The new batch test first asserted that a missing file yields `status == "ERROR"`. The
   pipeline actually returns `REJECTED` at stage `decode` with reason
   `unreadable: FileNotFoundError`, which is the better contract. The test was corrected to
   pin that behaviour, including an assertion that it stays distinguishable from an OOD
   rejection.

---

## Deliverables

| file | status |
|---|---|
| `RUN_LOG.md` | this file |
| `manifest.json` | 26 hashed inputs, environment, split roles, seeds |
| `scanner_class_counts.csv` | 10 rows |
| `data_quality.csv` | 7 rows |
| `split_provenance.csv` | 9 rows |
| `baseline_reproduction.csv` | 22 rows, 8/8 reported figures reproduce |
| `source_probe_report.md` + `source_probe_metrics.json` | complete |
| `PRIMARY_EXPERIMENT_PREREGISTRATION.md` | blocker documented, successor design pre-registered |
| `error_analysis.csv` + `score_distribution_report.md` + `score_distribution_metrics.json` | complete |
| `artifact_integrity.json` | all checks pass |
| `REPORT_CONSISTENCY_REVIEW.md` | 3 proposed corrections, none applied |
| `PRIOR_ART_STATUS.md` | complete |
| `test_results.txt`, `commands.log` | complete |
| `FINAL_REPORT.md`, `NEXT_PROMPT.md` | complete |

No background job is left running.
