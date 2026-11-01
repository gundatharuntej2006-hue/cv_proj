# DA/DB offset-correction validation — report

## Decision: **NOT EVALUABLE**

Provenance and application requirements failed before scoring. **No DA/DB score was
computed, no AUROC was calculated, and no comparison was made.**

Run directory: `artifacts/dadb_offset_validation_20261021_0255/`
Repository HEAD, unchanged: `f878a6c9e9746803938463941ae1ffaa5d1fd96a`

---

## 1. What was run

Phase 0 only: protocol extraction, cohort verification and provenance verification. The
run stopped at precondition 6 and did not proceed to Phase 1 or 2.

| phase | status |
|---|---|
| 0 — read the protocol, verify preconditions | **complete; precondition 6 FAILED** |
| 1 — freeze the evaluation | not reached |
| 2 — one-shot paired evaluation | **not run** |
| 3 — apply the decision rule | **NOT EVALUABLE** |
| 4 — deliverables | complete |

## 2. Was DA/DB genuinely usable for this one-shot test?

**No — but not because of anything wrong with DA/DB.** The cohort itself passed every
check. The correction is the problem.

### The cohort is sound

| check | result |
|---|---|
| case count | **176, reconciles exactly** with the reported figure |
| duplicates | **zero** — 176 unique case IDs |
| missing files | 0 |
| label mapping | filename first character, `p` → 1 else 0; only `n` (102) and `p` (74) occur |
| DA | 72 cases, 22 TB |
| DB | 104 cases, 52 TB |
| pooled | 176 cases, 74 TB, prevalence **0.4205** |
| against `dadb_third_site.json` | DA ✓ DB ✓ pooled ✓ |
| overlap with training / development / calibration / internal_test | **zero** |
| read by any selection script | only `p2_26_dadb_third_site.py` |
| in the decision provenance table | **absent from all fourteen decisions** |

### What is *not* known about DA/DB, stated rather than glossed

**RAD-DINO checkpoint exposure is UNKNOWN.** The model card says the released checkpoint
was chosen by linear probing on validation sets of *"the evaluation datasets described in
the paper"*, which it does not name. DA/DB is not among the listed pre-training corpora,
but exposure at checkpoint-selection time cannot be excluded.

**Patient-level overlap cannot be checked** — no patient identifiers ship with DA/DB.

So DA/DB is correctly described as **never used for any project-specific selection
decision**, which is verified. It is **not** described as untouched, which is not
establishable.

### Why the test could not run

**The correction has no parameters that can be carried to DA/DB.**

The frozen offsets are three numbers keyed to Vietnam `rad_system` ∈ {1, 2, 3}, estimated
as the median score of non-TB cases in the Vietnam TUNE half. DA and DB are different
acquisition sources at a different institution. No mapping to those groups exists, and
guessing one is forbidden and would be meaningless.

The fallback estimator specified in `NEXT_STEP.md` — *"the median of the other non-TB cases
at its site"* — **requires DA/DB healthy-case labels**, which precondition 6 explicitly
disqualifies. Leave-one-out prevents a case from influencing its own offset; it does not
make the procedure label-free.

A label-free variant (centre by the median of all cases at a site) is still *fitting
offsets on DA/DB*, which precondition 6 also bars, and would be a post-hoc substitution of
the preregistered method.

**The deeper point: a per-source additive offset is by construction a source-fitted
parameter. There is no frozen parameter to transfer to a new source — so this method cannot
be validated as frozen-parameter transfer on DA/DB or on any future cohort.** The previous
run's recommendation contained that flaw and did not notice it. Full reasoning, and the
two distinct questions that were being conflated, in `BLOCKER_REPORT.md`.

## 3. The exact paired effect and its uncertainty

**None was measured.** There is no DA/DB effect estimate, no confidence interval and no
p-value in this run, and none should be quoted from it.

For context, the result this run was meant to test — **on Vietnam REPORT, not on DA/DB**:

| | value |
|---|---|
| pooled AUROC, baseline → offset-corrected | 0.7242 → 0.7448 |
| paired difference | **+0.0206** |
| 95% paired bootstrap CI | **[−0.0030, +0.0436]** — includes zero |
| scanner-stratified bootstrap, 10,000 resamples | [−0.0029, +0.0440] — includes zero |
| permutation test | **p = 0.1575** |
| within-scanner AUROC change | **0.0000 on all three** |

That remains **inconclusive**, exactly as before. This run did not move it in either
direction.

The last row matters for interpretation: a source-specific additive offset leaves
within-source AUROC exactly invariant, so the pooled gain comes entirely from re-ordering
cases *across* sources. **It is not improved within-scanner discrimination**, and the
Vietnam measurements confirm that directly.

## 4. Did the preregistered rule pass?

**The rule was never reached.** It could not be applied, so it did not pass, fail, or come
out inconclusive — the classification is **Not evaluable**.

The rule itself was read from `NEXT_STEP.md` before any verification began, and is recorded
verbatim with its hash in `RUN_MANIFEST.md` §3: pooled difference **at least +0.02**, 95%
paired bootstrap CI excluding zero, and no site's AUROC falling by more than 0.02. **It was
not invented, modified or relaxed**, and no DA/DB outcome was seen at any point.

## 5. What cannot be concluded from this run

- **Nothing about whether the offset correction works.** The Vietnam evidence is unchanged
  and still inconclusive.
- **Nothing about DA/DB's difficulty or the model's performance there.** No score was
  computed.
- **Nothing about generalisation, clinical readiness or real-world validation.** This run
  produced a verification record and a stop, and in any case one cohort of 176 images from
  two sites could never support such a claim.
- The absence of a result here is **not** evidence against the offset correction.

## 6. Two further gaps recorded for whoever picks this up

- **No regional 12×12 DA/DB embedding exists.** The Vietnam result was measured on
  focal-pooling scores; DA/DB has only global 768-d embeddings. A matched baseline needs a
  new 176-image regional pass.
- **`radino_dadb_lungs_only_v1.npz` records `n_sanity_fail = 1`, yet `dadb_third_site.json`
  reports n = 176** for that variant — the case that failed the segmentation sanity check
  was scored, not excluded. Pre-existing in the published artifact; it needs an explicit
  exclusion rule before any future DA/DB evaluation.
- The DA/DB embeddings store no `image_id`; alignment is positional. Verified reproducible
  here, but future extractions should write IDs.

## 7. Recommended next step

**Do not spend DA/DB on this question.** Its single advantage is that nothing has used it
for selection, and with 176 cases and 74 positives it is *smaller* than the Vietnam REPORT
half that already failed to resolve a +0.02 effect. A null there would be uninformative,
and the cohort's one clean read would be gone.

Instead, decide which question is actually being asked (`BLOCKER_REPORT.md` §"The
underlying problem"):

- If the question is **"do these Vietnam offsets transfer?"** — withdraw it. No cohort can
  answer it, because the parameters are source-specific by construction.
- If the question is **"does per-source score centring help in general?"** — that needs a
  label-free estimator, explicit authorisation to fit offsets on the evaluation cohort, and
  a cohort with enough positives to resolve a 0.02 AUROC effect. Vietnam's 141 positives
  were not enough; DA/DB's 74 are fewer.

**The concrete recommendation: park the offset correction as a promising-but-unresolved
finding, and report it that way.** Resolving it needs a properly powered multi-site cohort,
not another underpowered read of a small one. If DA/DB is to be used at all, the higher-
value use is the one still outstanding from two runs ago — **scoring it under the
focal-pooling representation**, which is a plain descriptive measurement with no
preregistration problem, no label dependency, and which would complete the external
comparison table where DA/DB is currently the only gap.

---

## Artifacts

| file | contents |
|---|---|
| `RUN_MANIFEST.md` | frozen protocol, hashes, cohort and provenance verification |
| `BLOCKER_REPORT.md` | why the run stopped, and what clarification is needed |
| `DA_DB_OFFSET_VALIDATION_REPORT.md` | this file |
| `tables/dadb_case_inventory.csv` | 176 rows: case ID, site, label, inclusion, reason; scores marked `NOT_SCORED` |
| `tables/precondition_report.json` | machine-readable verification record |
| `scripts/verify_preconditions.py` | reproducible verification |
| `tests/test_preconditions.py` | 15 tests, all passing |

**Repository safety:** no commit, no push, no deletion. No production source, checkpoint,
historical prediction, existing report, `NEXT_STEP.md` or preregistration was modified —
asserted by hash in the test suite. Every file written is inside this run directory.
