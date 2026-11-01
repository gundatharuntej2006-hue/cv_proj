# Final report — overnight scanner run, 2026-10-21

**Output directory:** `D:\tb-triage\artifacts\overnight_scanner_run_20261021_0208\`
**Repository HEAD, unchanged start to end:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`
(`master`)
**Elapsed:** 02:08 → 02:46 local, **≈38 minutes**. All seven tasks completed plus a
robustness pass; nothing was blocked. No time was padded.

---

## Executive summary

The preregistered experiment ran, and it **failed its own acceptance criteria**. That is
the headline, and it is a useful failure: it rules out a popular class of method for this
problem on evidence rather than on argument.

Removing the scanner subspace from the frozen representation changed tuberculosis
discrimination by **−0.0002 AUROC** (95% CI [−0.0009, +0.0004]) on REPORT. The companion
diagnostic explains why: scanner identity is so redundantly encoded that stripping **44 of
768 dimensions** only moves scanner decodability from 0.9896 to 0.9118 against a chance
level of 0.337 — while TB discrimination stays flat throughout. **The scanner signal cannot
be linearly projected away, and the TB head was not using the part that can be.**

The disparity instead lives at the operating point. The median *healthy* patient scores
+0.51, −0.86 and +2.35 on the three machines — a 3.2-logit spread wider than the
within-group standard deviation. Correcting for that offset raises pooled REPORT AUROC from
0.7242 to **0.7448 (+0.0206)**, but the interval includes zero under three separate tests,
so it is **promising and unproven**, not a result.

Scanner-specific thresholds did what they are supposed to: **specificity spread across
machines fell from 0.263 to 0.025** and referral spread from 0.236 to 0.047, with five
fewer missed cases. But the gain is redistributive — scanner 3's sensitivity fell from
0.674 to 0.488.

A supporting technical result: the whole experiment ran under **0.33–1.85 GB of free host
RAM** via a streaming GPU loader proved **100.000000% elementwise identical** to the
committed implementation, reproducing the published 0.9368 and 0.7237 exactly.

---

## 1. Tasks

| task | status | evidence |
|---|---|---|
| 0 — preflight and immutable run record | complete | `preflight.json`, `RUN_MANIFEST.md` |
| 1 — per-scanner baseline diagnostics | complete | `tables/baseline_by_scanner.csv` |
| 2 — scanner-specific threshold calibration | complete | `THRESHOLD_RESULTS.md` + 3 tables |
| 3 — preregistered subspace removal | complete, **verdict REJECT** | `SUBSPACE_RESULTS.md`, `tables/acceptance_verdict.json` |
| 4 — representation diagnostics | complete, with one self-correction | `REPRESENTATION_DIAGNOSTICS.md` |
| 5 — calibration and error profile | complete | `CALIBRATION_AND_ERROR_ANALYSIS.md` |
| 6 — resource and reproducibility audit | complete | `RESOURCE_AND_TESTS.md`, `test_results.txt` |
| 7 — synthesis | complete | this file, `NEXT_STEP.md` |
| extra — robustness and determinism | complete | `tables/robustness_checks.json` |

**Nothing was blocked.** The RAM constraint that blocked this experiment in the previous
run was solved rather than worked around.

## 2. The four kinds of change, kept apart

The runbook asks that these never be merged into one "improved score" claim. They are not.

| | intervention | measured on REPORT | verdict |
|---|---|---|---|
| **(a) discrimination** | scanner-subspace removal, preregistered rank 3 | pooled AUROC 0.7242 → **0.7240**, Δ −0.0002 [−0.0009, +0.0004] | **no change** |
| | scanner-offset correction *(exploratory, not preregistered)* | pooled AUROC 0.7242 → **0.7448**, Δ +0.0206 [−0.0030, +0.0436] | **direction positive, not established** |
| **(b) threshold / calibration** | scanner-specific thresholds | pooled sens 0.624 → 0.660 [−0.028, +0.099]; spec 0.678 → 0.693 [−0.004, +0.034] | **no pooled change beyond noise** |
| **(c) representation probe** | iterative removal of 44 scanner directions | scanner balanced accuracy 0.9896 → **0.9118** (chance 0.337); TB AUROC 0.7237 → 0.7242 | **scanner info barely reduced; TB untouched** |
| **(d) workload / equity** | scanner-specific thresholds | specificity spread **0.263 → 0.025**; referral spread **0.236 → 0.047**; missed TB 53 → 48 | **large, consistent equalisation** |

Only (d) shows a substantial effect. (a)'s preregistered arm is a clean null, (a)'s
exploratory arm is underpowered, (b) is noise, and (c) is a negative result about the
method rather than about the model.

## 3. Did scanner-specific thresholds help operating-point metrics?

**On pooled metrics: no, not beyond noise.** Sensitivity +0.0355 [−0.0282, +0.0992],
specificity +0.0148 [−0.0041, +0.0338], referral rate −0.0096 [−0.0273, +0.0081]. Every
interval includes zero. Missed tuberculosis fell from 53 to 48 of 141.

**On equity: yes, decisively.** Specificity spread across the three machines fell from
**0.263 to 0.025**, referral-rate spread from **0.236 to 0.047**.

**With a cost that must be stated.** Scanner 3 sensitivity fell **0.674 → 0.488**, eight
more missed cases out of 43, because its patients were being over-referred under the pooled
threshold and some of that over-referral was catching real disease. Scanners 1 and 2 gained
thirteen detections between them. **This redistributes detection; it does not create it.**

Threshold uncertainty is the other cost: the scanner-1 threshold is estimated from 182
negatives with a bootstrap CI of width 0.87, against 0.42 for the pooled threshold on 1,213.

## 4. Did subspace removal pass its preregistered criteria?

**No. Verdict: REJECT.**

| criterion | value | pass |
|---|---|---|
| 1 — worst-scanner AUROC improves | 0.67447122 → 0.67452383 | technically yes |
| 2 — paired pooled CI excludes zero | −0.0002 [−0.0009, +0.0004] | **no** |
| both required | | **REJECT** |

Criterion 1's pass is an artifact: **+0.000053**, roughly one thousandth of the ±0.055
per-scanner noise floor. The criterion was written without a minimum effect size — a flaw
in the preregistration, reported rather than patched after the fact.

Two further defects in the preregistration are recorded in `SUBSPACE_RESULTS.md` §2: the
stated rank of 3 is wrong for a 3-class problem (the third singular value came out exactly
**0.0**, so one of the three removed directions is arbitrary), and the arithmetic behind it
was mistaken. Rank 3 was nonetheless scored, because that is what was preregistered.

The null does not depend on the seed: subspace seeds 17/29/43/61 give TUNE deltas of
−0.00043, −0.00026, −0.00058, −0.00029.

## 5. Did scanner prediction decrease, and was TB discrimination preserved?

**Scanner prediction decreased, but far too little to matter.**

| directions removed | scanner balanced accuracy | TB AUROC (TUNE) |
|---|---|---|
| 0 | 0.9896 | 0.7237 |
| 3 (preregistered) | 0.9763 | 0.7233 |
| 12 | 0.9604 | 0.7234 |
| 24 | 0.9428 | 0.7242 |
| **44** | **0.9118** | **0.7242** |

Chance is 0.3369 (20 label permutations, sd 0.0130). After sixteen rounds of iterative
nullspace projection the probe is still **0.575 above chance**.

**TB discrimination was fully preserved** — the total range across the entire sweep is
0.0011, with overlapping confidence intervals throughout.

**Stated as the runbook requires:** reduced scanner predictability does **not** by itself
prove a clinical shortcut was removed or that causal fairness improved. Scanner
predictability is a property of the representation, not of the decision, and only linear
structure was examined here. A nonlinear probe might recover scanner identity just as
easily from the projected features.

## 6. The most important caveat

**Vietnam TUNE chose the pooling configuration in earlier work, so REPORT sits inside a
cohort that already influenced the design.** Every REPORT number above is a controlled
comparison under a frozen manifest, **not an independent estimate of generalisation**, and
none of it is clinical validation. Three scanners in one country support no general claim
about acquisition hardware, and the per-scanner noise floor on this cohort is ±0.055 AUROC
between random halves — larger than most of the effects measured here.

Second caveat, specific to anything operational: both corrections **require the scanner to
be known at inference**, which the current pipeline does not read, and neither has defined
behaviour for a machine it was not fitted on.

## 7. Exactly one next action

**Validate the scanner-offset correction on DA/DB, the only external cohort never used for
any selection decision.**

It is the one result worth pursuing — a +0.0206 pooled AUROC point estimate from a change
that touches no weights — and the only reason it is not a finding is power: 1,356 cases
with 141 positives give a CI of [−0.0030, +0.0436], and a permutation test returns
p = 0.1575. The Vietnam cohort cannot settle it, because it is exposed.

DA/DB has 176 images across two sites (DA 72, DB 104), was scored once under the global
representation and **never informed any choice**, so it can function as a genuinely
untouched test of whether per-site score centring transfers. It also still lacks a focal-
pooling score, which this would supply at the same time. Cost: one 176-image embedding pass
plus the scoring already scripted here.

Declare the offset estimator and the acceptance rule before looking, exactly as was done
tonight.

## 8. Confirmation of non-interference

- Repository HEAD **unchanged**: `f878a6c9e9746803938463941ae1ffaa5d1fd96a`.
- `git status --short` at end: only the two untracked run directories.
- **No commits. No pushes. No files deleted.**
- No production source, test, checkpoint, embedding, prediction, manifest or prior report
  was created, modified or overwritten. Every file written is inside this run directory.
- The completed 18-configuration pooling sweep was not re-run. The OOD gate was not
  touched; its advisory default stands and its 11 safety tests pass.
- Tests: **11 new passed**, equivalence proof **passed**, full production suite **228
  passed, 1 skipped**, `py_compile` clean.
- **No background processes remain.**
