# Phase 3 — Protocol for a second screening cohort

**Status: design only. The cohort does not exist and data access is the blocker.**
Companion documents: `05_SAMPLE_SIZE_AND_POWER.md`, `06_DATA_ACQUISITION_CHECKLIST.md`,
`tables/data_source_shortlist.csv`.

---

## 1. What this study must settle, stated as questions

1. Does the curated-to-screening gap reproduce **outside Vietnam**?
2. Does the **past-TB deficit** reproduce — our AUROC 0.6182 against past-TB negatives
   versus 0.7811 against the rest — and does it reproduce for a commercial comparator too?
3. Does **arm L** (latent TB relabelled negative) help on a cohort that did not generate
   the hypothesis?
4. Is per-source threshold or offset correction safe, or does it cost detection as it did
   in Vietnam (scanner 3 sensitivity 0.674 → 0.488)?

Questions 1–3 are the reason for the study. Question 4 rides along if scanner metadata
exist.

## 2. Primary endpoint

**AUROC for active TB versus Xpert-negative participants with a documented history of
previous TB.**

This is deliberately *not* pooled AUROC. The pooled figure mixes an easy comparison
(active TB vs normal chests) with the hard one, and its value depends on how many past-TB
participants happen to be enrolled — which is a property of the recruitment site, not of
the model.

**Co-primary: sensitivity at 70% specificity**, both fixed on the locked development data
and transferred unchanged, because the intended use is triage against a fixed confirmatory
budget. The WHO target product profile is 90% sensitivity at 70% specificity; the current
system reaches 0.59 on Vietnam REPORT and the commercial comparator 0.65.

Both reported for the locked configuration and a matched baseline on identical patients,
paired, with bootstrap intervals.

## 3. Cohort requirements

| requirement | specification | why it is here |
|---|---|---|
| **recruitment** | consecutive, or an explicitly documented representative sampling frame with eligibility criteria and a participant flow diagram | the gap is the object of study; a case-control set cannot measure it |
| **reference standard** | microbiological (Xpert Ultra or culture) on **all enrolled participants**, not only those with abnormal films | see §4 |
| **previous TB history** | recorded for every participant, **and record-verified where possible** | the primary endpoint depends on it; Vietnam's and Zambia's are both self-reported |
| **negative-class detail** | other pulmonary diagnoses where known | hypothesis A refinement |
| **severity** | smear grade or Xpert semi-quantitative result | the one variable no held cohort has |
| **patient identifiers** | pseudonymous, stable, supplied — store hashes only | TBX11K ships none; patient-level splits are otherwise impossible |
| **acquisition metadata** | device, institution, view, protocol, date where permitted | enables the scanner analyses; **absence does not disqualify a cohort** |
| **images** | frontal, DICOM preferred at original resolution | lets preprocessing be rerun rather than inherited |
| **prevalence** | recorded, not controlled | a realistic negative spectrum is the point |

**A cohort from a different country and a different radiography vendor is worth more than
a larger one from the same setting**, because it separates "the Vietnam programme" from
"the method".

## 4. Verification bias, if universal testing is not feasible

Universal microbiological confirmation is expensive and sometimes impossible — participants
cannot always produce sputum. If testing is triggered by an abnormal radiograph or by
symptoms, then **the cases the model misses are disproportionately never tested, so they
never enter the denominator and measured sensitivity is biased upward.** This is partial
verification bias and it inflates exactly the endpoint this study exists to measure.

Mitigations, in order of preference:

1. **Universal testing.** Preferred; everything else is a compromise.
2. **Random sample of the screen-negative stratum**, tested regardless, with
   inverse-probability weighting and the sampling fraction prespecified.
3. **Documented follow-up** of untested participants for incident TB over 6–12 months, so
   missed cases surface later.
4. If none is possible, report sensitivity as **an upper bound** and say so in the
   abstract, not only in the limitations.

Kagujje et al. 2022 restricted analysis to symptomatic adults with complete data — a
reminder that even well-conducted ACF studies rarely achieve universal verification.

## 5. Data roles, declared in advance

| role | source |
|---|---|
| **model development** | TBX11K train (3,241 or 5,941 rows depending on arm) |
| **calibration / threshold selection** | TBX11K development (659) and calibration (899) |
| **final evaluation** | **the new cohort, scored once** |

**Previously exposed, and not relabellable as external validation:** TBX11K train /
development / calibration; **Vietnam TUNE and REPORT** (REPORT has been scored in at least
four runs); **Shenzhen**; **Montgomery**; **DA and DB** (informed no configuration
decision, but now scored). **RAD-DINO checkpoint exposure to any of these is unknown.**

**TBX11K `internal_test` (901) remains unread** and is the project's only unexposed
partition. It should be spent deliberately, not incidentally — it can answer "did the
locked configuration change hurt internal performance?" and nothing about screening.

## 6. Locking procedure

1. Lock weights, preprocessing, aggregation, label mapping and **all thresholds** on
   permitted development data. Record hashes.
2. Write the analysis manifest — endpoints, subgroups, bootstrap seed and count, decision
   rules, exclusion rules, missing-data handling — and hash it.
3. Score the new cohort **once**.
4. No refitting, recalibration or threshold adjustment for the primary analysis.
   Recalibrated performance is a clearly labelled secondary.

This procedure has been executed twice in this project and both times a test asserted the
manifest predated the results on disk. Keep that test.

## 7. Prespecified subgroups

Previous TB (primary); symptom status; age band; sex; diabetes; HIV where recorded; smear
grade; scanner/device; site. Each with intervals, each **marked underpowered unless
§05 says otherwise**.

**Worst-subgroup performance appears beside the pooled figure in every table.** A pooled
gain with worst-subgroup degradation is a failure — the rule already used in this
project's scanner preregistration.

## 8. Analysis plan, keeping the three things apart

- **Discrimination:** AUROC, AUPRC with baseline prevalence stated. Pooled and per-subgroup.
- **Calibration:** slope, intercept, Brier, calibration plot. Fitted on development data
  only.
- **Referral burden:** referral rate, PPV, NPV, confirmatory tests per 1,000 screened,
  number needed to test per case found, at the locked threshold.
- **Equity:** referral rate and specificity by subgroup, spread stated.

A threshold change cannot alter AUROC. An additive per-group offset cannot alter
within-group AUROC. Any table that merges these is wrong on its face.

## 9. Interpretation rules, fixed now

- **An underpowered null is inconclusive, not evidence of no effect.** If the cohort cannot
  detect the prespecified effect, the report says "inconclusive, power inadequate".
- A favourable point estimate with an interval including zero is not a positive result.
- Subgroup differences below the measured noise floor are reported as noise. On Vietnam
  that floor was ±0.055 AUROC between random halves of one scanner's patients.
- No single cohort establishes clinical readiness or generalisation.
- If the cohort informs any configuration choice, it becomes exploratory permanently.

## 10. What can be done before any new data arrive

- Preregister arm L as a primary comparison for the new cohort (cheap, mechanistically
  clear, +0.029 observed).
- Keep `internal_test` unread.
- Build the analysis pipeline and its tests against synthetic data so that scoring the new
  cohort is a single command.
- Pursue §06. **Do not run another architecture sweep** — see
  `FINAL_RESEARCH_DECISION.md` for why.
