# Pre-registered protocol — frozen candidate v3 versus commercial CAD on an untouched cohort

**Version 1, 2026-11-01. Hashed before any cohort is received.** `scripts/headtohead.py`
re-hashes this file, refuses to run if it changed, and writes a one-shot lock after the first
scoring pass. Amendments are a new numbered version with the old one kept.

## 1. Why this protocol exists

Every Vietnam number in the project is on an exposed cohort: the tuning half by design, the
reporting half by six scoring events (record R11). The candidate's encoder, training target
and clinical fusion were each chosen on the tuning half (R12, R20–R23). No number produced
on those patients can validate the candidate. This protocol fixes, in advance, the only
evaluation that can: a cohort nobody on the team has seen, scored once.

## 2. What is frozen

| component | frozen artefact | hash / revision |
|---|---|---|
| candidate v3 | `artifacts/p2/bundle_v3_candidate/` (head.npz, bundle.json, fusion.json) | head sha-256 `7f61a082…`; encoder `microsoft/rad-dino-maira-2` @ `a6132efb9268859545e9f68be18a76a6473f174a` |
| preprocessing | `src/p2_cls/pipeline_v2.py` (512 px, PSPNet lung mask, 12×12 regional max) | as committed on branch `atul` at the protocol's commit |
| thresholds | `bundle.json` → logit at 0.70 / 0.90 specificity, fitted on TBX11K development | −4.0317 / −2.3512 |
| fusion | `fusion.json` coefficients (12 fields) | as in bundle; **scanner indicators are site-specific and are NOT applied to a new site** (§5) |
| comparator | the commercial CAD score supplied with the cohort, as delivered | version recorded in the manifest |

Nothing in this table may be refitted, re-tuned or re-selected on the new cohort. The
scorer reads the bundle; it has no training code path.

## 3. Cohort requirements (what the data request asks for)

- **Screening population**, programmatic or prevalence-survey, enrolment rule stated;
  consecutive or systematic sampling preferred, risk-based acceptable if stated.
- **Microbiological reference for every participant** (Xpert Ultra or culture), not
  radiologist reading.
- **Previous-TB history** per participant (self-reported acceptable, source stated).
- **Acquisition metadata**: radiography system / site identifier per image.
- **Commercial CAD score per image**, with product and version, produced by the vendor's
  software without access to our outputs.
- **Minimum size**: see §7. Images at ≥ 512 px on the short side; frontal PA/AP.
- **Nobody on the team has had access to the images, scores or labels before the lock.**
  The data holder's statement of this is recorded in the manifest.

## 4. Pre-specified endpoints

**Primary.** Paired difference in AUROC for Xpert-positive vs Xpert-negative, candidate v3
**image-only score** minus comparator score, on all eligible participants. Paired bootstrap,
4,000 resamples, seed 0, participant level (cluster level if an event/site identifier is
supplied — then the resampling unit is the cluster and this is stated in the result).

**Decision framing, fixed now.** This is a **non-inferiority** comparison with margin
**0.03 AUROC**: the candidate is declared non-inferior if the lower 95% bound of
(candidate − comparator) exceeds −0.03. Superiority is claimed only if the lower bound
exceeds 0. Any other outcome is reported as "not non-inferior" with the interval.

**Secondary, in this order.**
1. Previous-TB subgroup: same contrast among participants with previous TB — reported with
   its interval; underpowered unless the cohort is large (§7), so exploratory unless the
   pre-specified sample size for it is met.
2. Candidate **with questionnaire fusion, no scanner term** (ten fields) minus comparator.
3. Scanner-stratified AUROC for both systems (within each radiography system), and the
   pooled-vs-within gap for each — the measured scanner-associated offset.
4. Sensitivity and specificity at the candidate's development-fitted 0.70-specificity
   threshold, and at the comparator's vendor threshold if supplied; referral burden per scanner.
5. AUPRC for both systems.
6. Localisation is **not** evaluated (no lesion boxes expected).

**Not permitted after the first scoring pass:** any threshold change, any re-run with a
different bundle, any exclusion not listed in §6.

## 5. Scanner term and site calibration

The fusion model's two scanner indicators were fitted to the Vietnam scanners and mean
nothing elsewhere; they are set to zero (reference scanner) for every participant. Label-free
per-scanner centering (R25) is **not applied** in the primary analysis. If the data holder can
supply a consecutive unlabelled calibration sample per scanner meeting the R25 guard
(≥ 25 images, composition within tolerance of the site's declared reference), the centered
score is reported as a **tertiary, exploratory** contrast, clearly separated, and the guard's
refusal reasons are reported if it refuses.

## 6. Eligibility and exclusions, fixed now

Included: every participant with an image, a microbiological result and a comparator score.
Excluded, and counted: images the quality gate rejects (reason recorded), images without a
comparator score, participants without a microbiological result. No exclusion on clinical
grounds. Prevalence and previous-TB fraction are reported as found.

## 7. Sample size

From the record's own paired intervals on the Vietnam tuning half (n = 1,352, 139 positive):
candidate-fused minus comparator **[−0.0193, +0.0527]** → standard error of the paired
AUROC difference ≈ **0.0184**; in previous TB (466; ≈ 50 positive) **[−0.046, +0.107]** →
SE ≈ **0.039**. The SE scales roughly with 1/√(positives).

| aim | α / power | required SE | positives needed | participants at 10% prevalence |
|---|---|---|---|---|
| non-inferiority, margin 0.03, true difference 0 | one-sided 0.025 / 80% | 0.0107 | ≈ 410 | ≈ 4,100 |
| superiority, true difference +0.02 | two-sided 0.05 / 80% | 0.0071 | ≈ 930 | ≈ 9,300 |
| superiority, true difference +0.03 | two-sided 0.05 / 80% | 0.0107 | ≈ 410 | ≈ 4,100 |
| previous-TB subgroup, difference +0.03 | two-sided 0.05 / 80% | 0.0107 | ≈ 650 previous-TB positives | not achievable in one programme |

**Minimum cohort for this protocol: 400 microbiologically confirmed cases.** Below that, the
analysis is still run once as specified and reported as underpowered; the non-inferiority
decision is then "inconclusive", not "non-inferior". The previous-TB contrast is exploratory
at any realistic size and is labelled so.

## 8. Procedure

1. Data holder confirms §3 and that no team member has seen the data; manifest built
   (`scripts/headtohead.py --build-manifest`), hashed, committed.
2. `headtohead.py --dry-run` validates the manifest and images without scoring.
3. One scoring pass: `headtohead.py --run`. Writes tables, the lock file, and a SHA-256
   manifest of outputs. A second `--run` refuses.
4. Results written up as they come out, interval first, with the decision rule of §4 applied
   verbatim. The record and both documents are updated in the same commit.

## 9. What a positive result would and would not mean

A non-inferior or superior interval on one untouched cohort supports the claim *for that
population and those scanners*. It does not establish scanner invariance, previous-TB
performance, or performance at a site with different enrolment. Those are the next
protocols, not this one.
