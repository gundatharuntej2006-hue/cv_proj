# Protocol — label-free per-scanner calibration of the v3 candidate score

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory, exposed.** Vietnam TUNE (encoder, target, fusion all chosen on it). One
bounded run; nothing promoted on any outcome.

## 1. Question

R23 showed the v3 candidate's score carries a per-scanner offset worth ≈ 0.014 pooled
AUROC and 0.043 in previous TB when corrected with a *supervised* scanner term. The
candidate's raw score on TUNE by scanner (measured 2026-11-01, before this protocol):

| scanner | n | prevalence | previous TB | score mean (sd) | within-scanner AUROC |
|---|---|---|---|---|---|
| 1 | 204 | 0.108 | 0.49 | −1.51 (1.98) | 0.759 |
| 2 | 665 | 0.113 | 0.28 | −1.77 (2.12) | 0.788 |
| 3 | 483 | 0.087 | 0.37 | **−0.36** (1.84) | 0.788 |
| pooled | 1,352 | 0.103 | | | **0.7585** |

Scanner 3 sits 1.3 logits above the others at similar prevalence; pooling costs ≈ 0.03
against the within-scanner numbers. The question:

> Can that offset be removed with **unlabelled** local images — and does removing it
> **destroy real differences** when a scanner's population genuinely differs (prevalence,
> previous-TB mix)? A method that fixes the first and fails the second is not usable.

## 2. Design

**Groups.** Each scanner's patients are split once, patient-disjoint, into a
**calibration pool** and an **evaluation group** (50/50, stratified on Xpert, seed 0).
The evaluation group is fixed for the whole run. Labels are never read during
calibration; they are used only (i) to stratify the split and construct the
composition-shift conditions, and (ii) to evaluate.

**Arms**, applied per scanner using a calibration batch of *n* images drawn from that
scanner's pool (n ∈ {25, 50, 100, 200}, capped at pool size; 50 draws per n, seeds 0–49):

| arm | correction | uses labels |
|---|---|---|
| **A** | none — raw candidate score | no |
| **B** | supervised: logistic regression on score + scanner dummies fitted on the pooled calibration batches (R23's "image + scanner"), applied to the evaluation group | **yes** |
| **C** | unlabelled **mean-centering**: s − mean of the batch's scores, per scanner | no |
| **D** | unlabelled **mean / sd alignment**: (s − mean) / sd of the batch, per scanner | no |

Per-scanner affine maps leave within-scanner AUROC unchanged by construction; only the
pooled ranking moves. That is reported, not hidden.

**Operating point, label-free.** Refer the top 30 % by corrected score, threshold = the
70th percentile of the corrected calibration-batch scores pooled over scanners. Report
sensitivity (pooled and per scanner) and **referral burden** (fraction referred per
scanner) on the evaluation group. Under arm A the same rule on raw scores.

**Composition-shift conditions** (n = 100, 50 draws): the calibration batch of **one
scanner (2, the largest pool)** is constructed with a different population while the
other scanners' batches are natural draws —

| condition | scanner-2 calibration batch |
|---|---|
| natural | random (prevalence ≈ 0.11) |
| prevalence 0 | negatives only |
| prevalence 0.30 | 30 % positives |
| previous TB only | past_tb = 1 only |
| no previous TB | past_tb = 0 only |

The evaluation group is unchanged. The damage metric is scanner-2 sensitivity and
referral burden at the fixed rule, plus pooled and previous-TB AUROC. Prevalence 0.30
is a stress case, not a claim about any site.

## 3. Evaluation

Per arm × n: pooled AUROC, within-scanner AUROC, previous-TB AUROC (evaluation group),
sensitivity and referral burden at the label-free rule — mean and 2.5 / 97.5 percentiles
over draws. Paired bootstrap (2,000 resamples, seed 0) of the draw-averaged C and D
scores against A on the evaluation group at n = 100. Composition-shift table as above.

**Reading rule, stated in advance.** Label-free calibration is "usable" only if, at
n ≤ 100, it recovers at least half of arm B's pooled gain over A **and** no
composition-shift condition moves scanner-2 sensitivity by more than 0.10 or its
referral burden by more than 0.10 absolute. Otherwise it is "not usable as is" and the
scanner term stays supervised.

## 4. Partitions

TBX11K `internal_test` and Vietnam REPORT are not read. Cohort fingerprint must equal
the pinned `1aa55cd5…`. No per-case output; no patient identifier in any table.
