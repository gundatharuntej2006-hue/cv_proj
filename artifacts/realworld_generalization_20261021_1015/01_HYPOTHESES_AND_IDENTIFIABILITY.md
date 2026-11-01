# Phase 1 — Hypotheses for the screening gap, and what is identifiable

The gap: **TBX11K development 0.9368 → Vietnam REPORT 0.7242**, same weights, same
preprocessing. Six candidate mechanisms, kept separate. Several are almost certainly
operating at once; the cohorts differ on all of them simultaneously, which is the core
identifiability problem.

**Before anything else: AUROC is invariant to class prevalence** when the class-conditional
score distributions are held fixed. The shift from 0.42 to 0.103 prevalence cannot by
itself move AUROC, so "it's the prevalence" is not among the hypotheses.

---

## A. Training negative-class composition is unrepresentative

**Supporting.** Training negatives are **healthy-only** — 2,700 `healthy` images, with all
2,700 `sick_non_tb` excluded. The model never sees an abnormal chest labelled negative.
Vietnam negatives include 850 participants (**35.0%**) with documented past TB, whose
median score is **+2.175** against **−0.338** for negatives without it.

**Counter-evidence, and it is strong.** Adding `sick_non_tb` back (arm S, Phase 2) **did
not help**: −0.0136 [−0.0352, +0.0073] on the past-TB subgroup and −0.0138 on the full
cohort. `hard_negatives.json` found the same thing earlier under global pooling. So the
simple version — "it has never seen an abnormal negative" — is **not supported**.

**Refined version that survives:** the *right* abnormal negatives are absent. TBX11K's
`sick_non_tb` are curated non-TB patients from the same pipeline, not post-treatment
fibrotic chests in a screening queue. **Testable only with a cohort containing them.**

**Falsifier.** A training set enriched with genuine post-TB chests that still fails to
improve the past-TB subgroup would refute this.

## B. Disease severity and clinical spectrum differ

**Supporting.** TBX11K positives are radiographically selected TB cases; Vietnam positives
are Xpert-confirmed, including minimal disease found by active case finding. The
commercial comparator also drops on the same cohort (0.7827 against its curated-benchmark
reputation), which says the cohort is genuinely harder, not that our model is uniquely bad.

**Counter-evidence.** Lunit reaches 0.7827 where we reach 0.7242, so difficulty does not
account for the whole gap — 0.058 of it is us.

**Testable now:** partially. Vietnam has symptoms, age band, contact and comorbidity, but
**no smear grade or severity score**, so severity itself is not measurable here.

**Falsifier.** Severity-stratified analysis showing a flat gap across severity strata.

## C. Scanner / acquisition shift

**Supporting.** Scanner is decodable from frozen features at **0.9737** balanced accuracy
(chance 0.3381). Score distributions shift bodily between machines — median healthy score
+0.51 / −0.86 / +2.35 — and referral rate varies twofold at one threshold.

**Counter-evidence, decisive.** Removing the scanner subspace changed TB AUROC by
**−0.0002** [−0.0009, +0.0004], and iterative removal of **44 of 768** directions left
scanner decodability at 0.9118 while TB AUROC stayed flat. **Scanner information is present
but not load-bearing for the TB decision.** Scanner and TB status are also near-independent
in this cohort (χ² = 4.051, p = 0.132).

**Status: largely ruled out as a cause of the discrimination gap.** It remains a real
*operating-point* problem.

## D. Label / reference-standard differences and verification bias

**Supporting, and this is the hypothesis Phase 2 tested.** TBX11K labels `latent_tb`
positive (`tb_label = 1`) while its own `active_tb_label` marks it 0. Vietnam's reference
is Xpert Ultra — active disease only. **The model is trained to call inactive TB positive
and then scored against a standard that calls it negative.** Reassigning latent to negative
(arm L) gave **+0.0292** [+0.0223, +0.0363] on the past-TB subgroup, +0.0141 on the full
cohort, both intervals excluding zero.

**Counter-evidence.** It recovers only 0.029 of the 0.101 gap to the commercial comparator
on that subgroup, and falls below the +0.05 declared meaningful. **Real but small.**

**Verification bias** in the other direction cannot be assessed: whether every Vietnam
participant was imaged and tested regardless of symptoms is not documented in the Dryad
README. Marked **unknown**.

## E. Preprocessing or selection leakage inflating curated results

**Supporting.** Shenzhen, Montgomery and Vietnam TUNE all informed preprocessing, negative
composition or the pooling choice. The curated numbers are therefore optimistic by an
unmeasured amount. RAD-DINO checkpoint exposure is **unknown**.

**Counter-evidence.** DA/DB informed no configuration decision and still scores 0.9755 /
0.9497 — high. So selection leakage does not explain why curated sets are easy.

**Status: inflates the curated end somewhat; does not explain the gap.**

## F. The feature extractor lacks information for subtle TB

**Supporting.** A frozen backbone cannot represent what it was not trained to represent,
and the learning curve is flat, so more TBX11K data does not help.

**Counter-evidence, decisive.** **Lunit reaches 0.7827 on identical images.** The
information is in the pixels. If our frozen features lacked it, no read-out could recover
it — but a better system does. So this is at most partial.

**Testable now:** only indirectly. Fine-tuning the backbone is out of scope and would
require the same data we lack to evaluate honestly.

---

## What is identifiable with existing data, and what is not

| question | identifiable now? |
|---|---|
| Does the past-TB subgroup drive the gap? | **Yes — measured.** 0.7811 vs 0.6182 |
| Is it a ranking failure or a threshold artefact? | **Yes — ranking.** It appears in AUROC, not only in thresholded specificity |
| Does label definition contribute? | **Partially — arm L, +0.029, exploratory on an exposed cohort** |
| Do TBX11K's non-TB-abnormal negatives fix it? | **Yes — answered, no.** Arm S, −0.0136 |
| Is scanner shift the cause? | **Yes — answered, no.** Subspace removal rejected |
| Does disease severity contribute? | **No** — no severity variable in any held cohort |
| Is there verification bias in Vietnam? | **No** — enrolment/verification completeness undocumented |
| Would genuine post-TB training negatives help? | **No** — no such dataset is held |
| Does any of this transfer to an unseen screening site? | **No** — one screening cohort exists |

**Three of six hypotheses have been materially narrowed using existing data; the remaining
three require a cohort this project does not have.** That is the Phase 3 case.

## Negative subclasses actually available, verified against source documentation

`tables/available_training_label_structure.csv`. Only categories present in the archive's
own structure are listed; **nothing is inferred from filenames**.

| source | category | n | current role |
|---|---|---|---|
| TBX11K `imgs/health` | `healthy` | 3,800 (2,700 train) | the only training negatives |
| TBX11K `imgs/sick` | `sick_non_tb` | 3,800 (2,700 train) | **excluded** from training |
| TBX11K `imgs/tb` | `active_tb` | 631 (426 train) | positive |
| TBX11K `imgs/tb` | `latent_tb` | 139 (94 train) | **positive**, though `active_tb_label` = 0 |
| TBX11K `imgs/tb` | `active_latent_tb` | 30 (21 train) | positive |

**There is no "healed TB" or "post-treatment" category in any training dataset held.**
`sick_non_tb` is the closest available proxy and Phase 2 shows it is not a good one. The
Vietnam cohort has `past_tb` but is an evaluation cohort, and using it for training would
destroy the only screening evaluation the project has.
