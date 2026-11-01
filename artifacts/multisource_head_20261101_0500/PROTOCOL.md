# Protocol — multi-source regional head

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory.** TBX11K calibration is the primary evaluation split; Vietnam TUNE is
exposed and reported for context only. Nothing is adopted into production.

## 1. Hypothesis

> Every production head so far was fitted on one source (TBX11K). The external gap
> (0.95 internal → 0.72 Vietnam) is at least partly a *source* gap: one scanner family,
> one curation style. Adding two further sources with their own scanners and case mix —
> Shenzhen (662) and Montgomery (138), already encoded with the identical lungs-masked
> 12×12 regional pipeline — to the **training** set should make the head less
> TBX11K-specific and improve discrimination on an unseen site, **in either direction**.

Context from the record: R15 showed that *refitting the head on the target site's own
images* adds nothing over a threshold change. This run asks a different question — not
"adapt to the target" but "train on more sources than one, and test on a site none of
them is".

## 2. Arms — one fitting path, four training sets

| arm | training rows | image target | notes |
|---|---|---|---|
| **A** | TBX11K train: healthy + TB (3,241) | `tb_label` | production; must reproduce 0.9498 on calibration vs active target |
| **M** | A + all 800 NLM images | `tb_label` (NLM: file-name label) | the candidate |
| **B1** | TBX11K train (3,241) | `active_tb_label` | R12 arm; must reproduce 0.9460 |
| **MB1** | B1 + all 800 NLM images | `active_tb_label` (NLM TB → 1) | candidate under the active target |

Everything else identical: frozen RAD-DINO 12×12 lungs-masked regions, `Linear(768→1)`
per region, max aggregation, AdamW 3e-3 / batch 256 / wd 1e-4 / ≤120 epochs, early
stopping on **TBX11K development** image-level AUROC (patience 15; NLM never enters the
development set), logit adjustment from the training prior, seeds 17 / 29 / 43 / 61 / 79,
seed-mean scores. No sample weighting, no source balancing — the plainest possible
pooling, so that a null cannot be blamed on a tuning choice.

**NLM labels.** Shenzhen and Montgomery carry one binary label per image
(`_1` = TB manifestation, `_0` = normal). They do not separate active from obsolete
disease; under the active target (MB1) every NLM TB image is treated as active. That
assumption is stated, not tested.

**Consequence.** Once NLM is in training it is no longer an external evaluation set for
these arms. R15's Shenzhen / Montgomery figures (0.9681 / 0.9483) apply to arm A only.

**Built-in correctness gate:** arms A and B1 must reproduce R12 / R17 calibration AUROC
to within 0.0005, or the run halts.

## 3. Evaluation

**Primary:** paired ΔAUROC on **TBX11K calibration** (899; 81 active-TB positives),
arm M minus arm A, scored against `active_tb_label`, paired bootstrap 4,000 resamples
seed 0, identical indices. Minimum meaningful effect **+0.02**, interval excluding zero.
Half-width > 0.05 → INCONCLUSIVE.

**Secondary (reported, not decisive):** MB1 − B1 on calibration; M − A and MB1 − B1 on
Vietnam TUNE (Xpert reference, exposed); sensitivity at the development-fitted
0.90-specificity threshold; AUPRC.

**Decision rule.** ACCEPT only if the primary contrast meets its bar. A TUNE gain
without a calibration gain is reported as exploratory and does not change the verdict.

## 4. Partitions

TBX11K `internal_test` (901) and Vietnam REPORT (1,356) are not read. No per-case
output. No Vietnam patient identifier is written to any table.
