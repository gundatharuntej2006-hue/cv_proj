# Protocol — which clinical fields carry the fusion gain

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory, exposed.** Vietnam TUNE by nested CV, as R16 / R22. Nothing adopted.

## 1. Question

R22's candidate (MAIRA-2 + active target + 12-field fusion) reaches 0.7941 on TUNE.
Before it can be frozen as a research candidate, two things must be known:

1. **Which of the 12 fields carry the gain**, so the form can be as short as the
   evidence allows and a missing field has a known cost.
2. **Whether every field is available at screening time.** All 12 come from the
   Dryad screening questionnaire (sex, age group, cough, fever, weight loss, night
   sweats, contact, previous TB, diabetes) or from the scanner identifier — none is a
   laboratory or follow-up result. This run states that; it does not test it.

## 2. Arms — M2 fusion exactly as R16 / R22, on the `ours_X` score

Image score: the R22 `X_noNLM` head (saved seeds), recomputed from the R20 MAIRA-2
TUNE cache; its raw AUROC must reproduce 0.7585.

| arm | clinical columns |
|---|---|
| **full** | all 12 — must reproduce 0.7941 |
| **leave-one-out ×12** | all but one field (`dm` and `dm_unknown` are dropped together as `dm`; `scan1` and `scan2` together as `scanner`) → 10 arms |
| **image + past_tb** | the one field the whole record is about |
| **image + scanner** | the acquisition confound only |
| **image + symptoms** | c, f, wl, ns |
| **image + demographics** | sex, age_group |
| **questionnaire only** | the 10 non-scanner columns (no scanner identifier) |
| **comparator full** | Lunit raw + all 12 — must reproduce 0.7778 |

10-fold stratified × 20 repeats, logistic regression C = 1 L2 on the fold-standardised
score plus the listed columns, out-of-fold predictions averaged over repeats. No new
feature, no interaction, no re-tuning.

## 3. Evaluation

For each arm: AUROC with repeat SD; AUROC in previous TB (466) and without; AUPRC;
sensitivity / specificity at the within-fold 0.70-specificity threshold. Paired ΔAUROC
against **full** (2,000 resamples, seed 20261101) for every arm, overall and in
previous TB. A field "carries" gain if leaving it out costs ≥ 0.01 with an interval
excluding zero; the run reports which do. There is no accept/reject here — it is a
decomposition, not an intervention.

## 4. Partitions

TBX11K `internal_test` and Vietnam REPORT are not read. Cohort fingerprint must equal
the pinned `1aa55cd5…`. No per-case output; no patient identifier in any table.
