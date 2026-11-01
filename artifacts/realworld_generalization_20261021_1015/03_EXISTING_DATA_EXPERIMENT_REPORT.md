# Phase 2 — Task-definition experiment: result

## Preregistered verdict: **INCONCLUSIVE**

Protocol: `02_EXPERIMENT_PREREGISTRATION.md`, sha256
`f748b4fc1cad44e69d401c8671bcd4e8bd22465901e973de27db0abf398c1dfa`, hashed at 10:27 before
any arm was fitted. Results: `tables/phase2_metrics.json`, `tables/phase2_predictions.csv`.

---

## 1. The decision rule, applied exactly as written

Primary comparison **X against H** on *active TB vs Xpert-negative participants with past
TB* (n = 1,130; 280 positives, 850 negatives):

| criterion | required | observed | pass |
|---|---|---|---|
| 1 — effect ≥ minimum | +0.05 AUROC | **+0.0074** | **no** |
| 2 — paired 95% CI excludes zero | yes | [−0.0123, +0.0264] | **no** |
| 3 — guardrail, full-cohort drop ≤ 0.02 | yes | 0.0003 | yes |

**Verdict: INCONCLUSIVE.** The combined change neither helped by a meaningful amount nor
demonstrably hurt. It is not promoted, and the hypothesis is not claimed as supported.

## 2. What the attribution arms show

The two component changes act in opposite directions and very nearly cancel.

### Primary endpoint — active TB vs past-TB negatives (n = 1,130, 280 TB)

| arm | AUROC | 95% CI | Δ vs H | 95% CI | |
|---|---|---|---|---|---|
| **H** baseline (current) | 0.6173 | [0.5797, 0.6538] | — | | |
| **L** latent TB → negative | **0.6465** | [0.6097, 0.6832] | **+0.0292** | [+0.0223, +0.0363] | **excludes 0** |
| **S** sick negatives added | 0.6037 | [0.5652, 0.6425] | −0.0136 | [−0.0352, +0.0073] | includes 0 |
| **X** both | 0.6248 | [0.5873, 0.6627] | +0.0074 | [−0.0123, +0.0264] | includes 0 |
| *comparator* | *Lunit 3.9.0.1* | *0.7192* | | | |

+0.0292 − 0.0136 ≈ +0.0074. The arithmetic is consistent: **X is L plus S, and S cancels
most of L.**

### Guardrail — active TB vs all negatives (n = 2,708, 280 TB)

| arm | AUROC | Δ vs H | 95% CI | |
|---|---|---|---|---|
| H | 0.7245 | — | | |
| **L** | **0.7386** | **+0.0141** | [+0.0095, +0.0188] | **excludes 0** |
| S | 0.7106 | −0.0138 | [−0.0301, +0.0020] | includes 0 |
| X | 0.7242 | −0.0003 | [−0.0148, +0.0139] | includes 0 |

### Negatives *without* past TB (n = 1,858, 280 TB)

| arm | AUROC | Δ vs H | |
|---|---|---|---|
| H | 0.7822 | — | |
| **L** | 0.7882 | **+0.0060** [+0.0019, +0.0101] | **excludes 0** |
| S | 0.7682 | −0.0140 [−0.0286, +0.0001] | includes 0 |
| X | 0.7777 | −0.0045 [−0.0173, +0.0081] | includes 0 |

**Arm L improves all three endpoints with intervals excluding zero, and it is consistent
across both halves** (primary endpoint: TUNE 0.6485, REPORT 0.6454, against H's 0.6159 and
0.6189).

## 3. What may and may not be concluded from arm L

**L was preregistered as a secondary attribution arm, not as the primary.** Promoting it to
a success now would be exactly the post-hoc endpoint switching the preregistration exists
to prevent. So:

- **Not claimed:** that relabelling latent TB as negative improves the system.
- **Observed, and worth recording:** a consistent positive effect of +0.029 on the primary
  endpoint, +0.014 on the full cohort and +0.006 on the easier subgroup, all with paired
  intervals excluding zero, and stable across both halves of an exposed cohort.
- **Crucially, still below the declared minimum effect.** The preregistration set +0.05 as
  the smallest difference worth a task-definition change. **L's +0.029 does not reach it**,
  so even on its own terms L would not have passed criterion 1. It is a real but small
  effect.

**S is mildly harmful**, which replicates the earlier `hard_negatives.json` finding that
adding `sick_non_tb` negatives worsens external transfer, now under regional pooling rather
than global, and now on a clinically defined subgroup rather than a pooled AUC.

## 4. Why this matters more than the verdict suggests

The hypothesis was that the **training task is defined differently from the evaluation
task**. The data support a weaker version of that:

- The system is trained to call radiographic signs of **inactive** TB positive, and scored
  against Xpert Ultra, which calls those participants **negative**. Removing that specific
  contradiction (arm L) helps measurably and consistently.
- But it recovers only **+0.029 of the 0.101 gap** to the commercial comparator on the same
  subgroup (0.6173 → 0.6465 against 0.7192). **Label definition is part of the story and
  clearly not most of it.**
- Adding abnormal-but-not-TB training negatives, the obvious remedy for a negative-spectrum
  problem, **did not help and slightly hurt.** That is evidence against the simplest version
  of hypothesis A: it is not merely that the model has never seen an abnormal negative chest.

The sharper reading: **TBX11K's `sick_non_tb` images are not the right abnormal negatives.**
They are curated non-TB patients from the same collection pipeline, not consecutive
screening attendees with healed TB. Supplying more of them does not teach the model what a
post-treatment fibrotic chest looks like.

## 5. Honest limitations of this experiment

- **Exploratory, not validation.** The hypothesis was generated by inspecting the full
  Vietnam cohort, so neither half is clean for it. Both halves are reported and agree, but
  that agreement is not independent confirmation.
- **The cohort is exposed** in several earlier ways (see `tables/cohort_exposure_registry.csv`).
- **One count in the preregistration was slightly wrong.** It stated 93 `latent_tb` rows in
  train; the actual figure is **94**, so arms L and X have 447 positives and 2,794/5,494
  negatives rather than 448 and 2,793/5,493. The arms as run are recorded in
  `tables/phase2_metrics.json`. The hashed preregistration was **not** edited.
- **The baseline reproduces to 0.0005, not exactly.** Arm H gives Vietnam full-cohort
  0.7245 against the published 0.7240. Earlier runs matched to four decimals; this is
  run-to-run GPU nondeterminism in head training, within the 0.01 tolerance the script
  enforces, and it is smaller than every effect discussed. Flagged rather than hidden.
- **Three endpoints were examined.** No multiplicity adjustment was applied, and none was
  preregistered. With three correlated endpoints the nominal 95% intervals are optimistic.
- **`past_tb` is self-reported history**, not a radiological determination of healed
  disease, and its accuracy is unknown.
- **TBX11K `internal_test` was not touched** and remains the project's only unexposed
  partition.

## 6. What this changes for the next study

**Arm L should be preregistered as a primary comparison on the next cohort**, not adopted
now. It is cheap — a label mapping, no new data, no architecture change — and it has a
clear mechanism and a consistent, if small, measured effect.

**Stop pursuing negative-class augmentation from TBX11K.** Arm S is the second independent
measurement showing it does not transfer. The negatives that matter — post-treatment chests
in a screening queue — are not in any dataset this project holds. That is a data problem,
which is Phase 3.
