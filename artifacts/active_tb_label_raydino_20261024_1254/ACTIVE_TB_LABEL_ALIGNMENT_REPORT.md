# Phase 2 — active-TB label alignment: result

**Protocol sha-256 `b88a637c3146f5822f577750902f24d7d338fc84995f75f3fca456d3df5e2f1b`,
hashed 2026-10-24T07:30:25Z, before any model in this run was fitted.** The experiment
script re-hashes `PROTOCOL.md` at start-up and refuses to run if it has changed. Nothing
in the protocol was altered after the results were seen.

**EXPLORATORY.** The same label change was already measured once (arm L, +0.0292 on
Vietnam). This is not confirmation and is not external validation.

---

## Verdict: **REJECT**

| criterion, declared in advance | required | observed | pass |
|---|---|---|---|
| primary paired delta AUROC | **>= +0.05** | **-0.0037** | **no** |
| 95% paired interval excludes zero | yes | [-0.0081, +0.0002] | no |
| interval half-width (power check) | <= 0.05 | **0.0041** | **yes** |
| active-TB sensitivity does not fall > 0.02 | yes | **+0.0123** | yes |

The experiment was **well powered for its own question** — the interval half-width is
0.0041, twelve times smaller than the effect it was built to detect. This is not a null
from noise. A +0.05 effect, had it existed, would have been unmissable.

---

## 1. What was compared

One frozen fitting path, three training targets. Everything else identical: cached
RAD-DINO ViT-B/14 features (rev `110cbc18`), 12x12 lungs-masked regional maps, one shared
`Linear(768 -> 1)` per region with max aggregation, AdamW (lr 3e-3, batch 256, wd 1e-4,
<= 120 epochs), logit adjustment, early stopping on development with patience 15, seeds
17 / 29 / 43 / 61 / 79, seed-averaged scores. `p2_32_mil_sweep.train/.score/.repool` were
imported unmodified.

| arm | training positives | training negatives | n train | pos |
|---|---|---|---|---|
| **A** | active + latent + mixed (`tb_label`) | healthy | 3,241 | 541 |
| **B1** | active + mixed (`active_tb_label`) | healthy **+ latent** | 3,241 | 447 |
| **B2** | active + mixed | healthy (latent dropped) | 3,147 | 447 |

All endpoints are scored against `active_tb_label`, for every arm, on identical rows in
identical order — fixed in the protocol before fitting, so that the comparison is of
discrimination and not of denominators.

**Evaluation split: TBX11K calibration, n = 899, 81 active-TB positives (9.01%
prevalence).** Development is consumed by arm-specific early stopping and could not serve.
Calibration had never been regionally embedded; this run extended the frozen feature cache
to cover it and **verified the extension bit-for-bit against 24 cached development images
before using it** (mean absolute difference 0.00e+00, cosine 1.000000 —
`outputs/embed_calibration.json`).

## 2. Primary endpoint

**TBX11K calibration AUROC vs `active_tb_label`, arm B1 minus arm A, paired bootstrap,
4,000 resamples, seed 0, identical resample indices for both arms:**

> **delta = -0.0037, 95% CI [-0.0081, +0.0002]**

| arm | AUROC | 95% CI | AUPRC | prevalence | sens (active TB) | spec |
|---|---|---|---|---|---|---|
| **A** | **0.9498** | [0.9277, 0.9686] | 0.6881 | 0.0901 | 0.8519 | 0.9156 |
| **B1** | 0.9460 | [0.9233, 0.9653] | 0.6689 | 0.0901 | **0.8642** | 0.9071 |
| B2 | 0.9493 | [0.9276, 0.9681] | 0.6832 | 0.0901 | 0.8519 | 0.9169 |

Thresholds are the smallest score reaching >= 0.90 specificity **on development**, fitted
separately per arm. Specificity on calibration lands at 0.9156 / 0.9071 / 0.9169 — not
0.90, which is the visible evidence that the threshold was not fitted on the evaluation
split.

Per-seed AUROC spread is **+-0.0004** in every arm (`tables/per_seed_results.csv`). The
difference between arms is roughly ten times the seed noise and is stable across all five
seeds. The result is reproducible, not a seed artefact.

## 3. The dissociation — the actual finding

| endpoint | B1 - A | 95% CI | B2 - A | 95% CI |
|---|---|---|---|---|
| **TBX11K calibration (primary)** | **-0.0037** | [-0.0081, +0.0002] | -0.0005 | [-0.0021, +0.0012] |
| calibration, 23 near-duplicates removed (n = 876) | -0.0040 | [-0.0082, -0.0001] | -0.0005 | [-0.0023, +0.0012] |
| mirror: calibration vs `tb_label` | -0.0029 | [-0.0071, +0.0015] | -0.0002 | [-0.0019, +0.0017] |
| **Vietnam TUNE vs Xpert (EXPLORATORY)** | **+0.0203** | [+0.0127, +0.0279] | +0.0023 | [+0.0002, +0.0045] |

**Two things fall out of this table, and they matter more than the verdict.**

**First: the label change helps externally and not internally.** On TBX11K it is worth
-0.004; on the Vietnam screening cohort it is worth +0.020. Same weights, same change,
opposite sign. The internal dataset cannot see the benefit, because separating active from
obsolete TB on TBX11K is already nearly solved (AUROC 0.95) — the 18 obsolete cases in
calibration are a rounding error. The Vietnam cohort is where past TB is 36% of the
positives and the bulk of the false positives, and that is where relabelling pays.

**An internal development split is therefore the wrong instrument for this hypothesis**,
and this run is the evidence for that statement rather than an assumption behind it.

**Second: the gain comes from using latent TB as a negative, not from removing it as a
positive.** B2 drops the 94 latent training images; B1 keeps them and flips their label.
On Vietnam, B2 is worth +0.0023 and B1 is worth +0.0203 — **an order of magnitude apart**.
The model has to be *shown* that an obsolete-TB radiograph is not active disease; merely
not being told it is TB achieves almost nothing.

That separation is what the earlier arm X could not provide, because it bundled the label
change with a negative-composition change and reported a single confounded +0.0074.

## 4. Where the small internal loss comes from

Post-hoc, declared post-hoc, not preregistered (`tables/posthoc_class_separation.csv`):

| contrast on calibration | n | arm A | arm B1 |
|---|---|---|---|
| active TB vs `latent_tb` | 81 vs **18** | 0.7414 | **0.7524** |
| active TB vs `sick_non_tb` | 81 vs 400 | 0.9114 | **0.9035** |

The intended effect appears — the active/obsolete boundary improves — and is swamped by a
loss against non-TB abnormality, which has 22 times the weight in the pooled denominator.
Net: -0.004. **The 18-case contrast is far too small to be an estimate**; it is a
direction, and it is consistent with the Vietnam result rather than independent of it.

## 5. Safety

Sensitivity among the 81 active-TB cases **rose**: 0.8519 -> 0.8642 (+0.0123, one
patient), while specificity fell 0.9156 -> 0.9071. That is an operating-point shift, not a
discrimination gain, and it is why AUROC is the primary endpoint and sensitivity is only a
guardrail.

**The guardrail is underpowered and was declared underpowered before fitting.** With 81
positives, 0.02 is under two patients. It can detect a catastrophe; it cannot detect a
subtle harm, and it was not asked to.

Sensitivity on `latent_tb` cases necessarily falls under B1 — those cases are negatives by
design. That is the intervention, not a harm.

## 6. Threats to this result

| threat | handled how | residual |
|---|---|---|
| near-duplicate leakage between train and calibration | 23 of 899 affected; primary recomputed without them: -0.0040 vs -0.0037 | **negligible** — they affect both arms identically |
| no patient grouping | TBX11K ships no patient ID; splits are image-level and are not described otherwise | **real and unquantifiable** |
| provisional C01 | the definitive split from P1 is outstanding | **real** — every number here is conditional on it |
| development used for early stopping | that is why calibration, not development, is the primary split | handled |
| TBX11K is the training distribution | it is; that is the point of §3 | stated, not fixed |
| `tb1199` | one box-less TB image, latent under C01, active in the v1 label CSV; 1 image in train | cannot move any figure |
| Vietnam TUNE is heavily exposed | it selected the pooling configuration; the TUNE numbers are labelled exploratory everywhere | stated |

**Arm A reproduces production exactly.** Its Vietnam TUNE AUROC is **0.7237** — the
figure already on record for the TUNE half in `offset_vs_ranking.json`. The fitting path
was faithfully reproduced, not approximated.

## 7. What this does and does not license

**Does:**
- Reject `active_tb_label` as an internal-development improvement. The internal evidence
  is against it, and tightly so.
- Establish that the externally useful ingredient is **latent TB as a labelled negative**,
  not the removal of latent TB from the positive class.
- Establish that internal TBX11K splits cannot adjudicate this hypothesis.

**Does not:**
- Promote the +0.0203 Vietnam TUNE result. It is exploratory, on a cohort that selected
  the pooling configuration, for a hypothesis generated from the pooled 2,708. It is also
  the **third** estimate of substantially the same effect (arm L +0.0292 on Vietnam, the
  exploratory scanner offset +0.0206 on REPORT), and three exploratory estimates on
  exposed data do not add up to one confirmation.
- Say anything about clinical readiness, novelty or causality.
- Justify opening Vietnam REPORT or TBX11K `internal_test`.

## 8. Artifacts

| file | contents |
|---|---|
| `tables/metrics_by_arm.csv` | per-arm AUROC/CI, AUPRC, prevalence, sensitivity, specificity, mirror target, Vietnam TUNE |
| `tables/paired_differences.csv` | every paired endpoint with CI, criterion flags, leakage sensitivity |
| `tables/per_seed_results.csv` | all 15 arm-seed fits |
| `tables/score_distribution_by_class.csv` | score distributions over the five source classes |
| `tables/posthoc_class_separation.csv` | post-hoc contrasts, labelled post-hoc |
| `tables/run_manifest.json` | config, inputs, protocol hash, partitions accessed, verdict |
| `outputs/predictions/` | per-case scores, all five seeds, all three arms, calibration + Vietnam TUNE |
| `outputs/embed_calibration.json` | the cache-extension verification |
