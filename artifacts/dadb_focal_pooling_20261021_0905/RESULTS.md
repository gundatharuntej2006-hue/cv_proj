# DA/DB focal-pooling evaluation — results

**Descriptive, pre-specified, one-shot cross-cohort evaluation. Exploratory, not clinical
validation.**

Run directory: `artifacts/dadb_focal_pooling_20261021_0905/`
Repository HEAD, unchanged: `f878a6c9e9746803938463941ae1ffaa5d1fd96a`
Manifest frozen before any DA/DB metric: `tables/experiment_manifest.json`

---

## The five required answers, up front

**1. Did regional max-pooling outperform the matched global baseline on DA/DB?**
**Pooled, yes — but the effect is entirely from DB.** On DA the two arms are
indistinguishable.

**2. Paired effect size and uncertainty** (regional minus matched global, 4,000 paired
bootstrap resamples, seed 0, identical cases):

| cohort | n | TB | Δ AUROC | 95% CI | |
|---|---|---|---|---|---|
| **DA** | 72 | 22 | **−0.0009** | [−0.0357, +0.0299] | includes 0 |
| **DB** | 104 | 52 | **+0.0717** | [+0.0159, +0.1328] | **excludes 0** |
| *pooled (supplementary)* | 176 | 74 | *+0.0465* | *[+0.0091, +0.0888]* | *excludes 0* |

**3. Does the sanity-fail case affect the result?** **No.** Excluding `px4` moves the
pooled difference from +0.0465 [+0.0091, +0.0888] to +0.0454 [+0.0092, +0.0890], and DB
from +0.0717 to +0.0701. Every conclusion is unchanged.

**4. Descriptive only, or does it justify a follow-up?** **Descriptive, and it justifies a
follow-up.** One site shows a clear gain, the other shows nothing, and 176 cases across two
sites cannot resolve why. This is a reason to look further, not a result to act on.

**5. What remains unproven?** See §6 and `LIMITATIONS.md`. In short: generalisation beyond
these two Delhi collections, any causal account of *why* DB gains and DA does not, and
whether the RAD-DINO checkpoint ever saw these images.

---

## 1. Setup

Both arms fitted on **TBX11K train only** (3,241 cases, 541 TB, healthy-only negatives),
early-stopped on **TBX11K development** (659, 59 TB), seeds 17/29/43/61/79, lungs-only
masking. **No DA/DB label was used to fit a head, set a threshold, choose a seed, select
preprocessing or pick a checkpoint.**

| arm | representation | aggregation | TBX11K dev AUROC |
|---|---|---|---|
| **A** `global_pooler` | RAD-DINO `pooler_output` | none | **0.9570** |
| **B** `regional_max_12x12` | 12×12 region logits | max | **0.9368** |

Both reproduce the published values from `p2_36_cross_dataset_pooling.py` **exactly**
(0.9570 and 0.9368), and the frozen thresholds reproduce to five decimals (0.697045 and
−3.336755). This is the end-to-end check that the arms here are the same arms as in the
Shenzhen/Montgomery comparison.

**Primary** uses the 5-seed ensemble for *both* arms — matched seed treatment.
**Secondary** uses seed 17 alone for arm A, replicating `p2_36` exactly so the DA/DB
numbers sit on the same footing as the published Shenzhen and Montgomery ones. Both were
declared in the manifest before scoring; they differ by at most 0.0012 AUROC.

## 2. Results per cohort

Cohorts are reported separately because **DA and DB are different collections with
different prevalence (0.306 and 0.500) and are not samples from one population.**

### DA — 72 cases, 22 TB, prevalence 0.306

| arm | AUROC | 95% CI | AUPRC | sens | spec | referral |
|---|---|---|---|---|---|---|
| A matched global | 0.9764 | [0.9423, 0.9973] | 0.9531 | 1.000 | 0.440 | 0.694 |
| **B regional max** | 0.9755 | [0.9354, 0.9990] | 0.9595 | 0.955 | **0.880** | **0.375** |

Paired Δ **−0.0009** [−0.0357, +0.0299]. **No discrimination difference.** Both arms are
near ceiling, so there is little room to separate them on 22 positives.

### DB — 104 cases, 52 TB, prevalence 0.500

| arm | AUROC | 95% CI | AUPRC | sens | spec | referral |
|---|---|---|---|---|---|---|
| A matched global | 0.8780 | [0.8058, 0.9409] | 0.8653 | 0.962 | 0.462 | 0.750 |
| **B regional max** | **0.9497** | [0.9028, 0.9853] | 0.9361 | 1.000 | **0.654** | 0.673 |

Paired Δ **+0.0717** [+0.0159, +0.1328], **excludes zero**.

### Pooled — supplementary only

176 cases, 74 TB, prevalence 0.420. A 0.9171 [0.8700, 0.9569], B 0.9636 [0.9354, 0.9859],
Δ **+0.0465** [+0.0091, +0.0888].

**Read this with care.** Pooling two collections with different prevalence and different
score scales lets cross-cohort ordering move the AUROC on its own, independently of
within-cohort discrimination — the same mechanism documented for scanner offsets in the
previous run. The per-cohort figures above are the interpretable ones.

## 3. Operating points at the TBX11K-development threshold

Thresholds were fixed at 70% specificity on TBX11K development and transferred unchanged.
**No DA/DB threshold was chosen.**

| arm | cohort | TP | FP | FN | TN | sens | spec | referral |
|---|---|---|---|---|---|---|---|---|
| A | DA | 22 | 28 | 0 | 22 | 1.000 | 0.440 | 0.694 |
| **B** | DA | 21 | 6 | 1 | 44 | 0.955 | **0.880** | **0.375** |
| A | DB | 50 | 28 | 2 | 24 | 0.962 | 0.462 | 0.750 |
| **B** | DB | 52 | 18 | 0 | 34 | 1.000 | **0.654** | 0.673 |

The transferred threshold is poorly calibrated for both arms on this cohort — specificity
0.44–0.46 for arm A against the 0.70 it was set for — but **arm B transfers substantially
better** (0.880 and 0.654), and refers 37.5% of DA against arm A's 69.4%. On a balanced
case-control set that difference is not a deployment estimate, but it is consistent across
both sites.

## 4. Seed sensitivity

Five seeds were pre-specified in the original protocol, so this is not post-hoc seed
hunting. Arm B pooled AUROC by seed: **0.9638, 0.9636, 0.9628, 0.9625, 0.9634** — a spread
of **0.0013**, two orders of magnitude below the effect. DA 0.9736–0.9755, DB 0.9479–0.9501.

## 5. A finding larger than the one we came for

The published DA/DB lungs-only figure is **0.7791**; the matched healthy-only global head
is **0.9172** on the seed-17 convention used on both sides of this comparison (the 5-seed
primary arm A is 0.9171). That is not
a discrepancy — it is a different training configuration, and the cause was verified by
reproducing the published number exactly:

| configuration | n train | TBX11K dev | DA+DB | DA | DB |
|---|---|---|---|---|---|
| `p2_26` replica, **all negatives** | 5,941 | 0.9889 | **0.7791** | **0.8691** | **0.7973** |
| *published `dadb_third_site.json`* | | | *0.7791* | *0.8691* | *0.7973* |
| matched, **healthy-only negatives** | 3,241 | 0.9570 | **0.9172** | 0.9764 | 0.8791 |

Both rows use the seed-17 head, so the comparison is internally matched. (The 5-seed
primary arm A reported elsewhere in this document is 0.9171 pooled and 0.8780 on DB; the
seed choice moves the pooled figure by 0.0001.)

**Switching the negative composition moves DA/DB by about +0.138 AUROC — roughly three
times the +0.0465 that pooling buys on the same cohort.** DA/DB informed neither choice, so
this is an out-of-sample confirmation of the negative-composition decision originally made
on Shenzhen and Montgomery.

**It does not establish a general improvement.** Healthy-only negatives remove
sick-but-not-TB patients from training, which makes a curated case-control benchmark easier
without representing the negatives a screening clinic actually sends. Under the same
configuration the screening cohort reaches only **0.7242** (Vietnam REPORT, n = 1,356).

**It also means the project's "India 0.7932" ladder entry is stale.** Under the current
configuration DA/DB scores 0.9171 (5-seed global) and 0.9636 (regional). **Corrected on
2026-10-21** in `docs/p2_vietnam_result.md`, `docs/p2_external_validation.md`,
`report/paper_draft.md`, `report/p2_00_abstract_intro.md`, `report/p2_99_conclusion.md` and
`report/slides/p2_slides_outline.md` — see
`artifacts/doc_finalisation_20261021_0930/DOC_CONSISTENCY_CHECK.md`.

## 6. What this does and does not support

**Supported.**
- Under identical training data, preprocessing, seeds and thresholds, regional max-pooling
  scored higher than matched global pooling on **DB** (+0.0717, CI excluding zero) and
  **equal** on DA (−0.0009).
- The direction is now consistent across four external cohorts: Shenzhen +0.0279,
  Montgomery +0.0401, DB +0.0717, DA ≈ 0. **No cohort has shown regional pooling to be
  worse.**
- DA/DB is the only cohort never used for any project selection decision, so this is the
  cleanest such comparison the project has.

**Not supported, explicitly.**
- **Not clinical validation**, not prospective, not a deployment estimate.
- **Not generalisation to new hospitals.** Two collections from one institution.
- **No causal account.** Why DB gains and DA does not is unexplained; DA being near ceiling
  is a plausible but untested explanation.
- **Not architectural novelty.** Patch-level aggregation over a frozen CXR foundation model
  is established prior art (RADAR).
- **Not comparable to the Vietnam screening cohort.** DA/DB is a balanced case-control set
  (prevalence 0.42) where the same configuration scores 0.9636; on the consecutive Vietnam
  cohort (prevalence 0.103) it scores 0.7242. **That gap is the project's central finding
  and this result does not narrow it.**
- **Checkpoint exposure unknown** — see `LIMITATIONS.md`.

## 7. Recommended next step

**Correct the stale ladder entry, then stop adding cohorts.**

The immediate, cheap action is documentation: DA/DB under the current configuration is
0.9171/0.9636, not 0.7932, and the negative-composition effect (about +0.138) deserves a
line of its own because it is the largest single intervention effect measured on a cohort
that informed no decision. *(Done — see the correction note in §5.)*

Beyond that, the useful question is no longer "does regional pooling help on another
curated benchmark" — it helps at three sites and is neutral at a fourth, which is enough.
It is **why the advantage does not carry to the screening cohort**.

A note on how not to phrase that. **AUROC is invariant to prevalence** when the
class-conditional score distributions are held fixed, so the gap between DA at 0.976 and
Vietnam at 0.724 is *not* explained by prevalence as such. The plausible contributors are
differences in case mix and disease severity, the spectrum of the negative class
(curated healthy controls against consecutive attendees who are frequently abnormal for
other reasons), acquisition and site, reference-standard definition, and selection into
the cohort. **The present evidence does not identify which of these dominates**, and it
cannot: these cohorts differ on all of them at once. Another balanced case-control cohort
would not separate them either. A second consecutive screening cohort, with case-mix and
acquisition metadata recorded, would.
