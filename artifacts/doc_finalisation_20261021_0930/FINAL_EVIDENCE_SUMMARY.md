# Final evidence summary — TB screening project, as of 2026-10-21

Every figure below is traced to a committed artifact. Anything that could not be verified
is marked **unknown**. **This document improves the correctness of the reporting; it does
not improve any model score, and nothing was retrained or rescored to produce it.**

---

## 1. Verified facts

### 1.1 The configuration currently in use

Frozen RAD-DINO ViT-B/14 (revision `110cbc18d5133582e320b43d53bf5c44e410c936`, 518 px,
patch grid 37), **never retrained or fine-tuned**. Lungs-only masking. Heads fitted on
**TBX11K train only**, healthy-only negatives (3,241 rows, 541 TB), early-stopped on
TBX11K development (659, 59 TB), seeds 17/29/43/61/79. Two arms: matched global pooling
(`pooler_output`) and regional 12×12 max-pooling.

Source: `artifacts/dadb_focal_pooling_20261021_0905/tables/experiment_manifest.json`.

### 1.2 Discrimination across cohorts, current configuration

| cohort | n | TB | matched global | regional max 12×12 | source |
|---|---|---|---|---|---|
| TBX11K development | 659 | 59 | 0.9570 | 0.9368 | `dadb_focal_pooling_…/tables/experiment_manifest.json` |
| Shenzhen | 662 | 336 | 0.9411 | 0.9690 | `artifacts/p2/audit/cross_dataset_metrics.csv` |
| Montgomery | 138 | 58 | 0.9289 | 0.9690 | same |
| **DA** | 72 | 22 | **0.9764** | **0.9755** | `dadb_focal_pooling_…/tables/metrics.json` |
| **DB** | 104 | 52 | **0.8780** | **0.9497** | same |
| *DA+DB pooled* | *176* | *74* | *0.9171* | *0.9636* | *supplementary — see §2.2* |
| Vietnam, full cohort | 2,708 | 280 | 0.6743 | — | `artifacts/p2/dryrun/hard_negatives.json` |
| **Vietnam REPORT** | **1,356** | **141** | — | **0.7242** | `artifacts/p2/regional/mil_final.json` |

The two Vietnam rows have **different denominators** and are not a paired comparison.

### 1.3 Paired pooling comparison on DA/DB

4,000 paired bootstrap resamples, seed 0, identical cases, thresholds fixed on TBX11K
development. Source: `dadb_focal_pooling_20261021_0905/tables/metrics.json`.

| cohort | Δ AUROC (regional − global) | 95% CI | |
|---|---|---|---|
| **DA** | **−0.0009** | [−0.0357, +0.0299] | includes 0 |
| **DB** | **+0.0717** | [+0.0159, +0.1328] | **excludes 0** |
| *pooled* | *+0.0465* | *[+0.0091, +0.0888]* | *excludes 0, supplementary* |

Excluding the single label-blind sanity failure (`px4`, one lung field detected where two
are expected): pooled **+0.0454**, DB **+0.0701**. **No qualitative change.**

### 1.4 Training negative composition, not pooling, is the larger effect on DA/DB

Refitting the global head with the original **all-negatives** rows (5,941) reproduces the
published DA/DB figures **exactly**: DA+DB 0.7791, DA 0.8691, DB 0.7973. Switching to
**healthy-only** negatives (3,241) moves the same cohort to 0.9172 — about **+0.138**.
Both sides use the seed-17 head. Pooling adds **+0.0465** on top.

Source: `dadb_focal_pooling_20261021_0905/tables/negative_composition_check.json`.

**This is a change in training-data composition, not an effect of regional pooling, and
not a demonstrated improvement for screening.** Healthy-only negatives remove
sick-but-not-TB patients from training, which makes a curated case-control task easier
without representing the negatives a screening clinic sends. The screening figure under
the same configuration is **0.7242**.

### 1.5 Reproducibility and integrity, current run

| check | result | source |
|---|---|---|
| streaming loader vs committed `repool(X,12,12)` | **100.000000% elementwise identical**, max abs diff 0.0 | `experiment_manifest.json` → `streaming_equivalence` |
| arms reproduce p2_36 TBX11K dev AUROCs | 0.9570 and 0.9368, **exact** | same → `tbx11k_dev_auroc` |
| thresholds reproduce p2_36 | 0.697045 and −3.336755, to 5 dp | same |
| DA/DB cases / duplicates / missing | 176 / 0 / 0 | `metrics.json` → `integrity` |
| labels and sites re-derived from the filesystem | match all 176 | same |
| published per-site counts (DA 72/22, DB 104/52) | match | same |
| regional feature shape | (176, 144, 768) float16, patch grid 37 asserted | `tables/regional_extraction.json` |
| non-finite scores / shared score arrays | 0 / none | `metrics.json` → `integrity` |
| seed sensitivity, arm B pooled | 0.9625–0.9638 across the five pre-specified seeds | same |
| tests | 19 run-directory passed; production suite 228 passed, 1 skipped | `test_results.txt` |

### 1.6 Provenance of DA/DB

**Verified:** zero overlap with any training, development, calibration or internal-test ID;
read by exactly one script (`p2_26_dadb_third_site.py`); absent from all fourteen rows of
`artifacts/p2/audit/provenance_table.csv`. **DA/DB informed no project configuration
decision.**

**Unknown:** whether the RAD-DINO checkpoint encountered these images. The model card
states the released checkpoint was selected by linear probing on validation sets of "the
evaluation datasets described in the paper", **which it does not name**. DA/DB is not among
the listed pre-training corpora, but checkpoint-selection exposure **cannot be ruled out**.

**Unknown:** patient-level overlap. No patient identifiers ship with DA/DB.

---

## 2. Exploratory findings — real, but not established

### 2.1 Regional max-pooling

Direction is consistent across four curated cohorts and **no cohort shows it worse**:
Shenzhen +0.0279, Montgomery +0.0401, DB +0.0717, DA −0.0009. **But DA and DB disagree**,
and Shenzhen and Montgomery both informed earlier preprocessing choices, so only DA/DB is
an out-of-sample read. The honest statement is: *an exploratory paired gain on DB, a
near-zero observed difference on DA.*

### 2.2 The DA+DB pooled increase is supplementary

DA and DB have prevalence 0.306 and 0.500 and different score distributions. Pooling them
can move AUROC through **cross-cohort ranking effects alone**, independently of
discrimination within either site — the mechanism measured directly in
`overnight_scanner_run_20261021_0208`, where re-aligning three score scales moved pooled
AUROC by +0.03 while leaving every within-group AUROC at exactly 0.0000 change. The pooled
row is not a single-site result.

### 2.3 The scanner-offset correction remains promising-but-unresolved

Vietnam REPORT 0.7242 → 0.7448, Δ **+0.0206**, 95% CI **[−0.0030, +0.0436]**;
scanner-stratified bootstrap over 10,000 resamples [−0.0029, +0.0440]; permutation
**p = 0.1575**. Within-scanner AUROC change **exactly 0.0000** on all three scanners.

**Not validated.** The offsets are keyed to Vietnam `rad_system` values and **must not be
transferred to new sources** — the DA/DB validation attempt was correctly stopped for
exactly this reason (`artifacts/dadb_offset_validation_20261021_0255/BLOCKER_REPORT.md`).

### 2.4 Scanner information is not a linearly removable shortcut

Scanner identity is decodable from the frozen representation at **0.9896** balanced
accuracy (chance 0.3369). Iteratively removing **44 of 768** directions lowers that only to
**0.9118** while TB AUROC stays flat (0.7231–0.7242). The preregistered subspace-removal
experiment was **REJECTED** against its own criteria (Δ −0.0002, CI [−0.0009, +0.0004]).

---

## 3. Limitations and explicit non-claims

- **No novel pooling architecture.** Patch-level aggregation over a frozen chest-radiograph
  foundation model is established prior art (RADAR, MICCAI 2026 satellite, retrieved and
  read). The defensible contribution is the controlled measurement and the cross-cohort
  audit, not the aggregator.
- **No independent external validation.** Shenzhen, Montgomery and the Vietnam cohort all
  informed earlier preprocessing or configuration choices. DA/DB informed none, but is 176
  cases from one institution and carries unknown checkpoint exposure.
- **No clinical readiness.** Nothing is prospective; no clinical endpoint was measured.
- **No causal proof that scanner shortcuts explain site differences.** §2.4 argues against
  that explanation, and nothing here establishes a replacement.
- **No generalisation to all hospitals.** Four curated public cohorts plus one screening
  cohort in one country.
- **The prevalence difference is not the explanation for the screening gap.** AUROC is
  invariant to class balance when the class-conditional score distributions are fixed, so
  0.420 against 0.103 cannot by itself move AUROC. The gap is more plausibly associated
  with case mix and disease severity, the **negative-class spectrum** (curated healthy
  controls against consecutive attendees frequently abnormal for other reasons),
  acquisition and site, reference-standard definition, and selection. These cohorts differ
  on all of them at once, so **the existing evidence identifies no single cause.**

---

## 4. The defensible five-point summary

1. Regional max-pooling has an **exploratory paired gain on DB** (+0.0717,
   [+0.0159, +0.1328]) and a **near-zero observed difference on DA** (−0.0009,
   [−0.0357, +0.0299]).
2. The **pooled DA+DB increase is supplementary** and subject to cross-cohort ranking
   effects.
3. **Training negative composition substantially affects curated-cohort AUROC** (+0.138 on
   DA/DB), **but the screening-cohort gap remains** — 0.7242 against 0.9497–0.9755 on the
   curated sites.
4. **A second independent consecutive screening cohort is needed** to evaluate practical
   generalisation. More curated case-control cohorts will not resolve the deployment
   question.
5. The **scanner-specific offset experiment remains promising-but-unresolved**. It is not
   validated, and Vietnam-specific offsets must not be transferred to new sources.

---

## 5. Unresolved questions, in priority order

1. **Why does the curated-to-screening gap persist under every configuration tried?**
   Candidate mechanisms in §3; none isolated. This is the project's central open question.
2. **Does the scanner-offset correction help?** Underpowered at n = 1,356 / 141 positives.
   Needs a properly powered cohort, not a smaller one.
3. **Why do DA and DB disagree about regional pooling?** A ceiling effect on DA is the
   obvious hypothesis and was not tested.
4. **Was RAD-DINO's checkpoint selected on any of our evaluation cohorts?** Unanswerable
   from the model card. Only the authors can resolve it.
