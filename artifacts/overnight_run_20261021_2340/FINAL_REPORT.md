# Final report — overnight run, 2026-10-21

**Run directory:** `D:\tb-triage\artifacts\overnight_run_20261021_2340\`
**Repository HEAD, unchanged:** `f878a6c` on `master`
**Production code changed: none. Commits: none. Files deleted: none.**

---

## Executive summary

The planned primary experiment could not be run, for a reason that is itself a finding: **the
project has no source labels on any data it is allowed to train on.** A source-balanced
objective needs groups, and on TBX11K there is exactly one group. Fitting was therefore
skipped rather than improvised, as the runbook directs.

What the night produced instead is stronger than the experiment would have been.

Every reported number in the project was independently recomputed from committed case-level
files and **all 8 reproduce exactly**. A diagnostic probe shows the frozen representation
identifies the acquisition machine with **97.4% balanced accuracy** on a cohort the backbone
never saw, while that same machine identity carries almost no information about tuberculosis
(p = 0.132). And a scanner-stratified operating-point audit — new as of tonight — shows that
at a single referral threshold, **one machine's patients are referred at twice the rate of
another's despite having lower disease prevalence**, and that most of that gap is a
calibration offset rather than a model failure, so it is fixable without retraining anything.

One documentation defect was found: a row of the headline sweep table has no committed
artifact behind it. It is proposed as a patch, not applied.

---

## 1. Tasks completed and blocked

| priority | status | output |
|---|---|---|
| 0 — run record and repo protection | **complete** | `manifest.json` (26 hashed inputs), `RUN_LOG.md`, `commands.log` |
| 1 — verify experimental state | **complete** | OOD default confirmed advisory, all grids confirmed 12×12, arm↔file mapping established from source |
| 2 — scanner / label / data audit | **complete** | `scanner_class_counts.csv`, `data_quality.csv`, `split_provenance.csv`, `source_probe_report.md` |
| 3 — reproduce baselines | **complete** | `baseline_reproduction.csv` — 8/8 match |
| 4 — source-balanced head | **BLOCKED** | `PRIMARY_EXPERIMENT_PREREGISTRATION.md` — reason documented, successor design pre-registered |
| 5 — error analysis, distributions, integrity, prior art, report check | **complete** | `error_analysis.csv`, `score_distribution_report.md`, `artifact_integrity.json`, `PRIOR_ART_STATUS.md`, `REPORT_CONSISTENCY_REVIEW.md` |
| 6 — tests and reproducibility | **complete** | `test_results.txt` — 228 + 11 + 5 passed |

## 2. Did the source-balanced method improve, worsen, or prove inconclusive?

**None of these. It was never fitted, so there is no result to report, and none is implied.**

`artifacts/p2/labels_tbx11k_trainval_v1.csv` — the label file behind every head in the
project — has these ten columns and no others:

```
image_id, rel_path, official_split, tbx11k_folder, tbx11k_category,
tb_label, active_tb_label, has_boxes, n_boxes, box_cats
```

No scanner, device, manufacturer, institution or site field, and none is distributed with
TBX11K. `tbx11k_folder` encodes *class* (health 3,800 / sick_non_tb 3,800 / tb 800), not
source; using it as the grouping variable would be class balancing mislabelled as source
balancing. With one source group, Arm B reduces to Arm A.

The only cohort with a real source variable is Vietnam (`rad_system` ∈ {1,2,3}), and it is an
evaluation cohort: TUNE chose the pooling configuration, REPORT was read once. Training a TB
head on either would convert the project's only consecutive screening cohort into training
data and destroy the transfer-gap result the whole report rests on. The runbook's fallback —
source-aware CV within permitted training/development data — fails for the same reason as
the primary design. Shenzhen, Montgomery and DA/DB are each a single site, so source is
perfectly confounded with dataset, and all three are barred from fitting anyway.

**No surrogate source variable was invented.**

## 3. Key metrics, with denominators and uncertainty

### 3.1 Everything reported reproduces

Recomputed with `sklearn.metrics`, 4,000-resample bootstrap, from committed case-level files:

| cohort | arm | n | TB | prevalence | AUROC recomputed | as reported |
|---|---|---|---|---|---|---|
| Vietnam TUNE | focal 12×12 max | 1,352 | 139 | 0.103 | **0.7237** | 0.7237 ✓ |
| Vietnam REPORT | focal 12×12 max | 1,356 | 141 | 0.104 | **0.7242** | 0.7242 ✓ |
| Vietnam all | Lunit 3.9.0.1 | 2,708 | 280 | 0.103 | **0.7827** | 0.7827 ✓ |
| Shenzhen | A global / B mean / C regional | 662 | 336 | 0.508 | 0.9411 / 0.8471 / 0.9690 | all ✓ |
| Montgomery | A global / B mean / C regional | 138 | 58 | 0.420 | 0.9289 / 0.8772 / 0.9690 | all ✓ |

Selection optimism confirmed at **−0.0005**.

### 3.2 Source is almost perfectly readable from the frozen features

Vietnam **TUNE only**, n = 1,352, REPORT never opened. Multinomial logistic regression on
standardised 768-d frozen embeddings, 5-fold CV (one image per patient, so patient-grouped),
seed 17.

| quantity | value |
|---|---|
| balanced accuracy predicting `rad_system` | **0.9737** |
| macro F1 | 0.9772 |
| majority-class accuracy | 0.4919 |
| same, scanner labels permuted (10 repeats) | 0.3381 mean, 0.3515 max |

Twenty-eight of 1,352 images assigned to the wrong machine.

Yet scanner and TB status are close to independent in the same cohort: χ² = 4.051, df = 2,
**p = 0.132**, Cramér's V = **0.039**.

**This shows the information is available. It does not show the TB head uses it** — a
representation can encode a nuisance variable the downstream model ignores. Establishing use
requires the intervention pre-registered in `PRIMARY_EXPERIMENT_PREREGISTRATION.md` §4.

### 3.3 The new operational finding

At one referral threshold (2.4766 — the 70th percentile of non-TB scores on **TUNE**, the
declared development half; no REPORT label was used to pick it):

| scanner | n | TB | prevalence | **referred** | sensitivity | specificity | AUROC |
|---|---|---|---|---|---|---|---|
| rad_system=1 | 408 | 44 | 0.108 | 27.9% | 0.568 | 0.756 | 0.7114 |
| rad_system=2 | 1,332 | 151 | 0.113 | 24.6% | 0.570 | **0.795** | 0.7820 |
| rad_system=3 | 968 | 85 | **0.088** | **50.3%** | 0.741 | **0.520** | 0.7000 |
| all | 2,708 | 280 | 0.103 | 34.3% | 0.621 | 0.689 | 0.7240 |

The median score of a **healthy** person is −0.74 on rad_system 2 and **+2.33** on
rad_system 3 — a 3.07-logit gap, wider than the interquartile range within either group, and
the threshold sits at 2.4766.

**Ranking varies by 0.082 AUROC. The threshold for the same 70% specificity varies by 2.24
logits (1.49 vs 3.73).** Setting the threshold per machine returns rad_system 3's
specificity to 0.70 and removes the referral-load disparity entirely, with no change to the
model. Sensitivity then becomes 0.636 / 0.709 / 0.588.

The commercial comparator shows the same ordering across machines (0.7809 / 0.8095 / 0.7422),
so rad_system 3 is genuinely harder for both systems — but the gap to Lunit is widest on
rad_system 1, so the two are not simply tracking image difficulty.

### 3.4 How much of this is noise

Per-scanner estimates are far less stable than the pooled one:

| scanner | TUNE | REPORT | swing |
|---|---|---|---|
| rad_system=1 | 0.7013 (n=204) | 0.7190 (n=204) | +0.018 |
| rad_system=2 | 0.7762 (n=665) | 0.7890 (n=667) | +0.013 |
| rad_system=3 | 0.7290 (n=483) | 0.6745 (n=485) | **−0.055** |
| pooled | 0.7237 | 0.7242 | +0.001 |

**Any per-scanner AUROC difference below about 0.05 is inside this cohort's noise.** The
ranking spread (0.082) only modestly clears that; the calibration gap is far outside it.
That asymmetry is why the calibration conclusion is the solid one.

## 4. Data and split limitations that remain

1. **No source metadata on any trainable data.** This blocks every source-robustness method
   that needs group labels, not just tonight's. It is the single most consequential gap.
2. **No patient identifiers for TBX11K.** `image_id` is the only key, so patient-level
   grouping and patient-level resampling are impossible there. Vietnam and the NLM sets are
   one image per patient, verified.
3. **Three scanners, one cohort, one country.** Nothing here supports a general claim about
   acquisition hardware.
4. **Vietnam is exposed.** TUNE chose the pooling configuration, so even REPORT sits inside a
   cohort that influenced the design. Every Vietnam number is a *controlled comparison*, not
   an independent generalisation estimate.
5. **Shenzhen and Montgomery are exposed** and informed preprocessing and negative-composition
   choices. Their 0.969 figures are exploratory and reflect how much easier a balanced
   case-control benchmark is than a 10%-prevalence screening cohort.
6. **DA/DB has still not been scored under the regional representation** — the only external
   cohort where focal pooling remains untested.
7. **One sweep row is unverifiable.** See §5.

## 5. The documentation defect

`docs/p2_focal_pooling_final.md` presents an eight-row sweep table and says *"Eight region
grids"*. `artifacts/p2/regional/mil_sweep.json` contains grids 1, 2, 3, 4, 6, 12 plus 18 in
an addendum. **There is no grid-9 entry anywhere.** Every other cell matches exactly (one
6×6 value differs by 0.0001 through rounding).

`scripts/p2_32_mil_sweep.py:42` sets `GRIDS = (1,2,3,4,6,9,12,18)` and line 157 notes "18
gives grids 9 and 18", so the 9×9 run almost certainly happened in the pass that followed the
memory failure — but only the grid-18 rows were persisted. The document's saturation argument
leans on the 9×9 row, though it survives without it (6×6 0.6888 → 12×12 0.7239 → 18×18
0.7186 shows the same rise and plateau).

Two smaller items: the global baseline quoted as **0.6744** matches no artifact (saved values
are 0.6743 with five seeds and 0.6742 with three), and the cross-dataset prediction file mixes
a probability in arm A with logits in arms B and C, which leaves the per-arm thresholds
non-comparable while leaving every AUROC unaffected.

**All three are written as exact patches in `REPORT_CONSISTENCY_REVIEW.md` and none was
applied**, since none is an objectively false statement.

## 6. Was any code or production behaviour changed?

**No.** `git status --short` shows only the new untracked run directory, at the start and at
the end. HEAD is unchanged. No production file, test, checkpoint, embedding, prediction or
existing artifact was created, modified or deleted. The completed pooling sweep was not
re-run. The OOD gate's advisory default was not touched.

Five new tests were written **inside the run directory**, not in `tests/`.

## 7. The single most valuable next step

**Run the pre-registered source-subspace removal experiment** — `PRIMARY_EXPERIMENT_PREREGISTRATION.md`
§4. It is the one design that answers "does the TB head actually use the scanner
information?" without violating a split role: the TB head stays trained on TBX11K and is
never refitted; only a linear source-predicting subspace is fitted, on Vietnam **TUNE**
scanner labels, and projected out of the tokens before pooling. No outcome label from any
reserved partition enters it.

It needs the two regional caches resident (about 2 GB) and tonight's machine had 0.56 GB
free, so it was not attempted. It is streamable in chunks if memory stays tight.

The decision rule is fixed in advance: **accept only if the worst-scanner AUROC improves and
the paired pooled difference excludes zero.** Worst-scanner AUROC is currently **0.7000**,
and the improvement must clear the ±0.05 per-scanner noise floor measured in §3.4.

If that is too heavy, the cheapest genuinely useful alternative is **per-source threshold
calibration**, fitted on TUNE and evaluated on REPORT. It requires no new features at all and
tonight's numbers say it removes the referral-load disparity outright.

---

## What this run does not claim

- No AUC improvement was achieved tonight, because nothing was fitted.
- The scanner probe does not demonstrate shortcut *use*, only shortcut *availability*.
- Nothing here is clinically validated. All of it is retrospective analysis of saved scores.
- The architecture remains non-novel; prior art covers it (`PRIOR_ART_STATUS.md`).
