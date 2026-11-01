# Execution report — CLEAR concept audit on Kaggle

**Date of run:** 2026-10-23, 20:15:28 UTC (2026-10-24 ~01:45 IST)
**Platform:** Kaggle notebook, Tesla T4, 33.7 GB RAM, Python 3.13.15, Linux 6.18.55
**Wall clock:** 9.3 minutes end to end
**Repository HEAD:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` — unchanged by this work

> **No screening metric was recomputed and none changed. The reported Vietnam screening
> AUROC remains 0.7242.** This was a diagnostic run.

---

## 1. What we set out to do, and why it ran on Kaggle

The question: **does the frozen TB classifier's score track any recognisable chronic
radiological appearance among people who do not have TB?** The project had already
established that the score is elevated in participants with previous TB, and that this
elevation is specific to Xpert-negatives (in positives the previous-TB AUROC is 0.4973,
i.e. chance). What it had never had was any *visual* correlate for that — only patient
history, which is not an image annotation.

CLEAR (a chest-X-ray vision–language model) gives a way to ask the question without
labelling anything: encode each radiograph, and measure its similarity to specific
radiological phrases.

**Why Kaggle rather than locally:** the local machine had 0.6–1.5 GB of free RAM against
the ~3 GB needed to hold CLEAR's 603 MB checkpoint and a slice of its 1.13 GB concept
tensor. The blocker was memory, not rights, not GPU — the local RTX 4050 had 5.3 GB of
VRAM free throughout. Kaggle supplied 33.7 GB of RAM and a T4.

## 2. Setup

### Two private Kaggle datasets

| dataset | contents |
|---|---|
| `tb-triage-vietnam-tune-concept-audit` | 1,352 TUNE images + 4 metadata files, uploaded as a single zip (Kaggle auto-extracted it) |
| `tb-triage-clear-audit-code` | the 10 code and config files |

The data upload was built by `stage_kaggle_data.py`, which filters **everything** down to
the TUNE allow-list before anything leaves the local machine: Vietnam REPORT clinical
rows, REPORT scores and REPORT images are not in the upload at all. Both datasets are
**Private**. The images are CC0-1.0 (Dryad), which settles copyright and settles nothing
about patient sensitivity.

A zip was necessary because Kaggle's uploader caps files per dataset and 1,352 images
exceeds it. The archive is `STORED`, not deflated — PNGs are already compressed, so it is
a container to get under the file-count cap, not a size reduction.

### Pinned components

| | |
|---|---|
| CLEAR weights | HF `peterhan91/CLEAR`, revision `142f169c8b006b0f28cf55c6a237c3003db5ed8f` |
| CLEAR code | GitHub commit `57d2eae57ddabf8e56c8013ec675336d71b3d8e7`, installed `--no-deps` |
| `best_model.pt` | 603,060,057 B, sha256 `da1617a7…` — **matched** |
| `concept_embeddings_368294.pt` | 1,131,400,935 B, sha256 `7241c5b9…` — **matched** |
| `mimic_concepts.csv` | 22,208,546 B, sha256 `93e975b7…` — **matched** |

That last checksum matters beyond provenance: it is the same vocabulary file the nine
concept indices were mapped against, so the pinned indices refer to the vocabulary that
was actually loaded.

Internet had to be enabled: the weights come from huggingface.co and the DINOv2
architecture from github.com at run time. A fully offline run is not possible with this
package, and the preflight says so rather than implying otherwise.

## 3. What was executed

One notebook cell, six stages, each raising on failure so that a failed gate aborts the
whole run rather than letting later stages proceed.

| stage | what it did | outcome |
|---|---|---|
| 1 | locate the code package; install CLEAR pinned by commit | code found, CLEAR installed |
| 2 | preflight: runtime, dependency versions against pins, model revision and asset checksums, cohort, partition scan | **GO** |
| 3 | cohort and partition gate | **PASSED** |
| 4 | encode 1,352 images at 448 px, dot against 9 pinned concept embeddings | (1352, 9) in 3.7 min |
| 5 | Spearman + bootstrap CIs + BH; incremental R²; distributions; AUROC/AUPRC | 4 tables |
| 6 | validate every output file, column set and row count | **PASSED** |

### Cohort as loaded

```
cases               : 1352
TB positive         : 139
previous TB         : 466
cohort fingerprint  : 1aa55cd5c0eb2088  (matches the pinned value)
concepts (exact)    : 9
excluded            : cavity, nodule, reticular opacity
images missing      : 0
images not approved : 0
```

`images not approved: 0` is the point of the staging step — earlier local runs saw 1,356
unapproved images sitting beside the cohort. The upload contained none.

### Statistical specification

Fixed before any result was seen: 10,000 case-level bootstrap resamples, seed 20261023,
Benjamini–Hochberg across the 9 concepts in the `negatives (all)` family only. Subgroup
p values carry **no** correction and are exploratory. Incremental R² is in-sample over a
baseline of scanner (2 dummies), age group, sex, symptom count and previous TB.

## 4. Partition discipline

| | |
|---|---|
| Vietnam **REPORT** | not uploaded, not attached, not read, not scored |
| TBX11K `internal_test` | not referenced anywhere in the code; preflight scans attached inputs and archive members for it |
| per-case concept scores | written as shards inside the Kaggle session only; **never exported**. Stage 6 asserts no per-case table was produced |

Selection is an allow-list (`pid in tune`), so contamination is prevented by construction
rather than detected after the fact. Separate guards cover a missing case, a duplicate
case, an overlapping split manifest, a cohort fingerprint that is not the pinned one, and
a missing approved image.

## 5. Four defects the run surfaced

Worth recording, because three of them would have produced numbers rather than errors.

**1. The package could not have run on Kaggle at all.** The original module derived every
input path from `HERE.parents[2]`, which resolves to `/kaggle` when the code sits in an
input dataset. Fixed by resolving the layout instead of assuming it.

**2. A stale copy of the code ran silently.** Two copies were attached at one point and
the discovery took whichever sorted first — the older one. The discovery now refuses when
it finds more than one complete copy, fingerprints each, and says whether they differ.

**3, 4. The hand-written encoder was wrong in two ways that would not have raised.**
Stage 4 initially stopped with `WeightLoadError: only 0.0% of the parameters could be
matched`, printing the checkpoint's actual key layout. Reading CLEAR's own `hub.py` and
`pipeline.py` then showed:

| | what the hand-written encoder did | what CLEAR does | would it have raised? |
|---|---|---|---|
| key prefix | tried `visual.`, `trunk.`, … | **`visual.backbone.`** | **yes** |
| projection | used the raw backbone output | visual tower is backbone **+ `nn.Linear`**; `encode_image` returns the **projected** vector | **no** |
| preprocessing | ImageNet mean/std, bilinear | **CXR statistics** (101.49/255, 83.44/255), **bicubic** | **no** |

Only the first would have stopped anything. The other two would have produced a complete,
plausible-looking table of meaningless numbers — similarities computed in the wrong vector
space, from wrongly normalised images.

**The fix was to delete the reimplementation** and call `clear.load_pretrained`, pinned by
commit, using CLEAR's own `build_cxr_preprocess`. Confirmation that this worked is in the
run log: the visual tower loaded **87,174,144 parameters**, which is ViT-B/14 (~86.6M)
plus the 768×768 projection (~0.59M). A test now forbids reintroducing a hand-rolled
encoder.

The fail-closed check is the only reason any of this was found before the numbers were
written down. A loader that had quietly proceeded at 0% would have given us a full results
table.

## 6. Results produced

Six files, all validated by stage 6:

```
run_manifest.json
tables/concept_associations.csv      36 rows (9 concepts x 4 groups)
tables/concept_incremental_r2.csv     9 rows
tables/concept_distributions.csv     36 rows
tables/concept_discrimination.csv     9 rows
tables/preflight_kaggle.json
```

All six are preserved in `outputs/` here, hash-verified against the downloads.

**Headline:** among the 1,213 TB-negative participants, the classifier's score is strongly
rank-correlated with the *scarring* concept (ρ = 0.566, 95% CI 0.528–0.602,
ΔR² = 0.198 over scanner, age, sex, symptoms and previous TB) and with *fibrosis*
(ρ = 0.417, ΔR² = 0.100). Eight of nine concepts survive BH at 0.05; pleural thickening
does not.

**Interpretation, the stratification, the consolidation anomaly and what it does and does
not imply for screening performance are in `RESULTS.md`.** The short version: scarring
predicts the model's output strongly and predicts TB status not at all (AUROC 0.494,
CI 0.445–0.542), which is the signature of a nuisance direction — but acting on that is
forbidden under the project's current rules and is not justified by TUNE data alone.

## 7. Environment drift, recorded not waved away

Kaggle's stack differs from the pinned versions:

| package | pinned | on Kaggle |
|---|---|---|
| numpy | 2.2.6 | 2.1.3 |
| pandas | 3.0.3 | 2.3.3 |
| scikit-learn | 1.9.0 | 1.6.1 |
| scipy | 1.17.1 | 1.16.3 |
| torch | 2.6.0 | 2.11.0+cu128 |

Preflight reported each as a warning and wrote them into
`outputs/tables/preflight_kaggle.json`. Rank correlations and percentile bootstrap
intervals are not sensitive at this level, but the drift is on the record.

## 8. What changed, and what did not

| | |
|---|---|
| model predictions, checkpoints, caches | **unchanged** |
| tracked source and documents | **unchanged** |
| previous artifact directories | **unchanged** |
| commits, pushes | **none** |
| screening AUROC | **unchanged at 0.7242** |
| new on disk | this directory, plus the execution package `artifacts/kaggle_exec_prep_20261023_1130/` (51 tests passing) |

Nothing was published, emailed or made public. Both Kaggle datasets are Private. No image
or patient identifier is in this artifact directory — it holds aggregate tables, code
hashes and documentation only.

## 9. Standing limits on how these numbers may be used

- A similarity to the sentence "fibrosis" is **not** fibrosis detection. Each score is a
  weakly supervised text-similarity proxy, not a validated detector and not radiologist
  ground truth.
- **TUNE is exposed.** This is exploratory. It is not validation and must not be used to
  set a threshold.
- `past_tb` is clinical history, not an image annotation.
- Association is not causation: none of this shows the classifier *uses* a visual cue.
- **cavity, nodule and reticular opacity were excluded, not measured** — no exact
  vocabulary entry exists for them, and the protocol forbids approximating a concept by
  pooling sentences after seeing the data. Cavity matters clinically in TB, and its
  absence is a limitation of the audit.
- No concept score may be subtracted from, or used to modify, any TB score.
- CLEAR's exposure to this cohort is unknown.
- Nothing here is a clinical-readiness, novelty or causality claim.

## 10. Next step

One sealed, preregistered test on the Vietnam REPORT half is the only legitimate way to
find out whether the nuisance-direction hypothesis converts into screening performance.
REPORT is single-use, so that preregistration should be written and agreed before it is
spent. It has not been written yet, and REPORT has not been touched.
