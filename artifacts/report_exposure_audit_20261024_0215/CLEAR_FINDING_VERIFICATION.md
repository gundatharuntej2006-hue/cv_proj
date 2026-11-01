# CLEAR finding — provenance and method verification

Every item below was checked against a named file. Nothing was re-run. Where a check could
not be completed from artifacts, it says so rather than filling the gap.

---

## 1. Cohort

| check | result | evidence |
|---|---|---|
| TUNE only | **confirmed** | `concept_audit_result_20261024_0130/outputs/run_manifest.json` → `data.cohort = "vietnam_tune"`, `report_partition_used: false`, `tbx11k_internal_test_used: false` |
| n = 1,352 | **confirmed** | same file, `data.n_cases: 1352`; preflight `cohort.n_cases: 1352` |
| fingerprint matches the pinned value | **confirmed** | `data.cohort_fingerprint_sha256 = 1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751`; preflight `fingerprint_matches_pinned: true`. The pinned value lives in `kaggle_exec_prep_20261023_1130/configs/tune_cohort_fingerprint.json` |
| composition | **confirmed** | 139 Xpert-positive, 466 previous TB (`preflight_kaggle.json` → `cohort`) |
| no unapproved image present | **confirmed** | `cohort.images_missing: 0`, `images_present_not_approved: 0` |
| no forbidden path visible | **confirmed** | `forbidden_paths_visible: []`, scan scope recorded in `partition_scan_scope` |

The ordered-id hash equals the cohort fingerprint here, which is expected: the score file
is already sorted ascending by `pid`, so load order and sorted order coincide.

## 2. Concept vocabulary and exclusions

| check | result | evidence |
|---|---|---|
| vocabulary identity | **confirmed** | `mimic_concepts.csv`, sha256 `93e975b722422b64…`, matched against the pinned value at run time (`preflight_kaggle.json` → `model.resolved.files`). This is the same file the indices were mapped against, so the pinned indices refer to the vocabulary actually loaded |
| vocabulary nature | 368,294 **free-text report sentences**, not a curated finding list | `dryad_rights_clear_audit_20261023_0900/CONCEPT_MAP.md`; `tables/concept_mapping.json` |
| 9 concepts used | **confirmed** | `run_manifest.json` → `concepts.names` |
| 3 excluded | **confirmed**: cavity, nodule, reticular opacity | `concepts.excluded`. Rule in `configs/concept_indices.json`: only an exact single vocabulary entry is used; concepts lacking one are *"EXCLUDED from the primary analysis and are not approximated by aggregating related sentences"* |
| why the exact-entry rule exists | negation is common among matching sentences — **28.6%** for consolidation, **24.5%** for focal opacity | `dryad_rights_clear_audit_20261023_0900/tables/concept_mapping.csv` |
| model pinned | HF revision `142f169c8b00…`; CLEAR package commit `57d2eae57dda…` | `run_manifest.json` → `model.revision`, `inference.clear_commit` |
| all three asset checksums matched | **confirmed** | `preflight_kaggle.json` → `model.resolved.files` vs `model.pinned_assets`, identical |

**Cavity is excluded.** It is the finding that most distinguishes active from healed
disease, and it was not measured at all. That is a limitation of the audit, not a result.

## 3. Multiple-testing family

| check | result | evidence |
|---|---|---|
| BH applied across the 9 concepts | **confirmed** | `concept_associations.csv`: exactly 9 of 36 rows carry a non-empty `p_bh_adjusted`, all in `negatives (all)` |
| subgroup p values unadjusted | **confirmed** | the other 27 rows have an empty `p_bh_adjusted` field |
| monotonicity | **confirmed by inspection** | every adjusted value ≥ its raw value |
| survivors | **8 of 9** at 0.05; pleural thickening does not (p_BH = 0.0747) | `concept_associations.csv` |
| bootstrap | 10,000 case-level resamples, seed 20261023 | `run_manifest.json` → `run.seed`, `run.n_boot` |
| patient-level = case-level | **confirmed** | Vietnam is one image per patient, 2,708 unique pid for 2,708 rows, asserted in code and in `prevtb_subgroup_20261021_1055/PREVIOUS_TB_SUBGROUP_REPORT.md:84-85` |

The effective number of independent tests is **smaller than 9** — scarring and fibrosis
are near-synonyms and their similarity vectors are certainly correlated. BH over 9 is
therefore conservative for the family as declared, but the family itself is not a set of
independent hypotheses, and the headline result is not close to any threshold
(p_BH ≈ 9e-103), so this does not change any conclusion.

## 4. The headline numbers, as recorded

From `concept_associations.csv` and `concept_incremental_r2.csv`, group
`negatives (all)`, n = 1,213:

| concept | ρ | 95% CI | p_BH | ΔR² |
|---|---|---|---|---|
| scarring | +0.5661 | 0.5276 – 0.6020 | 9.03e-103 | **+0.1975** |
| fibrosis | +0.4166 | 0.3693 – 0.4627 | 1.80e-51 | +0.0996 |
| consolidation | −0.3000 | −0.3511 – −0.2477 | 3.55e-26 | +0.0182 |
| calcified granuloma | −0.1652 | −0.2179 – −0.1105 | 1.29e-08 | **+0.0000** |
| volume loss | +0.1699 | 0.1150 – 0.2234 | 5.84e-09 | +0.0383 |
| bronchiectasis | +0.0815 | 0.0270 – 0.1373 | 6.28e-03 | +0.0097 |
| calcification | +0.0805 | 0.0253 – 0.1346 | 6.28e-03 | +0.0214 |
| focal opacity | −0.0795 | −0.1337 – −0.0228 | 6.28e-03 | +0.0079 |
| pleural thickening | +0.0512 | −0.0032 – 0.1062 | 0.0747 (ns) | +0.0066 |

Baseline R² = 0.2441 over scanner (2 dummies), age group, sex, symptom count and
`past_tb`. **ΔR² is in-sample and not cross-validated**, so a small positive value is
expected even for noise; the implementation's own test bounds the noise case below 0.02.

Discrimination of each concept against Xpert status (`concept_discrimination.csv`,
2,000 resamples): **scarring AUROC 0.4941 [0.4454, 0.5422]** — chance. bronchiectasis
0.6202, volume loss 0.5993, fibrosis 0.5794, pleural thickening 0.3358.

## 5. Interpretation checks required by the brief

| required statement | status |
|---|---|
| `past_tb` is clinical history, not an image annotation of fibrosis or scarring | **confirmed and documented.** It is a self-reported history field in the Dryad clinical CSV. `prevtb_subgroup_20261021_1055/.../verify_subgroup.py:135` marks it *"Self-reported; not record-verified. UNVERIFIED either way."* |
| concept-text similarity is a weak proxy, not radiologist ground truth | **confirmed and machine-recorded.** `run_manifest.json` → `concepts.score_kind`: *"text-similarity to one report sentence per concept: a weakly supervised proxy, NOT a validated detector"* |
| correlation / ΔR² does not establish that the classifier causally uses a visual cue | **confirmed.** Stated in `concept_audit_result_20261024_0130/RESULTS.md` §8 and in the run's own closing output |
| no TB score, model weight or threshold was modified | **confirmed.** `run_manifest.json` records inference provenance only; no weight or prediction file was written; the repository's tracked files are unchanged at HEAD `f878a6c` |

## 6. Two things that weaken a naive reading

**Consolidation runs the wrong way.** ρ = −0.300 among negatives and **−0.385 among
positives** (`concept_associations.csv`, rows for group `positives (all)`). Consolidation
is the canonical active-TB finding; the classifier's score *falls* as consolidation
similarity rises, in the group that actually has TB. Either these directions do not mean
what their names say, or something is inverted. **No explanation for this is available
from the artifacts**, and it is the strongest reason not to treat the nine axes as
radiological findings.

**No orthogonal negative control.** All nine concepts are TB-adjacent. The differentiated
pattern (consolidation negative, pleural thickening null, calcified granuloma ΔR² = 0.000)
argues against a single global confound, but it is a substitute for a control, not a
control. This is a design omission.

## 7. Unverifiable from artifacts

- **CLEAR's exposure to this cohort.** The model card names three training corpora and no
  evaluation set, so overlap with the Vietnam cohort is **unknown rather than excluded**
  (`dryad_rights_clear_audit_20261023_0900/MODEL_LICENCE_ASSESSMENT.md`).
- **RAD-DINO's exposure.** Recorded as `UNKNOWN` in the existing exposure registry; the
  model card does not name its checkpoint-selection datasets.
- **Whether the nine similarity vectors are mutually near-collinear.** The per-case matrix
  was deliberately not exported, so the concept-by-concept correlation structure cannot be
  computed from what is on disk. It would need a re-run that exports a 9×9 correlation
  matrix — an aggregate, not a per-case table.
