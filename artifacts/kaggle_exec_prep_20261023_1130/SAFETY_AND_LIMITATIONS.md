# Safety and limitations

Read this before writing up a single number from this run.

---

## 1. Partition controls

| control | where | can it actually fire? |
|---|---|---|
| **Allow-list selection** | `load_tune_only`: rows are kept only if `pid in tune` | It is the boundary, so it cannot "fire" — contamination is prevented *by construction*. A test asserts the allow-list is still how selection works, because an exclusion-list rewrite would look equivalent and would not be |
| **TUNE-only upload** | `stage_kaggle_data.py` filters both CSVs and copies only allow-listed images | REPORT clinical rows, REPORT scores and REPORT images are **not uploaded at all**. A second, physical boundary in front of the first |
| **REPORT intrusion check** | `load_tune_only`, staged layout only | **Yes** — a REPORT row in a staged upload means it was built wrongly, and the run stops. Inert in the repository layout, where all 2,708 rows legitimately exist; the doc says so rather than implying a guard that does nothing |
| **Split-manifest overlap** | before any row is read | **Yes** |
| **Missing / duplicate id, count mismatch** | after selection | **Yes**, each separately tested |
| **Cohort fingerprint** | against `configs/tune_cohort_fingerprint.json` | **Yes** — a re-cut split is caught before any GPU work |
| **Locked-partition scan** | `preflight.py`, over attached inputs | **Yes** — a blocker, not a warning |
| **Zip member scan** | `preflight.py`, when the upload is an archive | **Yes.** A zipped upload hides its contents from a filesystem walk, so member names are scanned too; otherwise the scan would silently pass on exactly the layout in use |
| **Per-archive scratch key** | `paths._zip_scratch` | **Yes.** The extracted small files are keyed to the archive's path, size and mtime. Keying on the filename alone let two different `upload.zip` archives share a cohort — caught by test, fixed |
| **Source reference scan** | every data module is checked for a reference to the locked partition | **Yes**. `preflight.py` is exempt by filename because it must name the partition to exclude it; a new module cannot opt itself out |
| **Resume safety** | shards keyed to the hash of the ordered id list | **Yes** — a resume onto a different id list is refused, not merged |

**TBX11K `internal_test` is not read, mounted, listed, scored or referenced at any point.**
Vietnam **REPORT** is not read, scored or used for any statistic. Neither appears in the
upload.

One honest note about the previous package, carried forward: a test there asserted that
injecting a REPORT row into the score file raised an error, and it did not — because the
allow-list filters the row out before the check. The guard was dead code giving false
assurance. It was replaced with guards that can fire, and the dead check is not reinstated
here.

## 2. Licensing — what is settled and what is not

| question | answer |
|---|---|
| May the Vietnam images be analysed? | **Yes** — CC0-1.0, dataset-specific Dryad API field |
| May they be redistributed? | Legally yes; **upload Private anyway** |
| May CLEAR be run on them? | **Yes** — Apache-2.0, revision pinned and checksum-verified |
| May RayDINO be run on them? | **No.** Gated weights, FAIR Noncommercial Research Licence. Out of scope |
| Does CC0 on the data imply anything about the models? | **No.** Separate instruments |

**CC0 waives copyright and database rights. It does not waive — and cannot waive —
ethics, privacy, institutional, human-subjects or platform obligations.** These are
de-identified chest radiographs of real patients. A public-domain dedication is not an
ethics approval, is not consent for a new purpose, and is not a finding that
re-identification risk is zero. Nothing is uploaded publicly, nothing is emailed, and no
image or identifier leaves the two private datasets.

This artifact package itself contains **no images, no clinical rows, no model weights and
no patient identifiers** — only code, configuration and hashes. A test enforces all four,
including scanning every text file in the package for real patient ids.

## 3. What a concept score is

Every one of the 9 scores is a **cosine similarity between a CLEAR image embedding and
the embedding of one free-text sentence** from a 368,294-entry vocabulary of observations
extracted from MIMIC-CXR reports.

| | |
|---|---|
| a direct model output? | **No** |
| a validated detector? | **No** |
| radiologist ground truth? | **No** |
| what it is | a **weakly supervised proxy**: similarity to one sentence, in a space learned from report text |

So a high `fibrosis` similarity is **not fibrosis detection**. The manifest records this
in `concepts.score_kind` so the caveat travels with the numbers, and a test asserts that
field still says it. Do not call any result "fibrosis detection", "cavity detection" or
"the model sees scarring".

**cavity, nodule and reticular opacity are excluded**, because no exact single vocabulary
entry exists for them and the project will not approximate a concept by pooling sentences
after seeing the data. Cavity matters clinically in TB. Its absence is a **limitation of
this audit**, not a finding about the images.

Negation is why the exact-entry rule exists: among sentences matching `consolidation`,
**28.6 %** assert its absence, and **24.5 %** for `focal opacity`. Pooling them would mix
assertion with denial.

## 4. Interpretation boundaries

1. **TUNE is exposed.** It has been used for selection throughout the project. Everything
   here is **exploratory**, never validation, and never a basis for a threshold.
2. **`past_tb` is patient-reported clinical history, not an image annotation.** It does
   not mark fibrosis, healed TB or a cavity on the radiograph, and must not be read as
   doing so.
3. **Association is not causation.** A correlation between a concept similarity and the
   classifier score does not show the classifier uses that visual cue — both could track a
   third thing, including scanner, body habitus or acquisition.
4. **No concept score may modify a TB score.** No subtraction from active-TB scores, no
   suppression of referrals, no recalibration, no threshold fitted on these results. This
   is a diagnostic read, not an intervention, and an intervention must not be optimised on
   the same data that suggested it.
5. **In-sample R².** `delta_r2` is not cross-validated; a small positive value is expected
   even for noise.
6. **AUROC/AUPRC of a concept against the TB label is descriptive.** A concept score is
   not a TB classifier and was not trained to be one.
7. **Multiplicity is corrected once**, by Benjamini–Hochberg across the 9 concepts in the
   `negatives (all)` family. The subgroup rows are **not** a second corrected family;
   their `p` values carry no correction and are exploratory. Reporting a subgroup p as
   though it were corrected would be wrong.
8. **Exposure is unknown.** CLEAR's card names three training corpora and no TB benchmark,
   but also names no evaluation set, so overlap with this cohort is **unknown rather than
   excluded**. If CLEAR has seen it, any association here is potentially exposed.
9. **Preprocessing fidelity is unconfirmed** — see `PREFLIGHT_REPORT.md` §5.
10. **Nothing here is a clinical-readiness, novelty or causality claim**, and the screening
    AUROC remains **0.7242** whatever the tables say.

## 5. The likely result, stated in advance

So that a confirmation is not mistaken for a discovery: the expected outcome is that a few
chronic-abnormality concepts correlate modestly with the classifier score among
TB-negative participants, more so among those with previous TB. That would be **consistent
with** the previously measured previous-TB score elevation, would **not** be independent
evidence for it, and would **not** show the classifier uses a particular finding. The
four-condition rule in `DECISION_RULES.md` of the previous package still governs whether
anything here becomes actionable; a score-by-history association alone does not qualify.
