# Documentation consistency check — 2026-10-21

**Repository HEAD, unchanged:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` (`master`)
**No commit, no push, no deletion, no rename.** No model, checkpoint, cache, prediction
file or raw data touched. Six tracked documentation files edited; full diff in
`tables/documentation.diff` (268 lines). Pre-edit copies of all eight touched files in
`backups/`, hashes in `tables/pre_edit_hashes.json` and `tables/post_edit_hashes.json`.

---

## 1. Discrepancy found during verification, before any edit

The task brief quoted the DA+DB pooled matched-global AUROC as **0.9172**. The committed
artifact gives **two** values, for two declared arms:

| arm | pooled AUROC (6 dp) | 4 dp |
|---|---|---|
| `A_primary_5seed` — 5-seed ensemble, the **primary** matched arm | 0.917064 | **0.9171** |
| `A_secondary_seed17` — seed 17 only, the `p2_36` replica | 0.917197 | **0.9172** |

So **0.9171** is the correct headline pooled figure and **0.9172** belongs to the
seed-17 replica. The brief's figure is not wrong, it is the other arm.

`negative_composition_check.json` legitimately uses **0.9172**, because that comparison is
seed-17 against seed-17 on both sides and is therefore internally matched. The previous
run's `RESULTS.md` §5 labelled that row "matched arm A" without saying which seed
convention it used, which invited exactly this confusion. **Corrected** — see §3.

Everything else in the brief verified exactly. No other discrepancy was found, so editing
proceeded.

## 2. Verification results — all figures traced to artifacts

| claim | artifact | result |
|---|---|---|
| DA n=72, TB=22; global 0.9764; regional 0.9755; Δ −0.0009 [−0.0357, +0.0299] | `dadb_focal_pooling_20261021_0905/tables/metrics.json` | **verified** |
| DB n=104, TB=52; global 0.8780; regional 0.9497; Δ +0.0717 [+0.0159, +0.1328] | same | **verified** |
| pooled global 0.9171 / regional 0.9636; Δ +0.0465 [+0.0091, +0.0888] | same | **verified** (0.9172 → see §1) |
| excluding `px4`: pooled Δ +0.0454, DB Δ +0.0701, no conclusion change | same, `sensitivity_175_excl_px4` | **verified** |
| all-negatives config reproduces DA+DB 0.7791, DA 0.8691, DB 0.7973 | `tables/negative_composition_check.json`, cross-checked against `artifacts/p2/external/dadb_third_site.json` | **verified, exact** |
| healthy-only config gives 0.9172 (seed 17) / 0.9636 | same | **verified** |
| Vietnam 0.7242 | `artifacts/p2/regional/mil_final.json` → `Vietnam REPORT`, grid 12 / agg max / seeds 17-79 | **verified**; n=1,356, 141 TB |
| Vietnam full-cohort global 0.6743 | `artifacts/p2/dryrun/hard_negatives.json`, lungs_only + healthy_only | **verified**; n=2,708 |
| Shenzhen 0.9411 / 0.9690; Montgomery 0.9289 / 0.9690 | `artifacts/p2/audit/cross_dataset_metrics.csv` | **verified** |
| TBX11K dev 0.9570 / 0.9368 | `dadb_focal_pooling_…/tables/experiment_manifest.json` | **verified** |
| duplicates 0, labels re-derived match, grid (176,144,768), 19+228 tests | `metrics.json` → `integrity`, `test_results.txt` | **verified** |

**Characterisation check.** The comparison 0.7791 → 0.9172 is described everywhere as a
**training-negative-composition** comparison, never as a pooling-only comparison. Verified
by reading each edited passage.

**Exact run identifiers.** All-negatives side: the `p2_26_dadb_third_site.py::train_head`
convention (`c01[c01.split == "train"]`, **no category filter**, 5,941 rows), replicated in
`dadb_focal_pooling_20261021_0905`. Healthy-only side: the same run's arm A (3,241 rows).
Both use seed 17.

## 3. Files and sections changed

### Tracked documentation

| file | section | change |
|---|---|---|
| `docs/p2_vietnam_result.md` | ladder table (line ~25) | Added a **configuration notice** marking the existing table as full-image/all-negatives, and a new section *"The same cohorts under the current configuration"* with the per-cohort table, per-site Δ with CIs, the supplementary-pooling caveat, and the negative-composition explanation. Existing table left intact. |
| `docs/p2_external_validation.md` | §"Third site (2026-10-19)" | Added a **superseded-in-part** banner; qualified "performance degrades further" as configuration-specific; named `px4` and recorded that it was counted but not excluded; added a new section *"DA/DB under the current configuration"* with per-site results, the pooled caveat, the sensitivity analysis and the negative-composition explanation. |
| `report/p2_00_abstract_intro.md` | after the ladder paragraph | Added a **configuration note** block giving the current-configuration figures and stating the ladder is **steeper**, not shallower, so the paper's argument is unaffected. |
| `report/p2_99_conclusion.md` | §"external validation conceals a ladder" | Added a paragraph giving the current-configuration rungs and stating the ladder holds in both configurations. |
| `report/paper_draft.md` | §4 table | Added a sentence stating all three rows use all negatives and pointing to §5.0. |
| `report/paper_draft.md` | after §5 ladder table | Added **§5.0**, the same ladder under the adopted configuration, with the three "read carefully" cautions (DA/DB disagree; pooled is supplementary; Vietnam rows have different denominators) and the negative-composition explanation. |
| `report/slides/p2_slides_outline.md` | slide 10 | Added a speaker note with the current figures and an explicit instruction not to say regional pooling improves every cohort. |

### Prior run artifacts, corrected for accuracy

| file | change |
|---|---|
| `artifacts/dadb_focal_pooling_20261021_0905/RESULTS.md` §5 | Disambiguated 0.9172 (seed 17) from 0.9171 (5-seed primary) and stated both sides of the negative-composition comparison use seed 17. Added that healthy-only negatives do **not** establish a general improvement, with the 0.7242 screening figure. Replaced "the difference is the population, not the model" with the correct account: **AUROC is prevalence-invariant** when class-conditional distributions are fixed, so prevalence is not the explanation; listed the plausible contributors and stated that the evidence identifies none of them. Marked the ladder correction as **done**. |
| `artifacts/dadb_focal_pooling_20261021_0905/LIMITATIONS.md` §4 | Added the same prevalence-invariance correction and the list of plausible contributors. |

**Nothing was deleted.** Every historical figure remains in place, labelled with the
configuration that produced it.

## 4. Claims checked and found absent or corrected

| unsupported claim | status |
|---|---|
| novel pooling architecture | absent; prior art (RADAR) acknowledged in `paper_draft.md` §1.1 and `docs/p2_focal_pooling_final.md` |
| independent external validation of exposed cohorts | absent; Shenzhen/Montgomery/Vietnam exposure is stated in each |
| clinical readiness | absent |
| causal proof that scanner shortcuts explain site differences | absent; the subspace experiment was **rejected** and the INLP result argues against it |
| generalisation to all hospitals | absent |
| "regional pooling improves every cohort" | **never asserted**; the two grep hits are the explicit negations added today |
| "four cohorts prove generalization" | absent |
| pooled DA+DB presented as a single-site result | **prevented**: every occurrence is italicised and labelled supplementary |
| prevalence as the cause of the screening gap | **corrected** in two files (§3) |

## 5. Checks and commands run

```
git rev-parse HEAD ; git status --short              # HEAD unchanged, only untracked run dirs
py -3 -m pytest artifacts/dadb_focal_pooling_20261021_0905/tests -q     # 19 passed
py -3 -m pytest artifacts/overnight_scanner_run_20261021_0208/tests -q  # 11 passed
py -3 -m pytest -q                                                      # 228 passed, 1 skipped
py -3 artifacts/doc_finalisation_20261021_0930/scripts/power_planning.py
git diff --numstat ; git diff > tables/documentation.diff
```

Full production-suite output: `tables/production_suite.txt`.

**One test failure, explained and not a defect.** Running the `dadb_focal_pooling` and
`overnight_scanner_run` suites **in the same pytest process** fails
`test_streamed_rows_are_l2_normalised` with a CUDA allocation error: both suites load
feature tensors onto a 6.4 GB GPU and the first suite's tensors are still resident. Each
suite passes cleanly in isolation (19 and 11). This is test-harness VRAM contention, not a
code or result defect, and no result depends on it.

**No new training or scoring was run.** `power_planning.py` resamples the committed
Vietnam per-case score file; it loads no model and scores no cohort.

## 6. Figures that could not be verified

| figure | status |
|---|---|
| RAD-DINO checkpoint exposure to any evaluation cohort | **UNKNOWN** — the model card does not name its checkpoint-selection datasets |
| patient-level overlap within or across DA/DB | **UNVERIFIABLE** — no patient identifiers distributed |
| Vietnam global-pooling AUROC *on the REPORT half specifically* | **NOT AVAILABLE** — only the full-cohort 0.6743 exists; computing it would require rescoring, which this task forbids. Documents state the denominators explicitly instead. |
| per-cohort CIs for the old all-negatives Shenzhen/Montgomery rows under the current config | **not recomputed** — out of scope; existing CIs retained with their original configuration labelled |

Everything else quoted in the edited documents traced to a committed artifact, and a
programmatic check re-verified all twelve headline figures against their source files
(all **OK**).
