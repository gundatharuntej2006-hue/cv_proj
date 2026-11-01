# Phase 0 — state and provenance audit

**Read-only.** Nothing in this phase fitted, scored or modified anything.

| | |
|---|---|
| git HEAD before | `f878a6c9e9746803938463941ae1ffaa5d1fd96a` |
| working tree | 7 tracked files modified by **earlier** documentation tasks (`docs/p2_external_validation.md`, `docs/p2_focal_pooling_final.md`, `docs/p2_vietnam_result.md`, `report/p2_00_abstract_intro.md`, `report/p2_99_conclusion.md`, `report/paper_draft.md`, `report/slides/p2_slides_outline.md`). Left untouched. Captured verbatim in `tables/git_status_before.txt` |
| protected artifacts | 14 files hashed before the run — `tables/protected_hashes_before.txt` |
| new work | confined to this directory |

---

## 1. The two target columns, and where their meaning actually comes from

Both come from the **dataset's own COCO annotations**, not from a project convention.
`data/raw/tbx11k/TBX11K/annotations/json/TBX11K_{train,val}.json` declares, verbatim:

```json
"categories": [
 {"id": 1, "name": "ActiveTuberculosis",            "supercategory": "Tuberculosis"},
 {"id": 2, "name": "ObsoletePulmonaryTuberculosis", "supercategory": "Tuberculosis"},
 {"id": 3, "name": "PulmonaryTuberculosis",         "supercategory": "Tuberculosis"}
]
```

Lesion-box category sets over the 800 trainval TB images, counted directly from the
annotation files:

| box category set | n | project category | `tb_label` | `active_tb_label` |
|---|---|---|---|---|
| `[1]` ActiveTuberculosis only | 630 | `active_tb` | 1 | 1 |
| `[1, 2]` both | 30 | `active_latent_tb` | 1 | 1 |
| `[2]` ObsoletePulmonaryTuberculosis only | **139** | `latent_tb` | **1** | **0** |
| no box at all | 1 (`tb1199`) | see §4 | 1 | disputed |

**Category 3 is never used in trainval** — zero images carry it, so the "unspecified"
class creates no ambiguity here. That was worth checking, because an image labelled only
`PulmonaryTuberculosis` would have no defensible active/inactive assignment.

So the reported mismatch is real: **139 images** carry only obsolete-TB boxes and are
positive under `tb_label`, negative under `active_tb_label`. Under the C01 split file the
mismatch set is **140**, because C01 also places `tb1199` (the one box-less TB image) in
`latent_tb` — see §4. Both numbers are correct for what they count, and the experiment
uses C01, so **140** is the number of images the arms actually relabel.

It is not a bug — the two columns encode two different questions, and only one of them is
the question the system is evaluated on.

## 2. Partitions (`artifacts/p2/dryrun/provisional_c01.csv`)

| split | n | healthy | sick_non_tb | active | latent | active+latent | pos `tb_label` | pos `active_tb_label` |
|---|---|---|---|---|---|---|---|---|
| train | 5,941 | 2,700 | 2,700 | 426 | 94 | 21 | 541 | 447 |
| development | 659 | 300 | 300 | 47 | 10 | 2 | 59 | 49 |
| calibration | 899 | 400 | 400 | 78 | 18 | 3 | 99 | 81 |
| **internal_test** | **901** | 400 | 400 | 79 | 18 | 4 | 101 | 83 |

Full table: `tables/partition_table.csv`. Crosswalk: `tables/target_crosswalk.csv`.

**This is the *provisional* C01.** The definitive split from P1 is still outstanding.
Every number produced here is conditional on it, and would have to be recomputed against
the definitive split before entering the paper.

## 3. `internal_test` — the 901 flag from the previous audit is RESOLVED

The previous audit (`artifacts/report_exposure_audit_20261024_0215/UNRESOLVED_ITEMS.md`,
item U1) flagged `artifacts/p2/sealed/sealed_result_rehearsal1.json` — n = 901,
n_tb = 101, AUC 0.992 — as possibly a scored pass over `internal_test`, inferred from the
sample count alone. **Metadata resolves it, without opening the split.**

`artifacts/p2/sealed/seal_rehearsal1.json` carries:

> `"note": "rehearsal on provisional C01; not the real one-shot read"`

and the partition table above shows the provisional C01 `internal_test` split holds
**exactly 901 images with 101 `tb_label` positives** — both numbers matching the sealed
result exactly.

**What this establishes.** The 2026-10-14 rehearsal scored the **provisional** C01
`internal_test` partition with an earlier model (`radino_full_clahe0_bin`, weight sha
`14a7e6cf...`), as a dry run of the sealing machinery.

**What it does not establish.** That those 901 images are still the definitive
`internal_test`. The seal `split_sha256` (`00417da6...`) does **not** equal the sha-256 of
`provisional_c01.csv` (`b228f85c...`), so it is a hash of some canonicalised form that
cannot be re-derived from what is on disk.

**Net:** the project claimed-pristine partition **has been scored once**, on a provisional
definition, by an earlier model. Whether that overlaps the definitive `internal_test` is
unknown and must be settled by P1/P4, not by opening the split. `internal_test` was
**not** read in this run.

## 4. A defect found in the shipped label table

`artifacts/p2/labels_tbx11k_trainval_v1.csv` and `provisional_c01.csv` **disagree on one
image**, `tb1199` — the single TB image with no lesion box:

| source | category |
|---|---|
| `labels_tbx11k_trainval_v1.csv` | `active_tb` |
| `provisional_c01.csv` | `latent_tb` |

`scripts/p1_make_tbx11k_labels.py` documents the intended resolution and sets
`NO_BOX_FALLBACK = "latent_tb"`, reasoning from the published train counts
(473 / 104 / 23): box-derived counts give 473 / 103 / 23, so `tb1199` can only be the
104th latent case. **C01 follows that reasoning; the v1 label CSV does not**, so the CSV
appears to predate the script as it now stands.

**Resolution used here:** the C01 `category` column is authoritative, because it matches
the published counts *and* is what the production training path reads. Impact: one image
in **train**. It cannot move any reported figure. Recorded so it is fixed rather than
inherited.

Two further quirks, both handled by deriving targets from `category` rather than trusting
a stored column:

- C01 stores `active_tb_label` as an **empty string** for all 139 latent rows, not `0`.
- Apart from `tb1199`, both tables' `tb_label` and `active_tb_label` columns are exactly
  reproducible from `category` — 8,400 rows checked, 0 other mismatches
  (`tables/phase01_audit.json`).

## 5. Patient grouping — not available, and not claimable

TBX11K ships **no patient identifier**. C01 and the label table carry `image_id` only.
Folds are image-level. Nothing in this run may be described as patient-grouped or
site-held-out.

What *can* be quantified is near-duplicate leakage. P2-04 found 158 cross-split
near-duplicate candidates at cosine >= 0.95 against the *official* TBX11K splits; mapped
onto the C01 partitions they fall out as:

| partition pair | pairs |
|---|---|
| train <-> calibration | **22** |
| train <-> internal_test | 24 |
| development <-> internal_test | 3 |
| calibration <-> development | 1 |
| at least one image outside C01 (extra/NLM) | 108 |

**23 of the 899 calibration images (2.6%)** have a near-duplicate in another partition
(`tables/near_duplicate_cross_split.csv`). These pairs are still unreviewed — P1-07 is
outstanding. They inflate absolute AUROC on calibration. They affect **both arms
identically**, so a *paired* difference is far less sensitive to them than an absolute
number, but the protocol pre-declares a leave-them-out sensitivity analysis anyway.

## 6. The fitting path to be held constant

Read from `scripts/p2_32_mil_sweep.py` and `scripts/p2_33_mil_final.py`:

| | |
|---|---|
| backbone | RAD-DINO ViT-B/14, frozen, rev `110cbc18` — **not re-run**; cached regional embeddings are reused |
| features | `artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz` (12x12 lungs-masked, L2-normalised per region) |
| head | one `nn.Linear(768 -> 1)` applied per region, then **max** over the 144 regions |
| optimiser | AdamW, lr 3e-3, batch 256, weight decay 1e-4, <=120 epochs |
| imbalance | logit adjustment by the training prior |
| early stopping | on **development** AUROC, patience 15 |
| seeds | 17, 29, 43, 61, 79 — seed-averaged scores |
| eligible training rows | `split == train` **and** `category in {healthy, active_tb, latent_tb, active_latent_tb}`; the 2,700 `sick_non_tb` train images are excluded by the current configuration |

`p2_32_mil_sweep.train` / `.score` / `.repool` are **imported unmodified** by this run's
experiment script. Nothing in `scripts/` or `src/` is edited.

**Consequence that shapes the design:** development is consumed by early stopping, in both
arms, against each arm's own target. A paired AUROC comparison *on development* would
therefore be contaminated by arm-specific model selection. The protocol makes
**calibration** the primary evaluation split instead — it is untouched by the regional-MIL
path (it was used only to fit thresholds for the earlier global-pooling head, which does
not affect AUROC).

## 7. The prior label experiment, and why repeating it is still exploratory

`artifacts/realworld_generalization_20261021_1015` ran a four-arm preregistered test
**evaluated on Vietnam**:

| arm | what changed | Vietnam primary endpoint | verdict |
|---|---|---|---|
| H | baseline, `tb_label` | 0.6173 | — |
| **L** | latent TB -> negative | 0.6465, **delta +0.0292** [+0.0223, +0.0363] | secondary; **not promoted** (below the +0.05 declared minimum) |
| S | `sick_non_tb` added as negatives | 0.6037, delta -0.0136 [-0.0352, +0.0073] | no effect |
| X | L + S, the **preregistered primary** | 0.6248, delta +0.0074 [-0.0123, +0.0264] | **INCONCLUSIVE** |

Arm L is the same label change as this run Arm B. **So why run it again?**

1. **It was measured on an exposed cohort.** Both Vietnam halves had already informed
   decisions, and the hypothesis was generated by inspecting the pooled 2,708. This run
   measures the same change on **TBX11K calibration**, a partition the regional-MIL
   pipeline has never used — a different, internal, held-out estimate.
2. **It was confounded with the primary arm.** X bundled the label change with a
   negative-composition change and came out at +0.0074 while L alone came out at +0.0292.
   This run isolates the label change and nothing else.
3. **It has never been evaluated against the target it is supposed to fix.** Arm L was
   scored against Xpert on Vietnam. Nobody has asked whether training on
   `active_tb_label` improves detection of *active* TB on the TBX11K active-TB labels.

**What it cannot do.** It cannot confirm arm L. The hypothesis is already in hand, the
point estimate is already known, and the evaluation split, while unused by this pipeline,
sits inside the same dataset the head is trained on. This run is **exploratory** and is
labelled so throughout.

## 8. Cohorts accessed by this run

| partition | accessed | how |
|---|---|---|
| TBX11K train (5,941) | yes | fitting, both arms |
| TBX11K development (659) | yes | early stopping + threshold fitting |
| TBX11K calibration (899) | yes | **primary evaluation** |
| TBX11K `internal_test` (901) | **NO** | never loaded; guarded by an assertion and a test |
| Vietnam TUNE (1,352) | yes | exploratory secondary only, no threshold fitted on it |
| Vietnam REPORT (1,356) | **NO** | never loaded; guarded by an assertion and a test |
| Shenzhen / Montgomery / DA / DB | no | out of scope |

Exposure status of each, carried forward: TBX11K train/development/calibration are
development data and are exposed by construction. Vietnam TUNE is **heavily exposed** — it
selected the pooling configuration. Vietnam REPORT is **exposed** (six scoring events; see
`artifacts/report_exposure_audit_20261024_0215/`). `internal_test` is **exposed once**
under a provisional definition (§3).

## 9. Blocking ambiguities — none

The one decision gate that could have blocked Phase 2 (§1, whether `active_tb_label` is
the documented active-TB target) is settled by the dataset own category declarations. The
three defects found — `tb1199`, the blank `active_tb_label` column, the 23 near-duplicate
calibration images — are documented and handled, not blocking.
