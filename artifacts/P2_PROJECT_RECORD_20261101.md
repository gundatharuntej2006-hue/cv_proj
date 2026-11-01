# TB-Triage, Person 2 — complete project record

**Compiled:** 2026-11-01
**Repository:** `D:\tb-triage`
**Git HEAD at time of writing:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`
**Working tree:** 7 tracked documentation files modified, uncommitted; 21 untracked
`artifacts/*` run directories

> **Written for:** Atul (P2) as the single place the whole P2 thread is recorded, and for
> P1/P3/P4 or an examiner who needs to understand what was done without reading 21 run
> directories. Every figure here is traced to a file on disk. Where something is
> unverified it says so.

---

# PART I — THE PROJECT

## 1. What this is

BAI506 Computer Vision mini project, BMSIT&M, 5th semester. Four people:

| | |
|---|---|
| P1 | Tharun — data, splits, label tables |
| **P2** | **Atul — classifier, evaluation, everything in this record** |
| P3 | Ritika |
| P4 | Kunal — sealed-evaluation machinery |

**Title:** *Explainable TB Screening from Chest X-Rays with Lesion Localisation for
Triage.*

**What P2 actually built:** a frozen-backbone tuberculosis screening classifier, evaluated
honestly — first on curated benchmark data, then on a programmatic risk-based active-case-finding cohort,
where it performed far worse. Most of the work since has been explaining that gap rather
than closing it.

## 2. Data

| dataset | n | role | notes |
|---|---|---|---|
| **TBX11K** | 8,400 trainval (core lists) | training + internal evaluation | COCO lesion boxes; categories `ActiveTuberculosis` (1), `ObsoletePulmonaryTuberculosis` (2), `PulmonaryTuberculosis` (3) |
| **Vietnam (Dryad)** | 2,708 participants | the real screening cohort | Xpert MTB/RIF Ultra reference standard; Codlin et al. 2025; DOI `10.5061/dryad.5tb2rbpcv` |
| Shenzhen | 662 (336 TB) | external | NLM |
| Montgomery | 138 (58 TB) | external | NLM |
| DA / DB | 176 (74 TB) | external | TBX11K extra lists |

### TBX11K partitions (`artifacts/p2/dryrun/provisional_c01.csv`)

| split | n | healthy | sick_non_tb | active | latent | active+latent | pos `tb_label` | pos `active_tb_label` |
|---|---|---|---|---|---|---|---|---|
| train | 5,941 | 2,700 | 2,700 | 426 | 94 | 21 | 541 | 447 |
| development | 659 | 300 | 300 | 47 | 10 | 2 | 59 | 49 |
| calibration | 899 | 400 | 400 | 78 | 18 | 3 | 99 | 81 |
| **internal_test** | **901** | 400 | 400 | 79 | 18 | 4 | 101 | 83 |

**This is the *provisional* C01.** P1's definitive split has never been delivered. Every
TBX11K number in this record is conditional on it.

### Vietnam partitions (`artifacts/p2/external/vietnam_tune_report_split.json`)

Split once, before any tuning, stratified by outcome and scanner, seed 20261020:

| half | n | Xpert-positive | role |
|---|---|---|---|
| **TUNE** | 1,352 | 139 | every tuning decision made here |
| **REPORT** | 1,356 | 141 | intended as read-once |

TUNE cohort fingerprint (sha-256 of sorted pids):
`1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751`. 466 of the 1,352 TUNE
participants have a history of previous TB.

## 3. The model

| | |
|---|---|
| backbone | **RAD-DINO ViT-B/14**, `microsoft/rad-dino`, revision `110cbc18d5133582e320b43d53bf5c44e410c936`, MIT, **frozen** |
| features | 512 px, lungs-masked, patch tokens average-pooled to a **12×12** grid of 768-d region vectors, L2-normalised per region |
| head | one shared `nn.Linear(768 → 1)` applied per region, **max** over the 144 regions |
| training | AdamW, lr 3e-3, batch 256, weight decay 1e-4, ≤120 epochs, early stopping on TBX11K development AUROC (patience 15), logit adjustment by the training prior |
| seeds | 17, 29, 43, 61, 79 — scores are the seed mean |
| eligible training rows | `split == train` and `category ∈ {healthy, active_tb, latent_tb, active_latent_tb}` → **3,241 images, 541 positive**. The 2,700 `sick_non_tb` train images are excluded |
| comparator | Lunit INSIGHT CXR 3.9.0.1, scored on identical patients |

Selection was done on the Vietnam TUNE half only (`mil_sweep.json`, `selected_on:
"vietnam TUNE half only"`, best grid 12 / max, TUNE 0.7239).

## 4. The headline results (`artifacts/p2/regional/mil_final.json`, 2026-10-20)

| cohort | n | TB | **AUROC** | 95% CI | Lunit |
|---|---|---|---|---|---|
| TBX11K development | 659 | 59 | **0.9368** | [0.9047, 0.9647] | — |
| Vietnam TUNE | 1,352 | 139 | 0.7237 | [0.6846, 0.7626] | 0.7780 |
| **Vietnam REPORT** | 1,356 | 141 | **0.7242** | [0.6843, 0.7618] | 0.7862 |

Selection optimism (TUNE − REPORT) = **−0.0005**, so the sweep did not overfit the tuning
half.

**This is the central fact of the project: 0.9368 internally, 0.7242 on a real screening
cohort.** The commercial comparator also drops, but less (0.7862).

Other configurations, for context — note each is a *different* head, so the numbers are
not interchangeable:

| configuration | TBX11K dev | elsewhere |
|---|---|---|
| full-image global head | 0.9950 | Shenzhen **0.9382**, Montgomery **0.8183** |
| global pooler (DA/DB run, arm A) | 0.9570 | DA+DB pooled 0.9171 |
| **regional 12×12 max (current)** | **0.9368** | DA+DB pooled 0.9636; Vietnam REPORT 0.7242 |

At Montgomery, at the TBX11K-fitted threshold, sensitivity is **0.6724** — one TB case in
three missed. Local recalibration recovers Shenzhen (0.9345 sensitivity) but not
Montgomery (0.7759).

---

# PART II — THE RUN-BY-RUN RECORD

Each entry: the brief that was given, what was done, what came out.

## R1 · 2026-10-14 — sealed-evaluation rehearsal

`artifacts/p2/sealed/` · model `radino_full_clahe0_bin`, issuer `person4`

A dry run of the sealing machinery. **n = 901, n_tb = 101, AUC 0.992.** The seal's own
note: *"rehearsal on provisional C01; not the real one-shot read."*

**This is the provisional C01 `internal_test` partition** (901 rows, 101 `tb_label`
positives — exact match). So the project's only claimed-pristine partition **has been
scored once**, by an earlier model, under a provisional definition. Whether those 901
images are still the definitive `internal_test` is unknown — the seal's `split_sha256`
(`00417da6…`) is not the hash of `provisional_c01.csv` (`b228f85c…`) and cannot be
re-derived. *Resolved in R12; flagged in R10.*

## R2 · 2026-10-20 — headline evaluation

`artifacts/p2/regional/mil_final.json` · first REPORT read → **0.7242**. The script was
written before the sweep results were seen. No selection contamination found.

## R3 · 2026-10-21 02:08 — overnight scanner run

`artifacts/overnight_scanner_run_20261021_0208/`

**Brief:** test whether scanner/acquisition shift explains the screening gap, under a
preregistered sealed pass over REPORT.

| arm | result | verdict |
|---|---|---|
| **INLP scanner-subspace removal** (preregistered) | Δ **−0.0002** [−0.0009, +0.0004] | **REJECT** |
| **per-scanner offset correction** (exploratory, *not* preregistered) | 0.7242 → **0.7448**, Δ **+0.0206** [−0.0030, +0.0436], p = 0.1575 | CI crosses zero |
| per-scanner thresholds | scanner-3 sensitivity **0.674 → 0.488** | recommended against |

Stripping 44 of 768 scanner-predictive directions left TB AUROC flat while scanner
decodability stayed at 0.9118. **Scanner shift is not the cause of the gap.**

The exploratory +0.0206 later propagated into `power_planning.json` and
`NEXT_SCREENING_COHORT_PROTOCOL.md` as `target_delta_auroc`. An exploratory REPORT result
became a design parameter.

## R4 · 2026-10-21 09:05 — DA/DB focal pooling

`artifacts/dadb_focal_pooling_20261021_0905/` · regional pooling beats global on DA/DB
(pooled +0.0465 [+0.0091, +0.0888]). Also documented that **a pooled AUROC can move purely
because group composition moved** — within-group AUROCs unchanged while pooled shifted by
0.03. Seed spread 0.9625–0.9638, so not seed-hunting.

## R5 · 2026-10-21 10:15 — real-world generalization (4-arm preregistered)

`artifacts/realworld_generalization_20261021_1015/`

**Brief:** hashed preregistration before scoring; primary endpoint = AUROC for active TB
vs Xpert-negative participants *with past TB* (n = 1,130; 280 TB).

| arm | what changed | AUROC | Δ vs baseline | |
|---|---|---|---|---|
| H | baseline (`tb_label`) | 0.6173 | — | |
| **L** | latent TB → negative | 0.6465 | **+0.0292** [+0.0223, +0.0363] | excludes 0 |
| S | `sick_non_tb` added as negatives | 0.6037 | −0.0136 [−0.0352, +0.0073] | includes 0 |
| **X** | L + S — **the preregistered primary** | 0.6248 | **+0.0074** [−0.0123, +0.0264] | includes 0 |
| *Lunit, same patients* | | *0.7192* | | |

**Verdict: INCONCLUSIVE.** Arm L improved on all three endpoints but was a *secondary*
arm and its +0.029 fell below the declared +0.05 minimum. **It was correctly not
promoted.** Decision: **NO-GO on further model changes before representative data.**

## R6 · 2026-10-21 10:55 — previous-TB subgroup

`artifacts/prevtb_subgroup_20261021_1055/`

| | estimate | 95% CI |
|---|---|---|
| our degradation (no-prev-TB minus prev-TB) | **0.1629** | [0.1372, 0.1890] |
| Lunit degradation | 0.0978 | [0.0770, 0.1196] |
| difference in differences | +0.0651 | [0.0399, 0.0900] |

Ours: 0.7811 → 0.6182. Lunit: 0.8170 → 0.7192. All seven reproduction checks pass.

> **SUPERSEDED IN PART by R7.** The +0.0651 above uses this project's *shared-positive*
> construction. Recomputed under **construction A**, the construction used by the published
> literature (Qin 2024, Codlin 2025), the comparison against Lunit becomes
> **+0.0481 [−0.0170, +0.1113] — the interval INCLUDES ZERO and the claim is NOT
> statistically supported** (`artifacts/prior_art_review_20261021_1200/PRIOR_ART_MATRIX.csv`,
> row `our_contribution_2_model_vs_comparator`). Our own degradation under construction A is
> past TB 0.6392 [0.586, 0.692] vs no past TB 0.7720 [0.740, 0.803], degradation
> **+0.1328 [0.0718, 0.1951]** — still supported. **Only the comparison *against the
> comparator* fails.** Precision changes, not the point estimate.

**Critical provenance note:** this was computed on the **whole 2,708 cohort, TUNE and
REPORT pooled**. The finding that motivates everything downstream was characterised using
REPORT data.

`past_tb` is **self-reported history** from the Dryad clinical CSV — *"past history of TB:
0/No; 1/Yes"* — **not** a radiologist annotation of fibrosis.

## R7 · 2026-10-21 12:00 — prior-art review

`artifacts/prior_art_review_20261021_1200/`

**Killed a novelty claim.** The previous-TB effect is published — Kagujje 2022, Qin 2024,
and **Codlin et al. 2025, the paper that released this very dataset, on exactly these
2,708 participants with exactly our comparator version.** All six of Codlin's AUCs
reproduce from the local copy to two decimals.

| claim | status |
|---|---|
| that we discovered the previous-TB effect | **no** — three prior papers |
| that the question is open on this cohort | **no** — Codlin published it |
| that model-specific degradation is new | **no** |

## R8 · 2026-10-21 18:30 — CLEAR/RayDINO package

`artifacts/kaggle_concept_raydino_20261021_1830/` · 27 files, 29 tests, **never executed**.
RayDINO blocked on **gated FAIR Noncommercial weights**, unresolved redistribution terms,
and unpublished input resolution. Re-verified 2026-10-24: 29/29 tests pass, 28/29 manifest
hashes match.

## R9 · 2026-10-23 09:00 — Dryad rights + CLEAR concept map

`artifacts/dryad_rights_clear_audit_20261023_0900/`

Established what CLEAR's vocabulary **is**: `mimic_concepts.csv` at revision
`142f169c8b006b0f28cf55c6a237c3003db5ed8f` — **368,294 free-text sentences from radiology
reports**, not a curated label set. Examples: *"et tube and left subclavian catheter are in
standard placements"*, *"no pathological parenchymal process"*.

**The mapping rule, fixed in advance:**

> Use only the **exact single vocabulary entry** whose text equals the concept term.
> Concepts with no exact entry are **EXCLUDED** and are **not** approximated by
> aggregating related sentences.

Rationale: aggregation needs a negation policy that CLEAR does not specify. It was already
measured there that **28.6% of consolidation-matching sentences are negated**. Three
concepts were excluded for having no exact entry: **cavity, nodule, reticular opacity**.

## R10 · 2026-10-23 → 10-24 — CLEAR concept audit (Kaggle)

`artifacts/kaggle_exec_prep_20261023_1130/` (package) → `artifacts/concept_audit_result_20261024_0130/` (result)

**Brief:** execution preparation only. *"Success means a correct, tested, user-runnable
Kaggle execution package. It does not mean improved AUC."* TUNE only; REPORT and
`internal_test` forbidden; no images or clinical rows in the downloadable package.

**Executed on Kaggle 2026-10-23T20:15:28Z**, Tesla T4, 9.3 min, all three asset checksums
matched, 1,352 / 139 / 466, fingerprint matched, 0 missing / 0 unapproved images, output
gate PASSED. Visual tower **87,174,144 parameters** = ViT-B/14 + 768×768 projection.

**Result — among 1,213 TB-negatives:**

| concept | ρ | 95% CI | p_BH | ΔR² |
|---|---|---|---|---|
| **scarring** | **+0.5661** | 0.5276–0.6020 | 9.03e-103 | **+0.1975** |
| fibrosis | +0.4166 | 0.3693–0.4627 | 1.80e-51 | +0.0996 |
| **consolidation** | **−0.3000** | −0.3511–−0.2477 | 3.55e-26 | +0.0182 |
| calcified granuloma | −0.1652 | −0.2179–−0.1105 | 1.29e-08 | **+0.0000** |
| volume loss | +0.1699 | 0.1150–0.2234 | 5.84e-09 | +0.0383 |
| bronchiectasis | +0.0815 | 0.0270–0.1373 | 6.28e-03 | +0.0097 |
| calcification | +0.0805 | 0.0253–0.1346 | 6.28e-03 | +0.0214 |
| focal opacity | −0.0795 | −0.1337–−0.0228 | 6.28e-03 | +0.0079 |
| pleural thickening | +0.0512 | −0.0032–0.1062 | 0.0747 (ns) | +0.0066 |

Baseline clinical-only R² = 0.2441 (scanner ×2, age group, sex, symptom count, `past_tb`).

**Two findings that mattered:**
- **scarring AUROC against Xpert status = 0.4941 [0.4454, 0.5422] — chance.** A large
  association with the score, zero information about TB. The signature of a nuisance axis.
- Stratified: scarring ρ = **0.570** in negatives *without* past TB (n=796) vs **0.418**
  *with* (n=417). **So it is not a past-TB subgroup effect.**
- **consolidation ran the wrong way** — ρ = −0.300 among negatives and **−0.385 among
  positives**, where consolidation is the canonical active-TB finding. Logged as **U3,
  unexplained.**

### Engineering problems solved along the way (Kaggle)

Worth recording because they cost most of the time: `HERE.parents[2]` resolving to
`/kaggle`; a silently random-initialised encoder from `load_state_dict(strict=False)` on an
empty filtered dict; a dead REPORT guard (allow-list filtered before the check); the 1000-file
upload cap → zip support; a zip-scratch collision keyed on stem; Kaggle **flattening** the
upload; two complete code copies attached with sort order picking the stale one; and the
decisive one — **`WeightLoadError: only 0.0%`**, which reading CLEAR's `hub.py` revealed
to be *three* errors: wrong key prefix (`visual.backbone.`), **missing projection**
(`encode_image` must be used), and **wrong preprocessing** (ImageNet/bilinear instead of
CXR stats/bicubic 448). Only the first would have raised. The other two would have produced
a full table of plausible, wrong numbers. **Fix: delete the reimplementation, call
`clear.load_pretrained`.**

## R11 · 2026-10-24 02:15 — REPORT exposure audit

`artifacts/report_exposure_audit_20261024_0215/`

**Brief:** read-only provenance audit; *"Do not conflate 'not accessed during this run'
with 'never previously used'."*

**Verdict: Vietnam REPORT is NOT pristine.** Scored in **six analyses across five runs**;
**four influenced a decision**.

| # | date | what | influenced a decision? |
|---|---|---|---|
| E1 | 10-20 | headline, 0.7242 | no |
| E2 | 10-21 | INLP arm → REJECT | **yes** |
| E3 | 10-21 | exploratory offset → +0.0206 | **yes** (became a design parameter) |
| E4 | 10-21 | scanner thresholds | **yes** |
| E5 | 10-21 | generalization arms | **yes** |
| E6 | 10-21 | previous-TB subgroup, on all 2,708 | **yes** |
| E7 | 10-21 | reproduction of 0.7242 | no |
| E9 | 10-23 | CLEAR audit — **REPORT not accessed** | no |

**The circularity:** the previous-TB pattern that motivates the concept hypothesis was
characterised on the pooled 2,708 **including REPORT** (E6).

**Also recommended NOT opening REPORT**, because the recurrent-TB safety endpoint has
~51 Xpert-positives with previous TB → 95% half-width ≈ **±0.13**, and a clinically
important 5-point sensitivity loss is **~3 patients**. The guardrail cannot detect the harm
it exists to prevent.

## R12 · 2026-10-24 12:54 — active-TB label alignment + RayDINO

`artifacts/active_tb_label_raydino_20261024_1254/` · protocol sha-256
`b88a637c…` hashed **before** fitting

**Brief:** settle whether `active_tb_label` is the documented target; then a controlled
label-alignment experiment; then assess RayDINO.

**Phase 1 — label semantics. Gate PASSES on primary-source evidence.** TBX11K's own COCO
`categories` block declares id 1 `ActiveTuberculosis`, id 2
`ObsoletePulmonaryTuberculosis` (the radiological term for healed disease). Box category
sets over the 800 TB images: `[1]` 630, `[1,2]` 30, `[2]` **139**, no box 1. **Category 3
never occurs**, so there is no ambiguous class.

**Phase 2 — three arms, one frozen fitting path, only the training target differs:**

| arm | training positives | negatives | n train |
|---|---|---|---|
| **A** | active + latent + mixed (`tb_label`) | healthy | 3,241 |
| **B1** | active + mixed (`active_tb_label`) | healthy **+ latent** | 3,241 |
| B2 | active + mixed | healthy (latent dropped) | 3,147 |

All scored against a **single fixed target** (`active_tb_label`) on identical rows, so the
comparison is of discrimination and not of denominators.

**Evaluation split: TBX11K calibration (899, 81 active-TB positives)** — development is
consumed by arm-specific early stopping. Calibration had never been regionally embedded;
the cache was extended inside the run directory and **verified bit-for-bit** against 24
cached development images (MAD 0.00e+00, cosine 1.000000) before use.

**Result — VERDICT: REJECT, and well powered:**

| criterion | required | observed | pass |
|---|---|---|---|
| primary paired ΔAUROC | ≥ +0.05 | **−0.0037** | no |
| CI excludes zero | yes | [−0.0081, +0.0002] | no |
| CI half-width (power check) | ≤ 0.05 | **0.0041** | yes |
| active-TB sensitivity drop | ≤ 0.02 | **+0.0123** (rose) | yes |

| arm | cal AUROC | AUPRC | sens | spec |
|---|---|---|---|---|
| A | **0.9498** | 0.6881 | 0.8519 | 0.9156 |
| B1 | 0.9460 | 0.6689 | 0.8642 | 0.9071 |
| B2 | 0.9493 | 0.6832 | 0.8519 | 0.9169 |

**The dissociation — the actual finding:**

| endpoint | B1 − A | 95% CI | B2 − A |
|---|---|---|---|
| **TBX11K calibration (primary)** | **−0.0037** | [−0.0081, +0.0002] | −0.0005 |
| **Vietnam TUNE (exploratory)** | **+0.0203** | [+0.0127, +0.0279] | +0.0023 |

**Same weights, same change, opposite sign.** TBX11K cannot see the benefit — separating
active from obsolete TB there is already at 0.95 with only 18 obsolete cases in the
evaluation split. Vietnam is where past TB is 36% of the positives.

**And B2 isolates the mechanism:** dropping latent images is worth +0.0023 on Vietnam;
*relabelling* them is worth +0.0203 — an order of magnitude apart. **The model has to be
*shown* that an obsolete-TB radiograph is not active disease.** This is what the earlier
arm X could not see, having bundled both changes.

**Phase 3 — CLEAR linkage: could not run.** Per-case concept scores were deliberately never
exported, and CLEAR has never been run on TBX11K. CLEAR was **not** re-run to manufacture
them.

**Phase 4 — RayDINO: still blocked** (licence, redistribution, unpublished resolution).
Target stays `tb_label`, which its config already pins.

**U1 resolved** — the 901-row sealed result is the provisional C01 `internal_test`
(R1), settled from the seal's own metadata without opening the split.

**Three defects found:** `labels_tbx11k_trainval_v1.csv` is **wrong on `tb1199`** (calls it
`active_tb`; published counts and the script's own documented `NO_BOX_FALLBACK` say
`latent_tb`); C01 stores `active_tb_label` as an **empty string** for all 140 latent rows;
and the **regional feature cache covered only 6,600 of 8,400 images** — undocumented
anywhere, and the reason the first attempt at the primary endpoint aborted.

**Leakage quantified:** 23 of 899 calibration images have a cross-split near-duplicate
(cosine ≥ 0.95). Removing them moves the primary endpoint by 0.0003. TBX11K ships **no
patient identifier**, so no fold here is patient-grouped and none is described as such.

## R13 · 2026-10-31 23:28 — B1 × CLEAR mechanism test

`artifacts/b1_clear_mechanism_20261031_2328/` · manifest sha-256 `eea6bfca…` hashed before
inference

**Brief:** does B1's TUNE gain come with a reduction in the score↔scarring association,
without harm among positives? Predeclared five-criterion decision rule.

**Deviation, declared:** ran **locally**, not on Kaggle — a *tightening* of the data rule
(nothing uploaded at all), and it made the reproduction check meaningful by running on
different hardware.

**Reproduction gate passed to five decimal places** against the Kaggle run: scarring ρ
0.5661 (diff 8e-06), ΔR² 0.1975 (diff 0.000000), clinical-only R² 0.2441 (diff 0.000000).

**Result — VERDICT: NO CLEAR MECHANISM EVIDENCE:**

| criterion | result |
|---|---|
| 1 scarring ρ lower under B1 | **met** — 0.5661 → 0.5317 |
| 2 paired CI excludes zero | **met** — [−0.0428, −0.0265] |
| 3 ΔR² directionally consistent | **met** — −0.0250 [−0.0315, −0.0184] |
| 4 control shows no comparable shift | **SPLIT** — `scoliosis` passes, `degenerative changes` fails |
| 5 no degradation among positives | **marginal** — sensitivity 0.9424 → 0.9281 |

- **scoliosis** (primary control): Δ −0.0070 [−0.0153, +0.0013] — interval covers zero ✓
- **degenerative changes** (secondary): Δ −0.0384 [−0.0475, −0.0297] — larger than scarring
- **8 of 11 concepts** shifted with intervals excluding zero → read at the time as
  *generalised re-ranking* (**later corrected in R14**)
- Safety: locked-threshold sensitivity 0.9424 → 0.9281 = **131 → 129 of 139**, two patients
- Scarring reduction concentrates among negatives *with* past TB (−0.0512) vs without
  (−0.0274)

**U2 answered:** the nine concepts are heavily inter-correlated — scarring × fibrosis
**+0.625**, focal opacity × consolidation **+0.797**, × volume loss +0.785. The BH family
was never nine independent tests.

## R14 · 2026-11-01 01:23 — consolidation forensic + B0/B1 audit

`artifacts/consolidation_b0_b1_forensic_20261101_0123/`

**Brief:** determine what the consolidation score actually measures; reproduce and stress
the correlations; decompose the B0→B1 change. Choose one of five dispositions.

### **Disposition: ARTEFACT_OR_PIPELINE_ERROR_IDENTIFIED**

> **The CLEAR anchors cannot distinguish a finding from its negation.**

**Text-side (no images, no patient data at all):**

| concept | negated phrase | **cosine** |
|---|---|---|
| volume loss | `no volume loss` | **+0.980** |
| calcification | `no calcification` | +0.961 |
| focal opacity | `no focal opacity` | +0.951 |
| pleural thickening | `no pleural thickening` | +0.915 |
| **scarring** | **`no scarring`** | **+0.906** |
| fibrosis | `no fibrosis` | +0.844 |
| **consolidation** | **`no consolidation`** | **+0.747** |

Among 1,500 real vocabulary sentences mentioning consolidation, **negated sentences are
*closer* to the anchor than affirmative ones** (median +0.6161 vs +0.6047).

**Why consolidation specifically — the asymmetry:**

| concept | entries containing the stem | negated | **% negated** |
|---|---|---|---|
| **consolidation** | 14,582 | 4,029 | **27.6%** |
| fibrosis | 1,201 | 76 | 6.3% |
| **scarring** | 4,752 | 76 | **1.6%** |
| scoliosis | 1,275 | 14 | 1.1% |

*"No focal consolidation"* is stock normal-report language. Its anchor therefore sits in
the **normality** region of the embedding.

**Image-side confirmation** — six probes declared in writing *before* any image-side
similarity was computed. Among the 1,213 TB-negatives, against the production score:

| anchor | ρ vs classifier score | ΔR² |
|---|---|---|
| **`no acute cardiopulmonary process`** | **−0.6012** | 0.1375 |
| **`no consolidation`** | **−0.5339** | 0.0796 |
| `clear lungs` | −0.4694 | 0.0602 |
| `normal chest` | −0.4348 | 0.0955 |
| `lungs are clear` | −0.4107 | 0.0334 |
| **`consolidation`** | **−0.3000** | 0.0182 |
| **`no scarring`** | **+0.3009** | 0.0645 |
| **`scarring`** | **+0.5661** | 0.1975 |

**`no consolidation` is more negative than `consolidation`. `no scarring` has the same
sign as `scarring`.** The consolidation anchor's −0.300 is a *weaker* version of the
normality signal, which reaches −0.601.

**Reproduction — exact:** consolidation −0.3000 / −0.3853, scarring +0.5661, fibrosis
+0.4166; BH values reproduce (scarring 9.27e-103).

**Eliminated:** sign flip (code read — plain cosine, both sides normalised, no temperature,
no contrast), index/column-order error, misalignment (join by pid, shuffle-tested), outliers
(leave-one-out range **0.003**), Simpson's paradox (**all twelve strata negative**).
**Partly responsible:** between-scanner composition (pooled −0.300 vs within-scanner
−0.140 / −0.241 / −0.147) and concept collinearity.

### Second result: R13's "broad re-ranking" was wrong

> **Spearman(text cosine to the scarring anchor, Δρ) = −0.824, p = 4.9e-05, n = 17.**

All **9** concepts at cosine ≥ 0.69 moved negatively (mean −0.0244); of the **8** below,
only 3 did (mean +0.0026). **The four normality anchors — which carry the largest
associations in the whole study — did not move at all** (|Δ| ≤ 0.0042, every interval
covering zero).

R13's reading was a sampling artefact: all nine primary concepts are TB-adjacent and seven
sit inside the scarring neighbourhood, so that set could not distinguish "this
neighbourhood moved" from "everything moved".

**`degenerative changes` was never a valid control** — cosine **0.730** to scarring, above
two *primary* concepts. `scoliosis` (0.524, image-side −0.005) was valid and did not move.

**But the limit still binds:** `scarring` moved −0.0344 and **`no scarring` moved −0.0343**.
B1 changed the model's relation to a *topic neighbourhood*, not to scarring the finding.

B0↔B1 rank agreement **0.9893** (mean rank shift 3.0% of the cohort) yet AUROC moves
+0.0204 — all concept changes are **ranking** changes, not calibration.

---

## R15 · 2026-11-01 03:00 — efficiency and site-adaptation benchmark

`artifacts/site_adaptation_benchmark_20261101_0300/` · protocol `a999e45a…` hashed first

**Brief (from the product-strategy exchange):** measure the real cost of screening and of
adapting to a new site, on GPU and CPU; and test whether **refitting the 769-parameter
head** on local labelled images beats **refitting only the threshold**, patient-disjoint,
at a fixed operating point.

**Cost — measured:**

| | GPU (RTX 4050, fp16) | CPU (8 threads, fp32) |
|---|---|---|
| encode, per image | **0.030 s** (33/s) | 1.25 s (0.8/s) |
| peak GPU memory | **0.69 GB** | — |
| head fit, 3,241 images | **6.6 s** | 61.6 s |
| full site adaptation, 331 local images | **16.2 s** | ~8 min |
| trainable / frozen parameters | 769 / 86,580,480 | same |

**Adaptation — Shenzhen (331/331 split) and Montgomery (69/69), 3 draws per n:**

- Frozen head, no adaptation: AUROC **0.9681** / **0.9483** on the held-out halves.
- Threshold-only refit reaches the oracle sensitivity at 0.70 specificity with **25 local
  images** (Shenzhen 0.970; Montgomery 0.931).
- **Head refit vs threshold-only: 0 of 30 comparisons met the "helps" rule**; 26 no
  benefit, 4 harm (spec −0.025 to −0.031). Max ΔAUROC **+0.0031**. Sensitivity identical
  in 28 of 30 (paired CI [0, 0]).

**Verdict: the "site-adaptive by head refit" clause of the product thesis is NOT
supported and is removed.** No headroom (frozen head already 0.95–0.97), conservative
design by protocol (unweighted, early-stopped on TBX11K dev), and the only sites where
adaptation would matter (Vietnam, 0.72) are exposed and unusable. Threshold recalibration
works and is parity with commercial practice, not an advantage.

**Corrections made during this exchange:** the earlier "Shenzhen 0.7589 → 0.9345"
site-adaptation claim was a **misreading** — those are sensitivities at 95% and 70%
specificity, both after local threshold fit, not before/after. "769 parameters" must be
stated as trainable alongside the 86.6 M frozen. Vendor comparisons (retraining contracts,
post-hoc heatmaps, inability to publish) were assertions, not findings, and are withdrawn.

# PART III — SYNTHESIS

## 5. What is actually known

1. **The model works on benchmarks and not on screening.** 0.9368 internal vs **0.7242**
   on a programmatic active-case-finding screening cohort (risk-based mobilisation, not
   consecutive enrolment), with a commercial comparator at 0.7862 on identical patients.
2. **It degrades badly in previous TB** — 0.7811 → 0.6182 (construction A: 0.7720 →
   0.6392, degradation +0.1328 [0.0718, 0.1951]). **This is published**, by the paper that
   released the dataset. **The claim that we degrade *more than the comparator* is NOT
   supported under the published construction** — difference-in-differences
   +0.0481 [−0.0170, +0.1113], interval includes zero (R7). Do not write the 1.7x framing
   into the paper.
3. **Removing the scanner-predictive embedding subspace did not improve screening AUROC**
   (−0.0002, R3). Scanner-associated *score offsets* were later found to matter for pooled
   ranking and operating points (items 14–15); the two findings are different and both stand.
4. **Clinical-context fusion is the first intervention to meet its preregistered bar on the
   screening cohort** — +0.058 overall, +0.100 in previous TB, specific to our score (the
   comparator gained nothing from the same fields in this experiment), bringing the point
   estimate level with the comparator's raw score on this exposed cohort with an interval that
   includes zero (R16). Item 14 later decomposed the gain: the scanner columns carry the part
   that matters most in previous TB.
5. **Aligning the training target to active TB helps externally and not internally**:
   +0.0203 on Vietnam TUNE, −0.0037 on TBX11K calibration — and the ingredient is
   **relabelling latent TB as negative** (+0.0203) rather than dropping it (+0.0023).
6. **The CLEAR concept axes are topic anchors, not finding detectors**, and this is now
   demonstrated rather than suspected. They cannot separate a finding from its negation.
7. **The consolidation anomaly (U3) is explained** and the explanation disqualifies the
   instrument rather than rescuing it.
8. **Vietnam REPORT is not pristine** — six scoring events, four influencing decisions.
9. **TBX11K `internal_test` has been scored once** under a provisional definition.
10. **The external gap is not a source-diversity gap.** Training the head on three
    sources (TBX11K + Shenzhen + Montgomery, 4,041 images) instead of one moves calibration
    by +0.0033 [+0.0006, +0.0064] and TUNE by +0.0036 (R19). Under the active target the
    three-source head reaches **0.7517** on TUNE — the best single head measured — at zero
    cost on calibration (MB1 − A = 0.0000); +0.020 of that is the known relabel, +0.008 the
    sources. Not adopted.
11. **Ensembling encoders adds nothing; choosing the encoder does.** Averaging RAD-DINO,
    RAD-DINO-MAIRA-2 and XraySigLIP heads is +0.0006 on calibration (R20). Replacing
    RAD-DINO by RAD-DINO-MAIRA-2 alone — same architecture, same cost — is −0.0004 on
    calibration and +0.0188 [+0.0015, +0.0364] on TUNE. A 2,304-d concatenated head has
    the best development AUROC ever fitted (0.9505) and loses externally (−0.0137):
    "more internal, less external" holds for capacity as well as for configurations.
    The three external levers now known — encoder, label, sources — are independent
    choices, each invisible internally and positive externally.
12. **The levers are not additive and head-level choices have a ceiling near 0.76 on
    TUNE** (R21). MAIRA-2 + active target reaches 0.7585 [0.722, 0.796], +0.0348
    [+0.0160, +0.0542] over production, −0.0032 on calibration; adding the three sources
    on top subtracts (−0.0062). The comparator's raw 0.7862 is inside that CI and not
    reachable from any head-level choice measured. Every lever was selected on TUNE, so
    0.7585 is a best case designed on the cohort, not an estimate.
13. **With fusion, the MAIRA-2 + active head reaches 0.7941 on TUNE — numerically above
    the comparator's raw 0.7780 (+0.0170 [−0.0193, +0.0527]) and its fused 0.7778**
    (R22). Fusion is worth +0.037 on this head (+0.058 on A): the gains overlap. The
    defensible sentence is "a numerically higher point estimate with a paired interval that
    includes zero; no superiority, equivalence or non-inferiority claim", on an exposed
    cohort that chose the encoder, the target and the fusion. The comparator
    keeps the better AUPRC (0.328 vs 0.300).
14. **The fusion gain is half scanner-associated adjustment, half diffuse questionnaire**
    (R23). Dropping the two scanner columns costs −0.0138 [−0.0250, −0.0029] and −0.043 in
    previous TB; no other field's loss excludes zero; dropping `past_tb` changes the
    previous-TB AUROC by +0.004. Without the scanner term the candidate is 0.7803 against
    the comparator's raw 0.7780, within noise. Scanner indicators improved our candidate
    substantially (≈ 0.02 overall, 0.04 in previous TB) and gave the comparator no material
    improvement in this experiment (+0.001 [−0.018, +0.019]); this does **not** establish
    scanner invariance in either system — that would need scanner-stratified performance and
    calibration analysis including an unseen scanner.
15. **The scanner offset can be removed without labels, under the nominal sampling condition
    tested** (R25; patient-disjoint evaluation group 677, 70 Xpert-positive). Centering the
    candidate score on a scanner's own unlabelled batch gives 0.7795 [0.769, 0.786] with
    **25** images per scanner, and at **100** images a paired **+0.0281 [+0.0074, +0.0486]**
    over the raw 0.7546 (previous TB +0.0557 [+0.0065, +0.1004]) — the same as the
    supervised scanner term. Referral burden at a label-free top-30 % rule moves from
    **31 % / 24 % / 50 %** across the three scanners to **40 % / 31 % / 32 %**. It removes
    population differences too: a calibration batch of previous-TB patients only lowers that
    scanner's sensitivity from 0.677 to 0.555 (move 0.122; pre-registered bound 0.10), so the
    verdict is **NOT USABLE AS IS**. The supervised term fails worse under prevalence shift
    (sensitivity 0.18 on a negatives-only batch; 46 % burden on an enriched one). The method
    needs a consecutive-screening calibration sample and a fail-closed guard; neither is
    tested, and the result is not a general representative-sampling guarantee.

16. **None of the raw scores is a probability; the fused output is one only where it was fitted**
    (audit A1, descriptive, no new fitting; `artifacts/probability_calibration_audit_20261101_1100`).
    The served v2 image logit read through a sigmoid has ECE **0.487 [0.464, 0.510]**, calibration
    intercept **−5.3**, slope **0.30** on TUNE (top decile: 0.996 predicted, 0.23 observed); the
    comparator's raw abnormality score behaves the same way (ECE 0.439, intercept −3.4, slope 0.68).
    The fused outputs, out-of-fold, are probability-shaped — v2 ECE 0.024 [0.018, 0.044], v3 ECE
    0.016 [0.013, 0.035], intercept 0.00, slope 0.94, Brier skill +0.09 / +0.11 over prevalence — but
    the intercept is TUNE's prevalence and the scanner columns are TUNE's scanners, so this holds
    for a site with the same prevalence and the same three scanners and for no other. Scanner 1
    (n = 204) sits above the 0.03 ECE bound in both. No recalibration map was fitted or saved; the
    API contract "screening score, not a probability" is now a measured statement.

## 6. Methodological rules that emerged (and held)

- **Preregister and hash before fitting.** Every experiment since R5 carries a sha-256 of
  its protocol, and the run script refuses to start if it changed.
- **Never lower a declared threshold to let a known result pass.** +0.05 was retained in
  R12 precisely because arm L's +0.0292 was already known.
- **Secondary arms are never promoted**, no matter how good they look (arm L, R5).
- **A fixed evaluation target across arms**, or the comparison is a denominator change.
- **Fail closed.** The encoder checks that `encode_image` and `visual.projection` exist —
  that check is the only reason the three-error CLEAR bug was caught before scoring.
- **Reproduction gates.** R13 and R14 reproduce the prior Kaggle run to 5 decimals on
  different hardware before being allowed to publish anything new.
- **Negative controls must be chosen from geometry, not intuition** — `degenerative
  changes` looked unrelated and was inside the region under test.
- **Power statements before results.** R12 declared "if the CI half-width exceeds the
  effect size, the verdict is INCONCLUSIVE regardless of sign."
- **No per-case export.** Concept similarities are computed in private scratch and deleted;
  tests assert no pid appears in any run directory.

## 7. Defects found in the project's own artifacts

| # | defect | impact |
|---|---|---|
| 1 | `labels_tbx11k_trainval_v1.csv` wrong on `tb1199` | 1 image in train; cannot move a figure |
| 2 | C01 stores `active_tb_label` as `""` for 140 latent rows | silent NaNs for any numeric reader |
| 3 | regional feature cache covered only 6,600 of 8,400 images | undocumented; aborted a run |
| 4 | 23 of 899 calibration images have cross-split near-duplicates | P1-07 review still outstanding |
| 5 | `degenerative changes` used as a negative control | invalid; it is inside the scarring neighbourhood |
| 6 | exploratory REPORT result (+0.0206) used as a design parameter | propagated into power planning |
| 7 | exact-entry concept rule does not achieve its purpose | negation contamination is in the embedding, not the aggregation |

**No arithmetic error has ever been found.** Every statistic re-checked has reproduced
exactly. What has repeatedly needed repair is **interpretation**.

## 8. Open items

| item | status |
|---|---|
| **U3** consolidation anomaly | **CLOSED** (R14) |
| **U2** concept collinearity | **CLOSED** (R13, re-measured R14) |
| **U1** 901-row sealed result | **CLOSED** (R12) — it was provisional `internal_test` |
| **U4** CLEAR's exposure to this cohort | **OPEN**, not resolvable from any artifact |
| definitive C01 split from P1 | **OUTSTANDING** — every TBX11K number is conditional |
| P1-07 near-duplicate review (158 pairs) | **OUTSTANDING** |
| cavity never measured | **OPEN** — no exact vocabulary entry |
| B1 positive-case safety | **OPEN and unresolvable here** — 2 of 139 |
| `rehearsal1` case list | **needs P1/P4 confirmation** before any one-shot read |
| RayDINO | **BLOCKED** on licence; package ready |
| head-level site adaptation as a differentiator | **NOT SUPPORTED** (R15) — removed from the thesis |
| clinical-context fusion | **SUPPORTED, exploratory** (R16) — +0.058 overall, +0.100 in previous TB; reaches comparator parity; build into serving as optional input |
| localisation vs TBX11K lesion boxes | **MEASURED** (R17) — top-1 hit 0.959 [0.898, 1.0] vs 0.367 lung baseline, n=49, exploratory |
| box-level supervision as a methodology | **CLOSED** (R17) — null internally, negative externally |
| multi-source head training (TBX11K + Shenzhen + Montgomery) | **CLOSED** (R19) — +0.0033 [+0.0006, +0.0064] calibration, +0.0036 TUNE; real, one sixth of the bar |
| multi-encoder ensemble (RAD-DINO + MAIRA-2 + XraySigLIP) | **CLOSED** (R20) — E3 − D +0.0006 [−0.0067, +0.0076] calibration; concat head overfits the source |
| RAD-DINO-MAIRA-2 as a drop-in encoder | **HYPOTHESIS** (R20) — −0.0004 calibration, **+0.0188 [+0.0015, +0.0364]** TUNE, same cost; to be stacked with the active target and three sources in a pre-registered run |
| three levers stacked (MAIRA-2 + active target + three sources) | **REJECT on calibration** (R21) — X − A −0.0010 [−0.0099, +0.0083]; TUNE +0.0286, not additive (predicted +0.047); best head on TUNE is MAIRA-2 + active target, TBX11K only: **0.7585 [0.722, 0.796]** |
| clinical fusion on the MAIRA-2 + active head | **REJECT on its bar, product finding** (R22) — fused X 0.7941 vs fused A 0.7809 (+0.0133 [−0.0019, +0.0296]); vs comparator raw 0.7780: **+0.0170 [−0.0193, +0.0527]** — first score numerically above the comparator; not significant; exposed cohort |
| which clinical fields carry the fusion gain | **DECOMPOSED** (R23) — only the scanner identifier carries it (−0.0138 [−0.0250, −0.0029]; −0.043 in previous TB); questionnaire-only fusion 0.7803 against comparator raw 0.7780 (within noise); image + scanner alone 0.7788 |
| research candidate v3 frozen | **BUILT, NOT PROMOTED** (R24) — `artifacts/p2/bundle_v3_candidate` (MAIRA-2 `a6132efb…`, active target, R22 head copied not retrained, dev-only thresholds, 12-field fusion with scanner flagged as site calibration); runtime reproduces cache to max |diff| 0.0046 on 16 images; `serve_v2.py --bundle` selects it; v2 stays default |
| label-free per-scanner calibration | **WORKS, NOT USABLE AS IS** (R25) — mean-centering on 25 unlabelled images/scanner recovers the full supervised gain (+0.028 [+0.007, +0.049]; +0.056 previous TB); fails the pre-registered stability bound on one stress condition (previous-TB-only batch: scanner sensitivity −0.122, bound 0.10); supervised term fails worse under prevalence shift; requires a consecutive-sample calibration batch — guard not yet tested |
| untouched-cohort head-to-head vs comparator | **PRE-REGISTERED, AWAITING DATA** — `artifacts/headtohead_prereg_v1/` (PROTOCOL v1 sha `16b44ec1…` hashed 2026-11-01; non-inferiority margin 0.03, previous TB pre-specified but exploratory, scanner-stratified; ≥ 400 confirmed cases for power; one-shot scorer with LOCK, no training path; DATA_REQUEST.md ready to send) |
| consecutive-sample calibration guard | **IMPLEMENTED AS A FAIL-CLOSED PROPOSAL ENDPOINT, STILL UNVALIDATED** (`src/p2_cls/site_calibration.py`, `/v2/calibration/propose`; 6 tests) — ≥ 25 unlabelled consecutive screening images, questionnaire composition checked against a site reference, **fail closed**: composition uncertain or outside validated limits → no correction, frozen uncorrected model, human review flag. Software cannot prove a batch is consecutive; that needs auditable enrolment records. Establishing that these safeguards make calibration safe requires a prospective evaluation, not another TUNE run |
| is any output a probability | **MEASURED** (audit A1, descriptive) — raw v2 logit through a sigmoid: ECE 0.487, intercept −5.3, slope 0.30 (not a probability; comparator raw the same); fused out-of-fold: ECE 0.024 (v2) / 0.016 (v3), intercept 0.00, slope 0.94 — a probability only at TUNE's prevalence and scanners; nothing fitted or saved for serving |
| served model = research model | **FIXED** (R18) — bundle_v2_regional + pipeline_v2 + serve_v2, smoke-verified |
| scanner composition in the consolidation association | quantified, not decomposed |

## 9. What to do next

**Stop the concept-intervention line.** The instrument cannot support an intervention
aimed at scarring, because it moves identically for `scarring` (−0.0344) and `no scarring`
(−0.0343). The obvious repair — a contrastive `sim(X) − sim(no X)` score — is unavailable:
exact negated entries exist for only 5 of the 9 primary concepts, and inventing the rest
would be post-hoc concept construction.

**The single next action is documentation, not an experiment:** in `report/paper_draft.md`
and the concept sections of the prior reports, replace every reading of a concept *name*
as a radiological finding with the bounded wording, and record negation-blindness as a
method limitation. The numbers stand; only the words need repair.

**The next milestone is not another model experiment.** Candidate v3 is frozen and
reproducible, and the evaluation that can test it is pre-registered and hashed
(`artifacts/headtohead_prereg_v1/`, protocol v1 `16b44ec1…`): send `DATA_REQUEST.md`, build the
manifest on receipt, dry-run, then the single `--run`. The question that can change the product is whether it beats the commercial
comparator on a cohort nobody here has touched — same patients, same reference standard,
previous TB as a pre-specified primary subgroup, protocol fixed before the first score.
Both Vietnam partitions are exposed (TUNE by design; REPORT by the six events at R7), so
that cohort has to come from the data collaboration.

**Product thesis, as it now stands (wording after the strategy exchange of 2026-11-01):**
*a lightweight regional chest-X-ray TB screening system built on a frozen medical-imaging
foundation model, combining clinical-context information with scanner-aware calibration
and exact region-level attribution, targeting screening deployment on modest hardware,
with particular interest in people with previous TB.* It is a **scanner-aware screening
research system with an experimentally investigated label-free calibration mechanism** —
not a self-calibrating system. Claims it does not make: superiority to, equivalence with
or non-inferiority to commercial CAD (the paired interval includes zero), scanner
invariance in either system, safe automatic calibration at a new site, better heatmaps,
or that commercial CAD cannot use clinical context. The Vietnam cohort is a programmatic
risk-based active-case-finding cohort, not consecutive enrolment; bootstrap intervals are
participant-level (no screening-event identifier is released). Parameter count to state: 769 per seed × 5 = 3,845 (head) + 13
(fusion) trainable; 86.6 M frozen. Throughput to state separately: encoder 0.03 s/image
GPU; full pipeline median 0.13 s/image (p90 0.15 s) with the segmenter on the GPU (R26-eng,
2026-11-01: masks agree with the CPU training cache at 99.998 % of pixels; `scripts/p2_41_segmenter_latency.py`,
`artifacts/p2/segmenter_latency.json`); ≈ 2.1 s/image on CPU. The proposal's < 2 s objective is met.

**Not recommended:** opening Vietnam REPORT, opening TBX11K `internal_test`, adopting B1,
score subtraction, apical weighting, concept-based recalibration, or starting RayDINO
before the licence question has an institutional answer.

## 10. The honest summary for the write-up

> A frozen RAD-DINO regional-pooling classifier reaches 0.9368 AUROC on TBX11K development
> and **0.7242** on a programmatic Vietnamese active-case-finding cohort, against a commercial
> comparator's 0.7862. It degrades sharply in participants with previous TB
> (0.7720 → 0.6392, +0.1328 [0.0718, 0.1951]), reproducing an effect already published on
> this dataset. Whether it degrades *more* than the commercial comparator is **not
> statistically supported** under the published construction (+0.0481 [−0.0170, +0.1113]).
>
> Scanner shift was tested and rejected. Aligning the training target to the dataset's own
> *active* TB definition is worth +0.0203 on the screening cohort and **nothing** on
> internal data, and the effective ingredient is labelling healed TB as negative rather
> than merely excluding it; it was **not adopted**, having failed its preregistered
> criterion.
>
> A concept-based attempt to explain the failure found a large association between the
> score and a "scarring" text axis (ρ = 0.566) that carries no information about TB
> (AUROC 0.494). **Forensic analysis then showed the axis cannot distinguish a finding
> from its negation** — `no scarring` correlates at +0.301, the same sign — so the
> association cannot be read as reliance on scarring. The anomalous negative
> "consolidation" association is explained the same way: its anchor sits in the
> normal-report region of the embedding.
>
> After the concept work, four further methodologies were tried on the screening gap
> (R19–R22). Multi-source training, encoder ensembling and box supervision are closed
> as nulls on the primary split. Swapping the encoder for RAD-DINO-MAIRA-2 and training
> to the active-TB target gives 0.7585 raw on the screening cohort (+0.035 over
> production, nothing on internal data), and with the 12-field clinical form 0.7941 —
> numerically above the comparator's 0.7780, interval [−0.019, +0.053]. Of that gain,
> the part that puts the point estimate above the comparator is a two-column scanner
> correction (R23); with the questionnaire alone the candidate is at parity (0.7803).
> Every one of those choices was made on the same exposed cohort; none is adopted, and
> none is validation.
>
> **The strongest defensible statement is: similarity to a short radiological text phrase
> is associated with the model score.** Nothing here establishes causation, clinical
> benefit, novelty, or readiness, and no result on an exposed cohort is external
> validation.

---

## Appendix A — pinned artifacts

| asset | revision / hash |
|---|---|
| RAD-DINO | `microsoft/rad-dino` @ `110cbc18d5133582e320b43d53bf5c44e410c936`, MIT |
| CLEAR | `peterhan91/CLEAR` @ `142f169c8b006b0f28cf55c6a237c3003db5ed8f`, Apache-2.0 |
| CLEAR package | commit `57d2eae57ddabf8e56c8013ec675336d71b3d8e7` |
| `best_model.pt` | `da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03`, 603,060,057 B |
| `concept_embeddings_368294.pt` | `7241c5b920c4394b507f2c4bd89f1d0b70775399caaa2f5323cb373afb1a00d6`, 1,131,400,935 B |
| `mimic_concepts.csv` | `93e975b722422b64f18b57161ac9894fe30ac7a04cc68b7aa8a2e01f6d0fc439`, 22,208,546 B |
| TUNE cohort fingerprint | `1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751` |
| RayDINO | `xray_dino_vitl16`, **FAIR Noncommercial, gated** — blocked |
| RAD-DINO-MAIRA-2 | `microsoft/rad-dino-maira-2` @ `a6132efb9268859545e9f68be18a76a6473f174a`, ungated (R20) |
| XraySigLIP | `StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli` @ `e2ad64eee3c2a2ec746f4c0934b018c718917714`, ungated (R20) |

## Appendix B — the nine concepts and their indices

`fibrosis` 835 · `scarring` 96952 · `volume loss` 368164 · `calcified granuloma` 209383 ·
`calcification` 77874 · `focal opacity` 281312 · `consolidation` 123742 ·
`pleural thickening` 341471 · `bronchiectasis` 57870

Excluded, no exact entry: **cavity, nodule, reticular opacity**.
Controls: `scoliosis` 82734 (valid), `degenerative changes` 339316 (**invalid**).
Probes (R14): `no consolidation` 55926 · `no scarring` 142706 · `clear lungs` 64918 ·
`lungs are clear` 66018 · `no acute cardiopulmonary process` 322541 · `normal chest` 290873.

## Appendix C — environment

Windows 11, Python 3.13.2, torch 2.6.0+cu124, CUDA 12.4, NVIDIA RTX 4050 Laptop,
numpy 2.2.6, scipy 1.17.1, scikit-learn 1.9.0. `HF_HOME=D:\hf-cache` (shared, never
delete). Kaggle Tesla T4 for R10 only.

## Appendix D — where to look

| you want | read |
|---|---|
| the headline number | `artifacts/p2/regional/mil_final.json` |
| the honest decision | `artifacts/realworld_generalization_20261021_1015/FINAL_RESEARCH_DECISION.md` |
| what REPORT has seen | `artifacts/report_exposure_audit_20261024_0215/` |
| the label experiment | `artifacts/active_tb_label_raydino_20261024_1254/FINAL_REPORT.md` |
| the concept mechanism test | `artifacts/b1_clear_mechanism_20261031_2328/RUN_REPORT.md` |
| **why the concepts are not findings** | `artifacts/consolidation_b0_b1_forensic_20261101_0123/FINAL_REPORT.md` |
| prior art and claim limits | `artifacts/prior_art_review_20261021_1200/CLAIM_BOUNDARIES_REVISED.md` |
