# P2 handover to the team — what changed, what is frozen, what each of you needs to know

**From:** Atul (P2 — classifier, evaluation, serving) · **Date:** 2026-11-01 · **Branch:** `atul` on `gundatharuntej2006-hue/cv_proj` (tip `83b505f` at the time of writing; the triage screen described in §5.4 is local and not yet pushed)

**Written for:** Tharun (P1), Ritika (P3), Kunal (P4), and the guide. Everything here is on the provisional TBX11K split and an exposed Vietnam cohort; §3 explains both. Where a number appears, its source run is named so you can open the table it came from.

---

## 1. One page: where P2 stands

| question | answer |
|---|---|
| Is any training needed? | **No.** Every model is frozen. Nothing in the repository trains on Vietnam, and the TBX11K heads are fitted in seconds from cached features if the split ever changes (§2). |
| What is served? | `scripts/serve_v2.py`: the v1 API plus `/v2/*` routes — regional head, exact region attribution, batch ranking, overlays, optional clinical fusion, a triage screen at `/v2/ui`. Default model = bundle v2 (RAD-DINO). Candidate v3 (MAIRA-2) selectable, not promoted. |
| Headline numbers | TBX11K development 0.9368; Vietnam REPORT **0.7242** vs Lunit 0.7862; with questionnaire fusion on the tuning half the v3 candidate reaches 0.7941 vs Lunit raw 0.7780, interval includes zero — **no superiority, equivalence or non-inferiority claim**. |
| Latency | Median **0.13 s/image** end-to-end on the laptop GPU after a ~40 s first-call model load (segmenter now on GPU); ≈2.1 s/image on CPU. The proposal's < 2 s objective is met. |
| Localisation | Decision cell overlaps an annotated active lesion in **95.9 %** of 49 development images (exploratory; 12×12 region cells, not lesion boundaries). |
| What is left | An untouched cohort for the pre-registered head-to-head (`artifacts/headtohead_prereg_v1/`); the college report in the department format; co-author sign-off on the paper. |

---

## 2. Training — status, and what would ever require it

**Nothing needs training.** The project's model is a frozen encoder plus a linear head; the heads already in the bundles were fitted, gated and tested, and the research phase is closed after 25 pre-registered runs (record R1–R25).

| component | trainable? | state | where |
|---|---|---|---|
| RAD-DINO encoder (`microsoft/rad-dino` @ `110cbc18…`) | no, frozen (86,580,480 params) | downloaded to `D:\hf-cache` | bundle v2 |
| RAD-DINO-MAIRA-2 (`microsoft/rad-dino-maira-2` @ `a6132efb…`) | no, frozen | same cache | bundle v3 |
| regional head, 5 seeds (769 params each) | yes — **already fitted** | `artifacts/p2/bundle_v2_regional/head.npz` (tb_label), `artifacts/p2/bundle_v3_candidate/head.npz` (active_tb_label) | fit time 6.6 s on cached features |
| clinical fusion (13 params) | yes — already fitted on Vietnam TUNE | `fusion.json` in each bundle | exploratory; TUNE is exposed |
| lung segmenter (torchxrayvision PSPNet) | no, third-party weights | auto-downloaded | — |

**When retraining would be required, and how long it takes**

1. **Tharun delivers the definitive C01 split** (the one outstanding item that changes every TBX11K number). Then: rebuild the regional feature cache for any newly included images (`scripts/p2_31_regional_mil.py --stage embed`; ~0.03 s/image on GPU — the full 8,400 is ≈ 5 min), refit the five seeds (`scripts/p2_37_build_bundle_v2.py`, `scripts/p2_39_build_bundle_v3.py`; seconds), re-run the smoke tests (`p2_38`, `p2_40`), and re-run the calibration-split runs whose tables feed the paper (each 2–8 min). Nothing on Vietnam needs re-running for a TBX11K split change; the fusion coefficients would be refitted only if the image scores change.
2. **A label correction in TBX11K** (see §3.1, `tb1199`): same procedure, one image.
3. **A new encoder or target** — do not. The record closes encoder ensembling, box supervision, multi-source training, head adaptation and attention pooling with numbers; the three levers that help externally overlap. Any further modelling on the Vietnam cohort produces numbers nobody can trust, including us.

**Never:** train, tune, select thresholds or refit on Vietnam REPORT or TBX11K `internal_test`; subtract concept scores; infer fibrosis from `past_tb` (it is self-reported history).

---

## 3. Datasets — what changed and what you must know

### 3.1 TBX11K (training and internal evaluation)

Provisional split C01 (`artifacts/p2/dryrun/provisional_c01.csv`), one image per individual:

| split | n | healthy | sick_non_tb | active | latent | active+latent | positives (`tb_label`) | positives (`active_tb_label`) |
|---|---|---|---|---|---|---|---|---|
| train | 5,941 | 2,700 | 2,700 | 426 | 94 | 21 | 541 | 447 |
| development | 659 | 300 | 300 | 47 | 10 | 2 | 59 | 49 |
| calibration | 899 | 400 | 400 | 78 | 18 | 3 | 99 | 81 |
| internal_test | 901 | 400 | 400 | 79 | 18 | 4 | 101 | 83 |

**Changes and defects you need to act on or know about:**

- **The production head trains on healthy + TB only (3,241 rows).** The 2,700 `sick_non_tb` train images are excluded (decision in R4/R5: healthy-only negatives). This is why TB-vs-sick is the harder, more honest internal metric, and why the screening cohort is where the model fails (§6 of the record).
- **Two targets exist.** `tb_label` (active + latent + mixed positive) is production; `active_tb_label` (latent/obsolete relabelled negative) is the v3 candidate's target. The COCO categories are `ActiveTuberculosis` (1) and `ObsoletePulmonaryTuberculosis` (2); category 3 never occurs in trainval.
- **Label defect, Tharun:** `labels_tbx11k_trainval_v1.csv` calls `tb1199` `active_tb`; the published counts and the script's own `NO_BOX_FALLBACK` rule say `latent_tb`. Not yet corrected in the label table.
- **Empty field, Tharun:** C01 stores `active_tb_label` as an empty string for all 140 latent rows instead of 0.
- **Near-duplicates:** 158 cross-split near-duplicate pairs at cosine ≥ 0.95 across the *official* TBX11K splits (52 train→val); within our C01, 23 of 899 calibration images have a cross-split near-duplicate (effect on the primary endpoint 0.0003). Candidate list: `artifacts/p2/dedup/provisional_near_dup_official_v0.csv`. P1-07 review still outstanding.
- **No patient identifier ships with TBX11K**, so no fold is patient-grouped; say so wherever a fold is described.
- **`internal_test` (901) has been scored once**, by the 2026-10-14 global model in the sealed rehearsal (R1). It has not been read since and must not be until the definitive split exists.
- **Feature caches** (ignored by git, regenerate with the scripts named): `artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz` covers train + development only (6,600 images, not 8,400); the calibration features live in `artifacts/active_tb_label_raydino_20261024_1254/outputs/regional_tbx11k_calibration_lungs_g12_v1.npz`. MAIRA-2 and XraySigLIP features for train/dev/cal/TUNE are under `artifacts/multiencoder_ensemble_20261101_0600/cache/` (2.4 GB) and MAIRA-2 NLM under `artifacts/stacked_levers_20261101_0700/cache/`.
- **Pre-masked images:** `data/processed/v1/tbx11k_lungs512/` (train + dev, 6,600) and `vietnam_img512_lungs/`, `nlm_img512_lungs/`. There is **no** pre-masked calibration folder; R20 built one at `artifacts/multiencoder_ensemble_20261101_0600/cache/lungs512_calibration/` with the same rule (outside-lung pixels → image mean).

### 3.2 Vietnam screening cohort (the real test, exposed)

Dryad `10.5061/dryad.5tb2rbpcv` (Codlin et al. 2025): 2,708 participants, 280 Xpert-positive, three radiography systems, twelve questionnaire/scanner fields, Lunit INSIGHT CXR 3.9.0.1 scores. Enrolment was **risk-based active case finding across 127 mobile events, not consecutive**. The CSV has **no event or cluster identifier**, so all our bootstrap intervals are participant-level.

- Split once before any tuning, stratified by outcome and scanner: **TUNE 1,352 (139 positive)** / **REPORT 1,356 (141 positive)**; TUNE fingerprint `1aa55cd5…`.
- **Both halves are exposed.** TUNE by design (every selection). REPORT was scored in six analyses, four of which influenced a decision (R11 audit). Nobody should describe REPORT as a clean test set; the record says so and the paper says so.
- The scanner field matters: the candidate's score sits 1.3 logits higher on scanner 3 at similar prevalence; the comparator's does not. See §4.3.

### 3.3 Shenzhen, Montgomery, DA/DB

Curated case-control sets. Shenzhen and Montgomery **are not untouched**: the lungs-only masking decision was informed by them (R4 note), and R19 used all 800 as additional *training* rows in an experiment that was rejected. Treat their AUROCs (0.969 / 0.969 for the production head) as exploratory. DA/DB: 176 images from the TBX11K extra lists, reported separately because they disagree (0.9755 / 0.9497).

### 3.4 Not used, and why

RayDINO (gated non-commercial weights; blocked). CLEAR concept probing — run, then shown to be negation-blind and withdrawn as an instrument (R14); its package stays in `artifacts/` for the record only.

---

## 4. Results you may quote, and the ones you may not

### 4.1 The ladder (record R2–R4)

| rung | cohort | regional 12×12 max | comparator |
|---|---|---|---|
| benchmark | TBX11K development (659) | 0.9368 | — |
| curated external | Shenzhen / Montgomery | 0.9690 / 0.9690 | — |
| curated external | DA / DB | 0.9755 / 0.9497 | — |
| **screening** | **Vietnam REPORT (1,356)** | **0.7242 [0.6843, 0.7618]** | **0.7862** |

### 4.2 Interventions (R3–R25), all pre-registered

| intervention | primary (TBX11K calibration) | screening (TUNE, exposed) | verdict |
|---|---|---|---|
| scanner-subspace removal | — | −0.0002 | null |
| active-TB label | −0.0037 | **+0.0203 [+0.0127, +0.0279]** | external only |
| head refit on local labels | 0 of 30 help | — | null |
| lesion-box supervision | +0.0019 | −0.0108 | null |
| clinical fusion (12 fields) | — | **+0.058 [+0.030, +0.085]** | external only |
| three training sources | +0.0033 [+0.0006, +0.0064] | +0.0036 | real, sub-threshold |
| encoder ensemble | +0.0006 | +0.011 | null |
| MAIRA-2 encoder | −0.0004 | **+0.0188 [+0.0015, +0.0364]** | external only |
| the three stacked | −0.0010 | +0.0286 (not additive) | — |
| label-free scanner centering | — | +0.0281 [+0.0074, +0.0486] at n=100 | works; **fails stability bound** |

### 4.3 What the fusion gain is

Leave-one-field-out (R23): only the **scanner identifier** carries the gain with an interval excluding zero (−0.0138; −0.0429 in previous TB). Questionnaire-only fusion = 0.7803 vs comparator 0.7780. **The point estimate above the comparator is a per-scanner intercept.** Do not write that the questionnaire solves previous TB, and do not write that the comparator is scanner-invariant — neither is established.

### 4.4 Sentences that are allowed

- "On an exposed screening cohort, the frozen candidate with questionnaire fusion achieved a numerically higher AUROC point estimate than the commercial comparator (+0.017 [−0.019, +0.053]); the paired interval includes zero, and no superiority, equivalence or non-inferiority claim is supported."
- "Attribution is exact at the 12×12 region level; on 49 development images with lesion boxes the decision cell overlapped an active lesion in 95.9 % (exploratory)."
- "The pipeline runs at a median 0.13 s per image on a 6 GB laptop GPU."

### 4.5 Sentences that are not

"Better than Lunit", "scanner-invariant", "self-calibrating", "consecutive cohort", "validated", "probability of TB" (the output is a screening logit at a development-fitted operating point — measured, not assumed: a sigmoid of it has ECE 0.49 and calibration slope 0.30 on the Vietnam cohort; `artifacts/probability_calibration_audit_20261101_1100`).

---

## 5. Software architecture — before and after

### 5.1 Before (13 Oct commits on `atul`)

Global linear probe on the RAD-DINO pooled embedding (`radino_full_clahe0_bin`), full image, all negatives; kNN OOD gate; v1 FastAPI app (`src/p2_serve/app.py`: `/health`, `/model`, `/analyse`, `/analyse-batch`) returning TB / REFER / NOT_TB with a conformal threshold. Internal 0.9949, Vietnam full cohort 0.6151. The border-only ablation showed 0.9094 of that came from outside the lungs.

### 5.2 After (this handover)

```
image (any size) ──standardise──> 512×512 u8 ──quality gate (C09 reasons)──> reject | pass
      │
      ├──lung mask (torchxrayvision PSPNet, GPU, float32; outside-lung → image mean)
      │
      ├──encoder (frozen ViT-B/14 @ 518 px: RAD-DINO [v2] or RAD-DINO-MAIRA-2 [v3])
      │       37×37 patch tokens ──adaptive avg-pool──> 12×12 × 768-d, L2 per region
      │
      ├──head: Linear(768→1) per region, 5 seeds ──max over 144 regions──> image logit
      │       (the arg-max region IS the decision: exact attribution)
      │
      ├──operating points: logit thresholds at 0.70 / 0.90 specificity (TBX11K development only)
      │
      └──optional fusion: logistic regression on standardised logit + 12 questionnaire/scanner fields
              (EXPLORATORY; refused unless all fields present; scanner columns are site-specific)
```

| layer | file | notes |
|---|---|---|
| pipeline | `src/p2_cls/pipeline_v2.py` | `load_bundle_v2`, `lungs_only`, `regional_features`, `head`, `fuse`, `encode_clinical`, `run_case_v2`; picks the encoder from `bundle.json`'s pinned backbone; segmenter on GPU when present |
| bundles | `artifacts/p2/bundle_v2_regional/`, `artifacts/p2/bundle_v3_candidate/` | `head.npz` (W [5,1,768], b [5,1]), `bundle.json` (provenance, thresholds, known external numbers, status), `fusion.json` |
| serving | `scripts/serve_v2.py` | registers `/v2/*` on top of `create_app()` from `src/p2_serve/app.py` — **no edit to the v1 app**; `--bundle DIR` or `P2_BUNDLE` selects the bundle; `--preload` |
| calibration guard | `src/p2_cls/site_calibration.py` | fail-closed label-free per-scanner centering proposal (R25) — EXPERIMENTAL |
| head-to-head | `artifacts/headtohead_prereg_v1/scripts/headtohead.py` | one-shot scorer for an untouched cohort; protocol hash guard; LOCK; no training path |
| builders / checks | `scripts/p2_37_build_bundle_v2.py`, `p2_39_build_bundle_v3.py`, `p2_38_smoke_v2.py`, `p2_40_smoke_v3.py`, `p2_41_segmenter_latency.py` | smoke tests prove the runtime path reproduces the cached training-time features (max |diff| 0.0056) |
| tests | `tests/test_pipeline_v2.py`, `test_bundle_v3.py`, `test_site_calibration.py`, `test_serve_v2_batch.py` | 27 tests, numpy-only, no model load |

### 5.3 API (`/v2`)

| route | in | out |
|---|---|---|
| `GET /v2/model` | — | model id, backbone, thresholds and where they were fitted, fusion status, provisional flag |
| `POST /v2/analyse` | multipart `file`; optional form `clinical` (JSON: sex, age_group, c, f, wl, ns, contact, past_tb, dm, rad_system) | `status`, `analysis.image_logit`, `operating_points`, `region_logits` (12×12), `argmax_region`, `lung_mask`, `fusion` or reason not applied, `latency_ms` |
| `POST /v2/batch` | multipart `files` (≤ 200); optional `clinical` only with a single file | cases ranked by screening score, flags at both operating points, decision cell, `overlay_key`; rejected cases unscored at the end; summary counts |
| `POST /v2/overlay` / `GET /v2/overlay/{key}` | file, or a key from `/v2/batch` | PNG: region logits shaded, decision cell boxed in green |
| `POST /v2/calibration/propose` | `{batch_scores, batch_clinical, site_reference}` | offsets or a refusal with reasons; always EXPERIMENTAL |
| `GET /v2/ui` | — | the triage screen |
| `GET /v2/schema/decision` | — | response shape |

Rules kept from v1: localhost, no auth, nothing persisted (overlays live in a 64-entry memory cache), a rejected image never carries a number, `provisional: true` always.

### 5.4 The triage screen (`/v2/ui`)

Drop up to 200 X-rays → ranked table (score, FLAG/clear at 0.70 and 0.90 specificity, decision cell, lung fraction, ms) → click a row to see the overlay. Optional questionnaire for a single image (fusion, labelled exploratory). Banner states the bundle, the backbone, where thresholds were fitted, and that the system is not clinically validated and scores are not probabilities. Live check on 2026-11-01: six development images ranked in 2.7 s (first call ≈2 s, then 120–150 ms each); two of three active-TB cases flagged, one missed at rank 4; healthy and sick-non-TB clear.

### 5.5 Running it

```
py -3 scripts/serve_v2.py --preload                      # http://127.0.0.1:8000/v2/ui  (bundle v2, default)
py -3 scripts/serve_v2.py --preload --bundle artifacts/p2/bundle_v3_candidate   # candidate v3, not promoted
py -3 -m pytest tests/ -q                                # 27 P2 tests + the team's
```

Environment: Python 3.13 locally (team CI is 3.10), torch (CUDA), transformers 5.14, torchxrayvision, scikit-learn 1.8, fastapi/uvicorn, PIL, opencv. `HF_HOME=D:\hf-cache`, `HF_HUB_DISABLE_XET=1` for downloads. **Windows gotcha:** import `pandas` before `torch` in any script that uses both, or the process segfaults (pyarrow/torch DLL clash) — `serve_v2.py` and `p2_41` do this deliberately.

---

## 6. Messages to each of you

### Tharun (P1 — data, splits, labels)

1. **The definitive C01 split is the single most important outstanding item.** Every TBX11K number in the paper, the record and the college report is conditional on the provisional one. When it lands, I rerun the cache/head/smoke chain (§2, ≈ 15 minutes) and regenerate the tables.
2. Please correct `tb1199` (`active_tb` → `latent_tb`) in `labels_tbx11k_trainval_v1.csv` and fill `active_tb_label` = 0 for the 140 latent rows in C01.
3. The 158 near-duplicate pairs (`artifacts/p2/dedup/provisional_near_dup_official_v0.csv`) are a dataset finding worth a paragraph in your data section; P1-07 review is still open.
4. If you have any contact with the Vietnam depositors, ask whether a **screening-event identifier** can be released — it would let us resample at cluster level.
5. The data request for an untouched cohort (`artifacts/headtohead_prereg_v1/DATA_REQUEST.md`) lists exactly which columns we need; if your data channels can reach TB Portals or a Bengaluru centre, that document is ready to send as is.

### Ritika (P3 — lesion detector)

1. The classifier now exposes a **12×12 region-logit map and the exact decision cell** on every analysed image (`analysis.region_logits`, `analysis.argmax_region`). Your detector's boxes can be compared against it directly: a box overlapping the decision cell is the "agreement" signal we measured at 95.9 % on 49 development images (`artifacts/box_supervised_head_20261101_0400/tables/localisation.csv`).
2. The link rule in `src/p2_cls/pipeline.triage` (`n_lesions` can move NOT_TB up to REFER, never down) is still unwired. If you want it live, give me a function that returns a lesion count for a 512×512 image and I'll call it from `run_stages_v2`.
3. Box supervision of the classifier was tested and rejected (R17): the detector should stay a separate component, not a training signal for the head.
4. The overlay endpoint draws cells, not lesions; if your detector produces boxes, drawing them on the same PNG is a small addition to `overlay_png` in `scripts/serve_v2.py`.

### Kunal (P4 — sealed evaluation, calibration, orchestration)

1. `serve_v2.py` is **additive** to your app: `register_v2(create_app())`. Nothing in `src/p2_serve/app.py` was edited. Your `/analyse` and `/analyse-batch` still answer with the v1 model.
2. The untouched-cohort head-to-head has its own one-shot scorer (`headtohead.py`, LOCK file, protocol hash guard). If you'd rather it go through your sealed machinery, the pieces to reuse are `check_protocol`, `build_manifest` and the LOCK semantics; the scoring itself is just `pipeline_v2.run_case_v2`.
3. **CI:** `.github/workflows/ci.yml` runs `pytest tests/` on Python 3.10 for PRs to main. My tests import numpy/PIL/fastapi only and skip when bundles are absent; `test_serve_v2_batch.py` needs `fastapi` and `httpx` (TestClient) in the CI requirements.
4. The PR `atul → main` is yours to review whenever the team is ready; the branch has 5 grouped commits plus the pre-registration commit. The triage screen (§5.4) is not yet pushed.
5. Conformal / calibration: no temperature scaling exists for the regional head; the operating points are plain logit thresholds fitted on development. If you want a calibrated probability output, that is a P4 addition on top of `image_logit`, and it must be fitted on calibration, never on Vietnam.

### Everyone

- Author order and the paper's final wording need the four of us to agree before anything goes to a venue. The paper (`report/paper_draft_v2.md`, PDF in `report/arxiv/`) and the P2 component report (`report/project_report_v2.md`) are on the branch.
- The project record (`artifacts/P2_PROJECT_RECORD_20261101.md`) indexes every run; if a number in any document disagrees with the record, the record wins and the document is wrong.

---

## 7. Open items

| item | owner | status |
|---|---|---|
| definitive C01 split | Tharun | outstanding — every TBX11K number conditional |
| `tb1199` label and empty `active_tb_label` fields | Tharun | outstanding |
| near-duplicate review (P1-07) | Tharun | outstanding |
| untouched cohort for the pre-registered head-to-head | team / Atul | data request ready; nothing received |
| college report in the department format | Atul, needs template | content final |
| co-author sign-off, preprint | all | pending |
| PR `atul → main`; triage screen push | Kunal / Atul | pending |
| detector ↔ classifier link rule | Ritika + Atul | unwired by design until the detector is final |
| calibrated probability on top of the logit | Kunal | not started |
