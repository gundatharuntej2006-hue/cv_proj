# Cross-dataset evaluation plan: Shenzhen and Montgomery

**Owner:** Person 2 · **Date:** 2026-10-21 · **Repo HEAD:** `a78e73a` ·
**Status: PHASE 4 NOT EXECUTED.** Two blockers, one of which requires your decision.

## 1. Cohorts as held in this repository

| cohort | source | n | TB | label source | version |
|---|---|---|---|---|---|
| Shenzhen | NLM / Shenzhen No. 3 People's Hospital, via `data/raw/_external_flat/shz__*` | 662 | 336 | filename suffix `_0`/`_1` | NLM public release |
| Montgomery | NLM / Montgomery County, MD, via `data/raw/_external_flat/mcu__*` | 138 | 58 | filename suffix `_0`/`_1` | NLM public release |

Both arrive as raw full-resolution PNG (Shenzhen ~3000×2919, Montgomery 4892×4020) and are
standardised by the same `cv2.resize(..., INTER_AREA)` path as every other cohort.

**Overlap with training:** verified absent. Maximum cosine similarity between any of the 800
NLM images and any of the 8,400 TBX11K core-list training embeddings is **0.8389**, with zero
pairs above 0.90. Patient identifiers are per-image filenames; the NLM sets are one image per
patient, so patient-level separation follows. Duplicate detection within the NLM sets
themselves has **not** been run and cannot be ruled out.

## 2. Provenance gate — the finding that matters

**These cohorts are NOT untouched for the configuration we would be evaluating.**

| decision in the current configuration | did it consult Shenzhen/Montgomery? | evidence |
|---|---|---|
| lungs-only masking (D09 revisited) | **YES** | `provenance_table.csv` row "D09 revisited → lung crop", cohorts = "TBX11K dev + Shenzhen + Montgomery"; `docs/p2_shortcut_does_not_transfer.md` §"This reopens D09, now on accuracy grounds", which cites Montgomery +0.037 as the motivating evidence |
| healthy-only negatives | no (full Vietnam cohort) | `provenance_table.csv` |
| pooling grid + aggregator | no (Vietnam TUNE only) | `mil_sweep.json`, scorer committed `1c81573` before results existed |

The current best configuration is **lungs-only masking + healthy-only negatives + 12×12 max
pooling**. Its *first* component was chosen partly because of how it scored on Montgomery and
Shenzhen. Scoring that configuration on those cohorts and presenting the result as
independent validation would be circular.

**Additionally, the foundation checkpoint's selection history cannot be cleared.** The
RAD-DINO model card states the released checkpoint was chosen by linear probing on the
validation sets of "the evaluation datasets described in the paper" and **does not name
them**. Shenzhen and Montgomery are among the most common public TB benchmarks. Whether they
influenced checkpoint selection is **unknown and not establishable from the card**. This
applies to every result anyone reports on these cohorts with this backbone, not only ours,
but it must be stated.

## 3. Technical blocker

**Regional embeddings do not exist for these cohorts.**

| cohort | global embeddings | regional (12×12) |
|---|---|---|
| TBX11K | yes | yes (1.35 GB) |
| Vietnam | yes | yes (0.56 GB) |
| **Shenzhen + Montgomery** | **yes** (`radino_nlm_lungs_only_v1.npz`) | **absent** |

There is also **no stored regional head checkpoint**. The MIL heads are retrained from cached
embeddings at every run with fixed seeds `(17, 29, 43, 61, 79)`; reproducibility rests on
seeds and cached features, not on a weights file. `artifacts/p2/bundle/head.npz` is the older
*global* head and is not the regional model.

Producing the missing features is a **feature-extraction pass with the frozen checkpoint** —
no training, no gradient, no tuning. It is nonetheless computation beyond "use existing
features", so per the brief it is reported as a blocker rather than performed.

Estimated cost: 800 images at 12×12 ≈ **2 minutes**, ≈ 170 MB.

```
py -3 scripts/p2_31_regional_mil.py --stage embed      # after adding the NLM image set
```

## 4. What a comparison could and could not support

**Could support (exploratory):**
- Whether regional max-pooling scores higher or lower than global pooling **on the same
  images, same labels, same preprocessing** — a within-pair comparison where the shared
  contamination affects both arms equally.
- Per-dataset AUROC/AUPRC with bootstrap intervals, reported separately.

**Could NOT support:**
- "Independent external validation" — the lungs-only component was selected using these
  cohorts.
- "Untouched test set" — see above, plus the unresolvable checkpoint-selection history.
- Any threshold selection or model choice. Thresholds would have to come from TBX11K
  development or the Vietnam TUNE half and transfer unchanged.
- A combined Shenzhen+Montgomery headline number. Montgomery has 58 positives; its interval
  will be very wide and pooling would hide that.
- Comparison against the published F1 figures (Shenzhen 0.7707, Montgomery 0.8348 from
  `TREAT-MMTB_011`) without matching their threshold and label protocol, which the page does
  not give.

## 5. Recommendation

**Run it, labelled explicitly as a paired within-cohort pooling comparison, not as external
validation.** The within-pair logic is what rescues it: both arms inherit the same
contamination, so the *difference* between global and regional pooling remains interpretable
even though neither absolute number is a clean generalisation estimate.

If you would rather not spend the two minutes of extraction, the alternative is to state in
the report that focal pooling has been evaluated on one cohort only — which is the current
honest position and is itself a reportable limitation.

**Decision required:** proceed with the extraction pass, or stop here.
