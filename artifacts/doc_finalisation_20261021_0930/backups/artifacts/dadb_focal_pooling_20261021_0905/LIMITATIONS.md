# Limitations — DA/DB focal-pooling evaluation

---

## 1. DA/DB's clean status is about *downstream selection*, not the backbone

**Verified:** no DA/DB case ID appears in `provisional_c01.csv` (train, development,
calibration, internal_test) or in `labels_tbx11k_trainval_v1.csv`. `git grep -l dadb` over
`scripts/` and `src/` returns one file. DA/DB appears in none of the fourteen rows of
`artifacts/p2/audit/provenance_table.csv`. So **DA/DB informed no project configuration
decision** — which is what makes this the cleanest cross-cohort comparison available here.

**Not verified, and not claimable:** that the RAD-DINO checkpoint never encountered these
images. The model card states the released checkpoint was selected by linear probing on
validation sets of *"the evaluation datasets described in the paper"*, **which the card
does not name**. DA/DB (NITRD, New Delhi) is not among the listed pre-training corpora
(MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX), but **checkpoint-selection exposure cannot
be ruled out**. DA/DB is therefore *unused for downstream selection*, not *untouched*.

**Patient-level overlap cannot be checked** — no patient identifiers ship with DA/DB, so
within- and cross-cohort patient duplication is unverifiable by construction.

## 2. The cohort is small, and one arm of the comparison rests on 22 positives

| cohort | n | TB | non-TB |
|---|---|---|---|
| DA | 72 | 22 | 50 |
| DB | 104 | 52 | 52 |

DA's 95% CIs are correspondingly wide — arm B [0.9354, 0.9990] — and its paired interval
[−0.0357, +0.0299] is wide enough to contain effects as large as the one DB shows. **DA's
null is not evidence that regional pooling does not help there**; it is a cohort too small
and too close to ceiling to tell. The DB interval [+0.0159, +0.1328] is also wide: the data
are consistent with anything from a marginal to a very large gain.

## 3. Pooling the two cohorts is not a bigger sample

DA and DB have prevalence 0.306 and 0.500 and are different collections. Pooling them does
not produce a draw from one population. Worse, **a pooled AUROC can move purely because the
two cohorts' score scales differ**, independently of within-cohort discrimination — the
exact mechanism quantified for scanner offsets in the previous run, where re-aligning three
score scales moved pooled AUROC by +0.03 while leaving every within-group AUROC at
precisely 0.0000 change. The pooled row is marked supplementary throughout for that reason.

## 4. This is a balanced case-control set, not a screening population

Prevalence 0.420. The project's central finding is the collapse from curated benchmarks to
consecutive screening: the same weights score **0.9636 here** and **0.7242** on the Vietnam
cohort (prevalence 0.103) with an Xpert reference standard. **Nothing in this result
narrows that gap**, and the DA/DB figures sit at the easy end of the same ladder as
Shenzhen (0.969) and Montgomery (0.969). Quoting 0.96 as this system's performance would be
a serious misreading.

## 5. Preprocessing-selection history contaminates the configuration, though not this cohort

DA/DB itself informed no choice, but **the configuration it scores was partly chosen using
other cohorts**. Lungs-only masking and healthy-only negatives were adopted with reference
to Shenzhen and Montgomery (`hard_negatives.json`, `learning_curve.json` both score them),
and the 12×12 max aggregator was selected on the Vietnam TUNE half. So DA/DB is an
out-of-sample read of a configuration selected elsewhere — which is genuinely informative,
and is *not* the same as an independent estimate of generalisation for the pipeline as a
whole.

## 6. The threshold does not transfer, and this evaluation does not fix that

Both arms were scored at a threshold fixed at 70% specificity on TBX11K development. On
DA/DB, arm A achieves specificity 0.44–0.46 against the 0.70 it was set for. Arm B
transfers better (0.880 and 0.654) but is still far from its target. **The operating-point
rows are descriptive of transfer failure, not estimates of deployment behaviour**, and no
DA/DB threshold was chosen — deliberately.

## 7. The sanity-fail case

One case, `px4`, fails the pre-existing label-blind validity rule
(`p1_build_c04_c05.roi_from_mask` → `sanity_pass`) on `components`: the segmenter found one
lung field of at least 1% of the frame, where two are expected. **The committed pipeline
counted but did not exclude it, so the primary analysis keeps all 176**, and a 175-case
sensitivity analysis is reported alongside. Excluding it changes the pooled difference from
+0.0465 to +0.0454 and flips nothing.

The case was identified from mask geometry alone and the exclusion rule predates this run;
neither was chosen with reference to the effect on any metric. But note the inconsistency
it exposes: `radino_dadb_lungs_only_v1.npz` records `n_sanity_fail = 1` while
`dadb_third_site.json` reports n = 176 — the published artifact scored a case its own
validity check had flagged. That is pre-existing and still unresolved upstream.

## 8. What cannot be concluded from this run

- **Not clinical readiness.** Nothing here is prospective, and no clinical endpoint was
  measured.
- **Not generalisation to new hospitals.** Two collections from one Delhi institution.
- **Not causal shortcut removal.** This run made no representational intervention and
  measured no shortcut. The previous run's finding stands: scanner information is not
  linearly removable, and the TB head does not appear to use the part that is.
- **Not architectural novelty.** Patch-level aggregation over a frozen chest-radiograph
  foundation model is established prior art; RADAR covers it and adds mitigation this
  project lacks.
- **Not an explanation of the DA/DB asymmetry.** Why DB gains +0.072 and DA gains nothing
  is unexplained. Ceiling effects on DA are the obvious hypothesis and were not tested.
- **Not a licence to pool.** Four cohorts now agree on direction, but they are four curated
  case-control sets plus one screening cohort that disagrees with all of them.

## 9. One result in this run is larger than the one it was designed to measure

Switching the training negative composition from all-negatives to healthy-only moves DA/DB
by **+0.138 AUROC** (0.7791 → 0.9172, verified by exactly reproducing the published
0.7791). Focal pooling adds **+0.0465** on top. This was not the question being asked, the
comparison was not pre-registered as a primary endpoint, and it is reported as an
observation — but it is the larger of the two effects on the only cohort never used for
selection, and it means **the project's "India 0.7932" ladder entry is stale**.
