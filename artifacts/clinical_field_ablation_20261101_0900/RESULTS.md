# Which clinical fields carry the fusion gain — results

**Run:** `artifacts/clinical_field_ablation_20261101_0900/` · 2026-11-01 · 5.3 min
**Protocol sha-256:** see `tables/protocol_hash.txt`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory, exposed.** Vietnam TUNE by nested CV. A decomposition, not an intervention.

---

## The result

> **One field carries the gain, and it is not a clinical one.** Leaving out the
> **scanner identifier** costs **−0.0138 [−0.0250, −0.0029]** overall and
> **−0.0429 [−0.0702, −0.0180]** in previous TB. No other single field has an interval
> excluding zero on the cost side; `age_group` is the nearest (−0.0096 [−0.0208, +0.0011]).
>
> **Without the scanner identifier the fused candidate is 0.7803 — parity with the
> comparator's raw 0.7780, not above it.** The "numerically above Lunit" point estimate
> of R22 is carried by a two-column acquisition correction that the comparator's score
> does not need (it gains nothing from fusion, R16). Image + scanner alone, no
> questionnaire, is 0.7788 — the comparator's number.

## 1. Gates — passed

| check | expected | here |
|---|---|---|
| X raw (R21/R22) | 0.7585 | **0.7585** |
| full 12-field fusion (R22) | 0.7941 | **0.7941** |
| comparator full (R16/R22) | 0.7778 | **0.7778** |

## 2. Arms — M2 fusion on the `ours_X` score, 10-fold × 20 repeats

| arm | fields | AUROC | ± | previous TB | AUPRC | sens @ 0.70 spec | Δ vs full | 95% CI | prev-TB Δ | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| **full** | 12 | **0.7941** | 0.0023 | 0.7349 | 0.300 | 0.741 | | | | |
| without sex | 11 | 0.7917 | | 0.7277 | 0.297 | 0.740 | −0.0026 | [−0.0101, +0.0058] | −0.0080 | [−0.0223, +0.0082] |
| without age_group | 11 | 0.7847 | | 0.7216 | 0.290 | 0.733 | −0.0096 | [−0.0208, +0.0011] | −0.0130 | [−0.0401, +0.0129] |
| without c (cough) | 11 | 0.7958 | | 0.7372 | 0.303 | 0.743 | +0.0014 | [+0.0010, +0.0019] | +0.0016 | [+0.0003, +0.0030] |
| without f (fever) | 11 | 0.7920 | | 0.7261 | 0.296 | 0.733 | −0.0023 | [−0.0088, +0.0030] | −0.0091 | [−0.0262, +0.0061] |
| without wl | 11 | 0.7922 | | 0.7334 | 0.296 | 0.742 | −0.0022 | [−0.0089, +0.0041] | −0.0014 | [−0.0192, +0.0130] |
| without ns | 11 | 0.7952 | | 0.7364 | 0.304 | 0.742 | +0.0011 | [+0.0006, +0.0017] | +0.0016 | [+0.0002, +0.0032] |
| without contact | 11 | 0.7948 | | 0.7340 | 0.304 | 0.744 | +0.0005 | [−0.0028, +0.0033] | −0.0019 | [−0.0101, +0.0058] |
| without past_tb | 11 | 0.7902 | | **0.7394** | 0.283 | 0.757 | −0.0040 | [−0.0137, +0.0059] | +0.0042 | [−0.0016, +0.0102] |
| without dm | 10 | 0.7956 | | 0.7314 | 0.302 | 0.736 | +0.0014 | [−0.0022, +0.0054] | −0.0042 | [−0.0128, +0.0038] |
| **without scanner** | 10 | **0.7803** | 0.0027 | **0.6941** | 0.297 | 0.718 | **−0.0138** | **[−0.0250, −0.0029]** | **−0.0429** | **[−0.0702, −0.0180]** |
| image + past_tb | 1 | 0.7591 | | 0.6443 | 0.270 | 0.683 | −0.0358 | [−0.0584, −0.0138] | −0.0929 | [−0.1478, −0.0408] |
| image + scanner | 2 | **0.7788** | | 0.7102 | 0.267 | 0.717 | −0.0163 | [−0.0335, +0.0002] | −0.0268 | [−0.0736, +0.0164] |
| image + symptoms | 4 | 0.7623 | | 0.6639 | 0.267 | 0.668 | −0.0325 | [−0.0513, −0.0129] | −0.0721 | [−0.1119, −0.0314] |
| image + demographics | 2 | 0.7692 | | 0.6676 | 0.269 | 0.675 | −0.0256 | [−0.0438, −0.0068] | −0.0689 | [−0.1067, −0.0303] |
| questionnaire only (no scanner) | 10 | 0.7803 | | 0.6941 | 0.297 | 0.718 | −0.0138 | [−0.0250, −0.0029] | −0.0429 | [−0.0702, −0.0180] |
| comparator full | 12 | 0.7778 | 0.0019 | 0.7144 | 0.306 | 0.745 | | | | |

## 3. Readings

- **The fusion gain decomposes as roughly +0.020 scanner correction + +0.022
  questionnaire** (raw 0.7585 → image+scanner 0.7788 → full 0.7941; raw → questionnaire
  only 0.7803). The questionnaire's contribution is diffuse: no single field carries it,
  and dropping cough or night sweats *helps* by a hair (fewer parameters).
- **`past_tb` is not what helps in previous TB.** Dropping it leaves the previous-TB
  AUROC unchanged (+0.004); dropping the scanner costs −0.043 there. Our model's
  previous-TB weakness is, to a measurable extent, a scanner confound expressed in that
  subgroup (R7's finding that scanner differs by history now has a cost attached).
- **The comparator does not have this confound.** Its score is already scanner-invariant,
  which is why fusion gave it nothing (R16) and why the acquisition correction is a gain
  for us and not for it. "Fused ours above raw Lunit" is therefore, in substance,
  "ours after per-scanner recalibration at parity with Lunit without one".
- **Availability at screening time:** every field is from the screening questionnaire or
  the acquisition record; nothing here is a laboratory or follow-up result. But the
  scanner identifier is a *per-site* variable — its coefficients are fitted to these two
  scanners and have no meaning at a third site. A new site needs its own scanner term
  fitted on local labelled data, which is R15's threshold recalibration in another form.

## 4. What changes

- **R22's sentence is revised:** on this cohort, the candidate with the 10-field
  questionnaire is at **parity** with the comparator (0.7803 vs 0.7780); the two-column
  scanner correction, not clinical context, is what puts the point estimate above.
- A candidate v3 should expose the scanner term as what it is — site calibration — and
  the questionnaire as the clinical layer, with their separate contributions stated.
- The product thesis gains a measured statement of what the comparator has that we lack:
  **scanner invariance**, worth ≈ 0.02 on this cohort and 0.04 in previous TB.

## 5. Files

`tables/ablation_results.csv` · `paired_vs_full.csv` · `reproduction_gate.csv` ·
`run_manifest.json` · `protocol_hash.txt` · `scripts/ablation.py` · `tests/`. No per-case
output; no patient identifier. REPORT and `internal_test` not referenced.
