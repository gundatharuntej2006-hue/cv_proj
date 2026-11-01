# Label-free per-scanner calibration of the v3 candidate — results

**Run:** `artifacts/unlabelled_scanner_calibration_20261101_1000/` · 2026-11-01 · 1.7 min
**Protocol sha-256:** see `tables/protocol_hash.txt`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory, exposed.** Vietnam TUNE, evaluation group 677 (70 Xpert+), calibration
pools patient-disjoint. Nothing promoted.

---

## The result, in two halves

> **Half one — it works, with 25 unlabelled images per scanner.** Mean-centering the
> candidate's score on a scanner's own unlabelled batch recovers the *entire* supervised
> scanner-term gain: **C − A +0.0281 [+0.0074, +0.0486]** pooled, **+0.0557
> [+0.0065, +0.1004]** in previous TB (n = 100, draw-averaged); at n = 25 it is already
> 0.7795 against the supervised arm's unstable 0.7667. Referral burden, which under the
> raw score is 50 % on scanner 3 and 24 % on scanner 2 at the same rule, becomes
> 32 % / 31 % / 40 %. Labels never touched.
>
> **Half two — it fails the pre-registered stability bound, on one stress condition, by
> 0.02.** When scanner 2's calibration batch is drawn from previous-TB patients only, that
> scanner's sensitivity at the fixed rule falls from 0.677 to 0.555 (**move 0.122**,
> bound 0.10) and its burden from 0.301 to 0.198. The pre-registered reading rule
> therefore says **NOT USABLE AS IS**, for both C and D. The rule was written before the
> numbers; it is not relaxed after them.

## 1. Groups

Per-scanner stratified 50/50, seed 0: pools 102 / 332 / 241; evaluation 677 with 70
positives. Raw candidate on the evaluation group: pooled **0.7546**; within-scanner
0.684 / 0.787 / 0.828 — the pooled number is 0.03–0.07 below every scanner's own.

## 2. Arms × calibration batch size (per scanner; 50 draws; mean [2.5, 97.5 %])

| n | arm | pooled AUROC | previous-TB AUROC | sens @ top-30 % rule | burden s1 / s2 / s3 |
|---|---|---|---|---|---|
| — | **A raw** | 0.7546 | 0.6285 | 0.66 | 0.31 / 0.24 / **0.50** |
| 25 | B supervised | 0.7667 [0.697, 0.784] | 0.669 | 0.67 | 0.36 / 0.32 / 0.34 |
| 25 | **C centre** | **0.7795 [0.769, 0.786]** | 0.679 | 0.70 | 0.40 / 0.32 / 0.33 |
| 25 | D centre+scale | 0.7791 [0.767, 0.788] | 0.675 | 0.71 | 0.41 / 0.31 / 0.33 |
| 50 | B | 0.7749 [0.736, 0.785] | 0.673 | 0.68 | |
| 50 | C | 0.7807 [0.772, 0.785] | 0.680 | 0.69 | |
| 100 | B | 0.7786 [0.752, 0.785] | 0.678 | 0.70 | 0.42 / 0.31 / 0.31 |
| 100 | **C** | **0.7818 [0.778, 0.784]** | **0.683** | 0.70 | 0.40 / 0.31 / 0.32 |
| 100 | D | 0.7817 [0.777, 0.786] | 0.677 | 0.71 | |
| 200 | B | 0.7828 [0.779, 0.785] | 0.686 | 0.69 | |
| 200 | C | 0.7821 [0.780, 0.784] | 0.683 | 0.69 | |

Within-scanner AUROC is identical in every arm (per-scanner affine maps), as the protocol
said it would be: the whole effect is in the pooled ranking and the operating point.

Paired bootstrap at n = 100, draw-averaged scores vs A, evaluation group:

| contrast | Δ pooled | 95% CI | Δ previous TB | 95% CI |
|---|---|---|---|---|
| B (supervised) − A | +0.0281 | [+0.0070, +0.0484] | +0.0557 | [+0.0069, +0.1019] |
| **C (unlabelled centre) − A** | **+0.0281** | **[+0.0074, +0.0486]** | **+0.0557** | **[+0.0065, +0.1004]** |
| D (unlabelled centre + scale) − A | +0.0279 | [+0.0094, +0.0460] | +0.0513 | [+0.0091, +0.0898] |

## 3. Composition shift — scanner 2's batch drawn from a different population (n = 100)

Scanner-2 sensitivity / burden at the fixed rule; natural condition in bold for reference.

| scanner-2 batch | A sens / burden | B sens / burden | **C sens / burden** | D sens / burden | C pooled AUROC |
|---|---|---|---|---|---|
| **natural** | 0.587 / 0.237 | 0.673 / 0.309 | **0.677 / 0.301** | 0.668 / 0.293 | 0.7821 |
| prevalence 0 (negatives only) | 0.602 / 0.250 | **0.177 / 0.064** | 0.684 / 0.330 | 0.682 / 0.322 | 0.7829 |
| prevalence 0.30 | 0.574 / 0.215 | 0.813 / **0.457** | 0.604 / 0.246 | 0.586 / 0.231 | 0.7771 |
| previous TB only | 0.572 / 0.214 | 0.638 / 0.269 | **0.555 / 0.198** | 0.560 / 0.203 | 0.7651 |
| no previous TB | 0.605 / 0.252 | 0.679 / 0.326 | 0.686 / 0.347 | 0.686 / 0.332 | 0.7835 |

| | C | D | bound |
|---|---|---|---|
| worst scanner-2 sensitivity move | **0.122** | 0.108 | ≤ 0.10 |
| worst scanner-2 burden move | 0.103 | 0.090 | ≤ 0.10 |
| gain vs A at n = 100 (B: +0.024) | +0.027 | +0.027 | ≥ half of B |
| **verdict** | **NOT USABLE AS IS** | **NOT USABLE AS IS** | |

## 4. Readings

- **The scanner gap is closable without labels, and cheaply.** 25 consecutive unlabelled
  images per scanner is enough; the supervised term needs ~200 labelled ones to be
  stable (n = 25: interval [0.697, 0.784]). This is the deployment advantage the product
  thesis was missing — provided the next point is handled.
- **The failure mode is exactly the one GPT named.** Centering removes whatever makes the
  batch's mean differ from other scanners, acquisition *or* population. A batch of
  previous-TB patients scores higher (healed-disease appearance), so centering on it
  pushes that scanner down and loses 12 points of sensitivity there. Prevalence 0.30
  costs 7 points. Prevalence 0 costs nothing (positives are 10 % of the mean).
- **The supervised term fails worse, in the opposite direction**: negatives-only
  batches collapse it (sens 0.18), enriched batches inflate burden to 46 %. The unlabelled
  method is the *more* robust of the two; it is still not robust enough by the rule.
- **What would make it usable:** the calibration batch must be a *consecutive screening
  sample* of the site's population, not a selected subgroup — a deployment requirement
  that can be checked (batch previous-TB fraction, batch size) and that the product can
  enforce by refusing calibration on fewer than 25 images or on a batch whose
  questionnaire composition is far from the site's. Testing that guard is a new run, not
  this one.
- The candidate with unlabelled centering sits at 0.782 pooled on this evaluation group,
  with no label, no questionnaire and no site-specific fit — the clinical layer is then
  additive on top. That is the configuration to carry into the write-up as "scanner-
  aware calibration", with the stability caveat in the same sentence.

## 5. Files

`tables/groups.json` · `arms_by_size.csv` · `paired_vs_raw_n100.csv` ·
`composition_shift_n100.csv` · `run_manifest.json` (verdict block) · `protocol_hash.txt` ·
`scripts/calib.py` · `tests/`. No per-case output; no patient identifier. REPORT and
`internal_test` not referenced.
