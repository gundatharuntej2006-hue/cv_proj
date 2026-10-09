# P3-12 · Grad-CAM review session with the guide (development cases)

> **Standing caveat.** Heatmaps are *attention evidence, not a finding*. Agreement between a heatmap and the
> radiologist boxes is overlap with incomplete annotations; it is **not** proof of clinical correctness, and
> disagreement is not proof of error. Development data only; no internal-test or challenge cases are shown.

## 1. Case selection rule (pre-registered, written 2026-10-09, BEFORE any case was selected)

Source: P4's C12 triage table for development, `artifacts/p4/triage/triage_tbx11k_development_v1.parquet`
(P4's mock under `mock/` only for dry runs), joined on `image_id` with C01 (`data/splits/tbx11k_splits_v1.csv`)
for the true label.

1. **Eligible:** development rows with `gate_pass = true` (rejected images have no classifier output or
   heatmap) and `tb_label` 0 or 1.
2. **Correct / incorrect** uses the screen-positive definition of methodology section 12 (TB and REFER count
   as positive), *with* the link rule: correct ⇔ `screen_positive == (tb_label == 1)`.
3. **Four cells, 10 cases each:** true positive (TB, correct), true negative (non-TB, correct),
   false negative (TB, incorrect), false positive (non-TB, incorrect). This gives 20 correct + 20 incorrect
   with TB and non-TB in both groups.
4. **Draw:** within each cell, sort the image ids and draw 10 without replacement with
   `numpy.random.default_rng(20261009)` (cells in the order TP, TN, FN, FP, one generator). If a cell has fewer than
   10 eligible cases, take all of them and fill the shortfall from the other cell of the same group
   (correct or incorrect) with the same generator; the shortfall is recorded in `selection.csv`.
5. **Nothing else** influences the choice: heatmaps and overlays are generated only after the 40 ids are fixed,
   and no case is swapped afterwards. The 40 cases are shown in a shuffled order (same seed) so the sheet does not
   group correct and incorrect cases.

Tool: `python -m p3_xai.review select` (writes `artifacts/p3/review/selection.csv`), then
`python -m p3_xai.review render` (40 overlays + `review_sheet.html`).

## 2. Session

| | |
|---|---|
| Date | ____ |
| Place / call | ____ |
| Attendees | guide: ____ ; P3: ____ ; others: ____ |
| Classifier (model_id, D11) | ____ |
| Detector (D17 checkpoint SHA, D18 threshold) | ____ |
| Grad-CAM configuration (D19) | ____ |
| Selection file + seed | `artifacts/p3/review/selection.csv`, seed 20261009 |

## 3. Cases

Review order as on the sheet. *Correct* = screen-positive decision matches the true label.

| # | Case ID | True label | Predicted category | Correct? | Guide comment | Action |
|---|---|---|---|---|---|---|
| R01 | | | | | | |
| R02 | | | | | | |
| R03 | | | | | | |
| R04 | | | | | | |
| R05 | | | | | | |
| R06 | | | | | | |
| R07 | | | | | | |
| R08 | | | | | | |
| R09 | | | | | | |
| R10 | | | | | | |
| R11 | | | | | | |
| R12 | | | | | | |
| R13 | | | | | | |
| R14 | | | | | | |
| R15 | | | | | | |
| R16 | | | | | | |
| R17 | | | | | | |
| R18 | | | | | | |
| R19 | | | | | | |
| R20 | | | | | | |
| R21 | | | | | | |
| R22 | | | | | | |
| R23 | | | | | | |
| R24 | | | | | | |
| R25 | | | | | | |
| R26 | | | | | | |
| R27 | | | | | | |
| R28 | | | | | | |
| R29 | | | | | | |
| R30 | | | | | | |
| R31 | | | | | | |
| R32 | | | | | | |
| R33 | | | | | | |
| R34 | | | | | | |
| R35 | | | | | | |
| R36 | | | | | | |
| R37 | | | | | | |
| R38 | | | | | | |
| R39 | | | | | | |
| R40 | | | | | | |

## 4. Overall observations

- Where does attention fall on correct TB cases (lesions, lung apices, outside the lungs, markers/text)? ____
- Patterns on errors (false negatives / false positives)? ____
- Any sign of shortcut learning (borders, devices, labels)? ____
- Actions agreed (owner, due date): ____

## 5. Sign-off

The guide has reviewed the 40 cases above. The comments are recorded as given, and the caveat at the top applies.

Guide: ______________________  Date: __________     P3: ______________________  Date: __________
