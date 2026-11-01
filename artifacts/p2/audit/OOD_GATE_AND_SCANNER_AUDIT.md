# OOD gate decoupling, count verification, and scanner-stratified audit

**Owner:** Person 2 · **Date:** 2026-10-21 · **Task type:** implementation + verification.
No model was retrained, no weights changed, no pooling sweep re-run, no AUC improved.

Companion outputs (new files, nothing overwritten):
- `scanner_stratified_summary.csv` — machine-readable Phase D table
- `vietnam_case_level_gate_audit.csv` — 2,708 per-case records (pre-existing, preserved)
- `provenance_table.csv`, `LEAKAGE_PROVENANCE_AUDIT.md` — pre-existing, preserved

---

## 1. What the code did before any change

`src/p2_cls/pipeline.py::run_stages` ran five stages in order. Stage 3 scored the k-NN OOD
gate and, when the score exceeded the threshold, **returned immediately with
`analysis=None`**.

The consequence is stronger than "the gate could override a referral": **the classifier
never ran at all** for a flagged case, so no referral could exist to be overridden. The case
left the pipeline with no probability and no triage category.

Two details made this worse and the remedy cheap:

- **The embedding is computed at stage 2, before the gate.** The expensive work was already
  done, so classifying anyway costs almost nothing. There was no efficiency justification
  for the early return.
- **`worklist()` sorted `analysis=None` last.** Even if a flagged case had been surfaced,
  the ordering would have buried it beneath every analysed case.

**Every consumer of the gate was mapped before editing:**

| location | role | suppressive? |
|---|---|---|
| `p2_cls/pipeline.py` stage 3 | primary inference | **yes — the only one** |
| `p2_gate/api.py::gate` | R2 contract; returns `gate_pass` as a field | no — advisory already |
| `p2_serve/app.py` | HTTP layer; passes `run_stages` output through | no |
| `p2_cls/contracts.py` | C09 schema, `gate_pass` column | no — storage only |

**Input integrity was already separate** and remains so: decode failures and the four C09
quality rules (`blank`, `under_exposed`, `over_exposed`, `heavily_cropped`) short-circuit at
stages 0–1, before any embedding or OOD scoring. These are genuine non-diagnostic states and
correctly return `analysis=None`.

---

## 2. Changes made

**One source file: `src/p2_cls/pipeline.py`.**

| change | rationale |
|---|---|
| OOD no longer returns early; default policy `advisory` | a flag must not cancel a referral |
| added `review_required`, `review_reasons` | additive review request |
| added `analysis.requires_additional_review` | flag travels with the analysis |
| `ood_policy="blocking"` retained, explicitly non-default | reproduce prior experiments |
| unknown policy raises `ValueError` | no silent fallback |
| `ood_policy` threaded through `run_case`, `run_stages`, `run_batch` | caller control |
| `worklist()` docstring states flagged cases keep priority | prevent re-suppression via ordering |

Quality and decode failures are untouched: still `REJECTED`, still `analysis=None`, still
distinct from model-based OOD.

**Test files:** one new (`tests/p2_cls/test_ood_cannot_suppress.py`, 11 tests), one updated
(`tests/p2_cls/test_pipeline.py`).

### The cost of this change, stated plainly

Under `advisory`, a genuinely alien input — a photograph, a lateral film — now receives a
probability instead of being stopped. That probability is meaningless. It is accompanied by
`review_required=True` and `out_of_distribution` in `review_reasons`, but it exists.

This is a real trade and was made deliberately: the measured harm of suppression (21
deferred tuberculosis patients, 7 of them intercepted mid-referral, threefold scanner
disparity) is concrete and falls on patients, whereas the harm of a flagged meaningless
probability falls on a reviewer who has been told not to trust it. A deployment that cannot
rely on reviewers reading flags should set `ood_policy="blocking"` and accept the measured
cost.

---

## 3. Phase C — counts recomputed from raw records

Recomputed from the 2,708 individual rows, not from the rendered summary.

**Partition check:** the three outcome categories sum to 2,708 and `pid` is unique across
all rows — every patient counted exactly once.

| quantity | count | denominator |
|---|---|---|
| total cases | 2,708 | — |
| TB cases | 280 | — |
| referrals **without** gate | 929 | all |
| — true positives | 174 | 280 TB |
| — false positives | 755 | 2,428 non-TB |
| referrals **with** gate | 878 | all |
| — true positives | 167 | 280 TB |
| — false positives | 711 | 2,428 non-TB |
| deferred | 284 | all |
| — TB deferred | 21 | 280 TB |
| — TB intercepted (was classifier-positive) | **7** | 280 TB |
| total human-handled with gate | 1,162 | all |

**All ten reported figures reproduce exactly. No discrepancy.** Net workload
= (878 + 284) − 929 = **+233**, computed on the all-case population throughout.

**Workload definition, stated rather than assumed:** referrals and deferrals are counted as
one human-handled case each. They are *not* assumed to cost equal effort — a deferral is a
"look at this image and decide" task, a referral is "arrange confirmatory testing". Totals
are reported separately so a reader can weight them differently.

**A deferral is not a missed diagnosis.** Each deferred case enters a human review pathway
whose outcomes this project has no data for. No end-to-end clinical sensitivity is estimated.

---

## 4. Phase D — scanner-stratified results

Bootstrap 95% intervals, 4,000 resamples.

| scanner | n | TB | prevalence | deferral rate | 95% CI | TB deferred | TB deferral rate | 95% CI | TB referrals intercepted |
|---|---|---|---|---|---|---|---|---|---|
| 1 JPI Healthcare | 408 | 44 | 0.108 | **20.10%** | 16.4–24.0 | 9 | 20.45% | 9.1–31.8 | **5** |
| 2 DRTECH | 1,332 | 151 | 0.113 | 6.91% | 5.6–8.3 | 7 | 4.64% | 1.3–8.0 | 0 |
| 3 Konica Minolta | 968 | 85 | 0.088 | 11.36% | 9.4–13.4 | 5 | 5.88% | 1.2–11.8 | 2 |
| **all** | 2,708 | 280 | 0.103 | 10.49% | 9.4–11.6 | 21 | 7.50% | 4.6–10.7 | 7 |

**The overall deferral disparity is established.** JPI's interval (16.4–24.0%) does not
overlap DRTECH's (5.6–8.3%). A patient's chance of being deferred depends roughly threefold
on which machine imaged them.

**The TB-specific disparity is suggestive, not established.** JPI's TB interval is
9.1–31.8% on 44 cases; it only just clears DRTECH's upper bound of 8.0%. Five of seven
intercepted TB cases are on JPI, which has 15% of the cohort — striking, but seven events is
too few to carry a strong claim.

**It is not explained by case mix.** Chi-square on scanner × TB status = **4.051 on 2 df**
(critical 5.99 at p = 0.05): scanner is **not** significantly associated with TB prevalence.
So the deferral disparity cannot be attributed to one machine seeing sicker patients.

**Causation is not claimed.** `rad_system` may proxy for site, catchment population,
acquisition protocol, or operator in a community screening programme. The defensible
statement is: *deferral rate varies threefold by scanner, this is not explained by differing
TB prevalence, and the mechanism is unidentified.* Establishing it would need site and
protocol metadata the deposit does not carry.

**TBX11K cannot be analysed this way.** Its COCO records carry `id`, `file_name`, `width`,
`height`, `date_captured`, `license`, `coco_url`, `flickr_url` — no site, hospital, scanner
or device field. Hospital-held-out validation there is not constructible and no site labels
were invented.

---

## 5. Unresolved questions

1. **Why does deferral vary by scanner?** Not prevalence. Possibly detector physics,
   post-processing, collimation habits, or site population. Needs metadata we do not have.
2. **What happens to the 21 deferred TB patients?** No human-review outcome data exists.
3. **Is the advisory trade correct in a real clinic?** It assumes reviewers read flags. That
   is an assumption about process, not a measured property.
4. **Does the scanner disparity generalise?** One cohort, three devices.
5. **Should the gate exist at all in this workflow?** On this cohort it removes 44 false
   positives and adds 233 reviews. Restricting it to input-validity checking remains the
   recommendation from `docs/p2_gate_operational_audit.md`.

---

## 6. Reproduction

```
py -3 -m pytest tests/p2_cls/test_ood_cannot_suppress.py -q     # 11 safety tests
py -3 -m pytest tests -q                                        # full suite
py -3 scripts/p2_35_gate_operational_audit.py                   # regenerates the audit JSON
```

The Phase C and D tables are recomputed from `vietnam_case_level_gate_audit.csv` by the
inline analysis recorded in this task; `scanner_stratified_summary.csv` is its output.

---

## 7. What this did not do

No retraining, no weight changes, no pooling sweep re-run, no threshold tuned on REPORT or
any held-out cohort, no new pooling methods, no prevalence adjustment. **AUC is unchanged at
0.7242** and that figure's provenance caveats from `LEAKAGE_PROVENANCE_AUDIT.md` still apply
in full.

**Recommended next research step:** re-score the three curated external sites under the
regional representation. Focal pooling has only been evaluated on Vietnam, so whether it
helps or harms Shenzhen, Montgomery and DA/DB is unknown — and that gap, not any further
gate work, is what currently limits what can be claimed about the pooling result.
