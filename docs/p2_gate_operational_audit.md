# The OOD gate is net-harmful on a real screening cohort

**Owner:** Person 2 · **Date:** 2026-10-20 · **Cohort:** Vietnam, 2,708 participants, 280
Xpert-confirmed TB · **Status:** exploratory (provisional split)

## What was previously claimed, and why it measured the wrong thing

Every earlier report of this gate quoted the same figure: **100% of lateral films and
non-medical images rejected, at a 0.2% in-distribution false-flag cost**. That is true, and
it is a statement about *adversarial inputs* — a photograph of a cat, a side-on chest film.

It had never been translated into the question that decides whether the gate belongs in a
screening workflow: **how many real tuberculosis patients does it turn away, and what does
the resulting human workload look like?**

## Definitions, so the numbers cannot be misread

| term | meaning |
|---|---|
| coverage | accepted / total |
| deferral rate | rejected / total — the gate's workload cost |
| **deferred** | sent for human review. **Not** a negative, **not** a caught case |
| conditional | computed only on accepted cases; never comparable to the ungated figure |
| referral | sent for further assessment. **Not** a diagnosis |

The operating threshold was refitted on the Vietnam **TUNE** half at 70% specificity. A
threshold carried over from TBX11K development produces an 89% referral rate, because the
MIL aggregator changes the score scale — a transfer failure worth recording in its own
right.

## Results

**Mode A — classifier alone:** sensitivity 0.6214, specificity 0.6890, 929 referrals.

**Mode B — gate first, rejections deferred:**

| | value |
|---|---|
| coverage | 0.8951 |
| deferral rate | 0.1049 |
| **TB cases deferred** | **21 / 280 = 7.50% [4.96%, 11.19%]** |
| non-TB cases deferred | 263 / 2,428 = 10.83% |
| conditional sensitivity (accepted only) | 0.6448 |
| conditional specificity (accepted only) | 0.6716 |

**Mode C — operational accounting, per 1,000 screened:**

| | per 1,000 |
|---|---|
| classifier referrals | 324 |
| deferred for human review | 105 |
| **total needing human time** | **429** |
| TB flagged by the classifier | 62 |
| TB deferred to review | 8 — found only if the human pathway works |
| TB missed outright | 34 — called not-TB, no further review |

## What the gate buys, and what it costs

**Denominators matter here and an earlier version of this table did not state them.** The
false-positive rows count non-TB cases only; the referral and workload rows count all cases.
Mixing the two produces a wrong net figure, so every row below names its population.

| quantity | population | without gate | with gate |
|---|---|---|---|
| referrals | all 2,708 | 929 | 878 |
| — of which true positives | 280 TB | 174 | **167** |
| — of which false positives | 2,428 non-TB | 755 | 711 |
| deferred to review | all 2,708 | 0 | 284 |
| — TB among deferred | 280 TB | 0 | **21** |
| — non-TB among deferred | 2,428 non-TB | 0 | 263 |
| **total cases needing human time** | all 2,708 | **929** | **1,162** |

Net workload = (878 + 284) − 929 = **+233 cases**. Computed on the all-case population
throughout; combining the non-TB false-positive counts with the all-case deferral count
would give 240 and would not be a meaningful quantity.

**The gate removes 44 false-positive referrals and adds 233 cases of human work, while
deferring 21 tuberculosis patients.** On this cohort it is not a safety feature; it is a cost.

**And it takes away correct referrals.** True-positive referrals fall from 174 to 167: of
the 21 deferred tuberculosis patients, **7 would have been correctly referred by the
classifier** and the gate intercepted them. The other 14 would have been missed anyway, so
for those the deferral is arguably neutral or mildly helpful — but only if the human pathway
finds them, which we have no outcome data for.

One mildly reassuring detail: TB cases are deferred at a **lower** rate than non-TB cases
(7.50% versus 10.83%), so the gate is not selectively discarding the sick. That is the
single thing it gets right here.

## Why this happens

The threshold was chosen to reject 100% of lateral films and non-medical images while barely
touching TBX11K. Both of those pools sit far from the training distribution — a photograph is
not a radiograph, and a lateral view is not a frontal one. A real screening cohort sits at an
intermediate distance: median OOD score 0.5312, against TBX11K's 0.3637 and the alien pools'
0.88. The threshold at 0.70 therefore clips the upper tail of a *legitimate* population.

This is the same failure already documented in `p2_vietnam_result.md` from the other
direction: the gate cannot distinguish "unfamiliar because it is not a chest X-ray" from
"unfamiliar because it is a chest X-ray from somewhere else". Here we can price that
confusion — 233 extra reviews and 21 deferred patients.

## Recommendation

**Do not deploy the gate in front of a screening population at this threshold.** Three
defensible options:

1. **Restrict its scope.** Keep it as an input-validity check against clearly alien inputs —
   wrong modality, wrong view, corrupt file — and set the threshold far higher so it never
   touches a plausible chest radiograph. It would reject the photographs and laterals it was
   built for and nothing else.
2. **Re-fit per site.** The threshold is site-specific; ~25 local normal radiographs are
   enough to re-fit it conformally (`p2_ood_gate_finding.md`). Then deferrals reflect that
   site's own distribution.
3. **Report both arms.** If the gate is kept, every result must be reported with and without
   it, with deferred TB cases stated explicitly — never folded into a specificity figure.

Option 1 is what the gate was actually validated for, and is what we recommend.

## What this audit cannot say

No end-to-end clinical sensitivity is estimated. A deferred case goes to a human pathway
whose outcomes we have no data for; whether those 21 patients are eventually diagnosed
depends entirely on that pathway. Calling deferrals "false positives prevented" would be
unsupported, and we do not.

## Caveats

- One cohort, provisional split, exploratory.
- The gate's threshold provenance is verified (`ood_threshold_rule` in the bundle) but was
  selected against two easy alien pools, so its metrics are plausibly optimistic for
  genuinely unusual clinical inputs.
- The classifier operating threshold was refitted on the TUNE half; the REPORT half was
  already read for the primary focal-pooling result, so these operational figures are
  computed on the full cohort and labelled exploratory.

Evidence: `artifacts/p2/external/gate_operational_audit.json`, via
`scripts/p2_35_gate_operational_audit.py`.
