# The next action

## Recommendation: **stop the concept-intervention line. Do not run another CLEAR experiment.**

The question the concept audits were built to answer — *does the classifier lean on a
chronic-appearance shortcut?* — cannot be answered with this instrument, and that is now
demonstrated rather than suspected.

### Why stopping is the evidence-based choice

**The instrument does not measure what its column names say.** An anchor and its explicit
negation sit at cosine +0.75 to +0.98 in text space, and on the image side `no scarring`
correlates with the classifier score at +0.301 while `scarring` correlates at +0.566 —
same sign. A measurement that cannot separate "scarring" from "no scarring" cannot support
an intervention aimed at scarring. No amount of extra TUNE inference fixes that, because
it is a property of the pinned encoder.

**The one remaining technical fix is unavailable.** The obvious repair is a contrastive
score — `sim(X) − sim(no X)` — which would recover polarity. It is not available here:
exact negated entries exist for only 5 of the 9 primary concepts, and building the missing
ones means inventing phrases, which the rules correctly forbid and which would in any case
be concept selection after seeing results.

**The scientific question has shifted.** The useful finding from this run is not about TB.
It is that **CLEAR concept scores are negation-blind on chest radiographs**, with a
worked example (consolidation) of how corpus composition then flips the sign. That is a
methodological result about the tool, and it is reportable on its own.

### What is NOT recommended, and why

| | |
|---|---|
| another CLEAR run on TUNE | the instrument is the problem; more measurements with it add nothing |
| a contrastive `sim(X) − sim(no X)` concept set | requires inventing phrases for 4 of 9 concepts; post-hoc concept construction |
| any concept-informed adjustment, subtraction or recalibration | forbidden, and now additionally unjustified — there is no validated axis to adjust against |
| opening Vietnam REPORT | nothing here is a hypothesis worth spending it on |
| opening TBX11K `internal_test` | remains sealed; unchanged |
| adopting B1 | the localisation finding is not a safety finding; the two-patient sensitivity loss is unchanged and unresolvable |
| a RayDINO benchmark | out of scope here and still licence-blocked |

### The single next action, if the project continues this line at all

**Correct the existing write-ups, then stop.** Specifically: in
`report/paper_draft.md` and the concept sections of the prior run reports, replace every
reading of a concept *name* as a radiological finding with the wording in
`LIMITATIONS_AND_CLAIM_BOUNDARIES.md` §4, and state the negation result as a limitation of
the method. The arithmetic in those reports is correct and reproduces exactly; only the
interpretation needs repair.

That is a documentation task, not an experiment. It costs no data, touches no reserved
partition, and removes a claim that a reviewer would otherwise find.

### Items this run closes and leaves open

| item | status |
|---|---|
| **U3 — consolidation anomaly** | **closed.** Explained: negation-blind anchor in a normality-dominated corpus region. See `FINAL_REPORT.md` |
| **U2 — concept collinearity** | **closed** by the previous run and re-measured here: text-side cosines up to +0.875, image-side up to +0.797 |
| "broad re-ranking" reading of B0 -> B1 | **corrected.** The change is localised to the scarring neighbourhood (Spearman −0.824) |
| `degenerative changes` as a negative control | **invalidated.** Cosine 0.730 to scarring — inside the region that moved |
| **U4 — CLEAR's exposure to this cohort** | **open**, and not resolvable from any repository artifact |
| cavity never measured | **open.** No exact vocabulary entry; deliberately not approximated |
| B1 positive-case safety | **open and unresolvable here** — two patients of 139 |

### Honest framing of this run's own result

The anomaly is explained, and the explanation disqualifies the measurement rather than
rescuing it. That is a useful outcome: it prevents an intervention that would have been
built on an axis nobody could interpret. It is not a result about tuberculosis, and it
should not be written up as one.
