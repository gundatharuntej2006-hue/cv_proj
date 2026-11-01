# Prior-art verification — targeted check, 2026-10-21

**Recommendation: DO NOT SEND the current outreach package.** The reasoning is in §5.
This is a targeted prior-art check on three papers, not a complete literature review.

---

## 1. The finding that changes everything

**Our Vietnam cohort is the test library of a published paper, and that paper already
reports the previous-TB subgroup analysis for the same Lunit version we used as a
comparator.**

> Codlin AJ, Vo LNQ, Dao TP, Forse RJ, Dang HTM, Nguyen LH, Nguyen HB, Dinh LV,
> Sidney Annerstedt K, Lundin J, Lönnroth K. "Comparison of different Lunit INSIGHT CXR
> software versions when reading chest radiographs for tuberculosis." *PLOS Digital
> Health* 2025;4(4):e0000813. DOI 10.1371/journal.pdig.0000813. Received 8 Jul 2024,
> accepted 5 Mar 2025, published **15 Apr 2025**. **Peer-reviewed, open access, full text
> retrieved and read.**

### Cohort identity: confirmed

| | our local copy | Codlin et al. |
|---|---|---|
| Dryad DOI in the README | `10.5061/dryad.5tb2rbpcv` | `10.5061/dryad.5tb2rbpcv` |
| README title | *"Comparison of different Lunit INSIGHT CXR software versions…"* | identical |
| participants | **2,708** | 2,708 |
| Xpert Ultra positive | 280, **10.34%** | 10.3% |
| previous TB | **35.16%** | 35.2% |
| setting | — | Ho Chi Minh City; 127 mobile CXR screening events run by Friends for International TB Relief; March 2019 – March 2021 |
| scanners | 1 → 408, 2 → 1,332, 3 → 968 | three systems (JPI, DRTECH, Konica Minolta per the Dryad README) |

**The decisive check: all six published AUCs reproduce exactly from our local copy.**

| stratum | version | n | pos | recomputed | published (2 dp) | match |
|---|---|---|---|---|---|---|
| overall | v3.1.0.0 | 2,708 | 280 | 0.7633 | 0.76 | ✓ |
| overall | v3.9.0.1 | 2,708 | 280 | 0.7827 | 0.78 | ✓ |
| **past TB** | v3.1.0.0 | 952 | 102 | **0.6701** | **0.67** | ✓ |
| **past TB** | v3.9.0.1 | 952 | 102 | **0.7279** | **0.73** | ✓ |
| no past TB | v3.1.0.0 | 1,756 | 178 | 0.8082 | 0.81 | ✓ |
| no past TB | v3.9.0.1 | 1,756 | 178 | 0.8126 | 0.81 | ✓ |

Six of six at two decimal places. **Cohort identity is established beyond reasonable
doubt** — not by matching identifiers, which we cannot do, but by reproducing every
published statistic from the local data.

Their reported finding: the newer version *"performed significantly better among people
with a past history of TB (AUC 0.67 vs 0.73, p = 0.003)"*, while no-past-TB performance
was unchanged (0.81 vs 0.81, p = 0.644). Gains were *"concentrated among older males with
a history of TB"*.

## 2. A methodological discovery: we used a different construction

Reproducing their numbers revealed that **Codlin et al. stratify every participant by
`past_tb`**, so each stratum has its own positives and its own negatives. Our earlier run
instead compared **all 280 positives** against the negatives of one stratum.

| | construction A — full stratification (**published**) | construction B — shared positives (**ours**) |
|---|---|---|
| past-TB stratum | 952 (102 pos, 850 neg) | 1,130 (280 pos, 850 neg) |
| no-past-TB stratum | 1,756 (178 pos, 1,578 neg) | 1,858 (280 pos, 1,578 neg) |
| question answered | how well does the model work **in** people with past TB | how well does the model separate active TB **from** past-TB negatives |
| strata independent? | **yes** | no — positives shared |

Both are legitimate. A is conventional and is what the literature reports. B needs the
shared-positive bootstrap our earlier run correctly used.

## 3. Our result under the published construction

| | past TB | no past TB | degradation | 95% CI | |
|---|---|---|---|---|---|
| **our prototype** | 0.6392 [0.586, 0.692] | 0.7720 [0.740, 0.803] | **+0.1328** | [0.0718, 0.1951] | excludes 0 |
| **Lunit v3.9.0.1** | 0.7279 [0.673, 0.782] | 0.8126 [0.780, 0.844] | **+0.0847** | [0.0221, 0.1495] | excludes 0 |
| **difference in differences** | | | **+0.0481** | **[−0.0170, +0.1113]** | **INCLUDES 0** |

**The difference-in-differences does not survive the published construction.** Under our
shared-positive construction it was +0.0651 [0.0399, 0.0900]; under full stratification it
is +0.0481 [−0.0170, +0.1113].

Why: sharing the same 280 positives across both strata removes a source of variation that
is genuinely there, so intervals are narrower than the conventional analysis warrants for
a between-stratum contrast. The point estimate barely moves (+0.065 → +0.048); the
precision does.

**Our own degradation remains statistically supported under both constructions** (+0.1328
[0.0718, 0.1951] under A). What fails is the *comparative* claim — the one the outreach
package was built around.

## 4. The other two papers

### Qin et al. 2024 — per-product prior-TB AUCs exist, values not retrieved

> Qin ZZ, et al. "Computer-aided detection of tuberculosis from chest radiographs in a
> tuberculosis prevalence survey in South Africa: external validation and modelled impacts
> of commercially available artificial intelligence software." *Lancet Digital Health*
> 2024;6(9):e605–e613. DOI 10.1016/S2589-7500(24)00118-3. **Main text retrieved via PMC**
> (PMC11339183); publisher URL returned HTTP 403.

Case-control sample of **774** participants from the South African TB prevalence survey
(258 bacteriologically positive, 516 negative); **12 commercial CAD products**; **197
(26%) with prior TB** — 91/258 (35%) of cases, 106/516 (21%) of controls.

Verified: *"Several products performed worse in individuals with a history of tuberculosis
than in those without (p <0·0042, appendix pp 20–24)."* For qXR at threshold 0.5,
specificity fell from 87.3% [83.7–90.4] to **37.3% [28.5–47.7]** with prior TB, while
sensitivity rose — a thresholded result, not an AUC.

**Per-product prior-TB AUCs are in the supplementary appendix (`mmc1.pdf`, pp 20–24).
Retrieval failed** — PMC serves a "Preparing to download…" interstitial for the binary
PDF. Values **UNVERIFIED** and not estimated.

### Kagujje et al. 2022 — thresholded and discrimination results kept apart

> *Clin Infect Dis* 2022;76(3):e894–e901. DOI 10.1093/cid/ciac679. Full text retrieved
> from two independent sources.

**Thresholded** (verified, main text), at fixed thresholds for ~90% overall sensitivity:
CAD4TB specificity **24.0% (19.9–28.5)** with previous TB vs **60.3% (57.4–63.0)** without;
qXR **22.2% (18.2–26.6)** vs **61.8% (58.9–64.5)**. Sensitivity preserved in both.

**Discrimination** (verified qualitatively, values not retrieved): *"Both CAD systems
demonstrated an AUROC of at least 0.85 among all participants"*, and *"The performance of
the CAD4TB system was higher in individuals without previous TB compared with those with
previous TB, while the performance of qXR was comparable in the 2 subpopulations."*
Stratified AUROCs are in Supplementary Table 2; **four retrieval attempts failed**, so
their exact values remain **UNVERIFIED**.

1,884 analysed; 452 (24.0%) with previous TB, **self-reported, not record-verified**;
Xpert MTB/RIF reference.

## 5. Assessment and recommendation

### The general previous-TB effect is not new — explicitly rejected as a claim

Three peer-reviewed publications report it: Kagujje et al. 2022 (two commercial products,
Zambia), Qin et al. 2024 (12 products, South Africa), Codlin et al. 2025 (two versions of
one product, **our exact cohort**). Qin et al. 2025 (*Sci Rep*) further show human readers
degrade on the same axis. **Any claim of discovery would be false.**

### What our analysis adds, stated minimally

1. A **research prototype** measured on the same axis — the published work covers
   commercial products and human readers only. Our degradation is +0.1328
   [0.0718, 0.1951] under the published construction.
2. A **model-versus-comparator contrast on identical participants**. The point estimate
   (+0.048) suggests our prototype degrades more than Lunit v3.9.0.1, but **the interval
   includes zero**, so it is not supported.
3. The observation that **construction choice materially changes the precision** of a
   between-stratum contrast on this cohort.

Item 3 is methodological housekeeping. Items 1 and 2 do not amount to a research question
worth another group's time — and item 2, the premise of the outreach package, **fails**.

### Limitations that compound

- **Post-hoc subgroup selection** — chosen after screening five clinical variables; not
  selection-corrected.
- **Prior cohort exposure** — this dataset informed our pooling configuration, and it is
  now also known to be a published test library.
- **Reference-standard uncertainty** — Xpert Ultra trace positivity in previously treated
  patients is unadjudicated in the released data; previous TB is self-reported.
- **No independent replication** — one cohort, one comparator version.
- **The DiD is not supported** under the conventional construction.

### Recommendation: **DO NOT SEND the current outreach package**

Both the email and the concept note rest on two premises that no longer hold: that the
previous-TB subgroup question on this cohort is open (**it is published, by the group that
produced the data**), and that our comparative observation is statistically supported
(**it is not, under their construction**). Sending it would ask a research group to help
validate a finding that is already in the literature, on the strength of a contrast whose
interval spans zero.

**What a revised approach would require** — a rewrite, not a patch:

- The past-TB framing is spent. The remaining defensible project finding is the
  **curated-benchmark-to-screening gap** (TBX11K 0.9368 → Vietnam 0.7242 under one frozen
  configuration), which is a different question and is not CIDRZ-specific.
- If anyone is approached, the honest ask is **methodological feedback on the gap work**,
  with the past-TB analysis presented as a replication of their own finding — not as a
  result needing validation.
- **Codlin et al. and the FIT group in Ho Chi Minh City are the more natural contacts**
  for anything concerning this cohort, since it is theirs.

### One thing worth doing regardless

The project's documentation should record that **the Vietnam cohort is the published test
library of Codlin et al. 2025**, and cite that paper wherever the cohort is described.
That is currently missing everywhere and is a straightforward provenance gap.
