# Claim boundaries

What we may say now, what is exploratory, and what we must not say. Every supported claim
below is traced to an artifact verified in `tables/claim_verification.json` (50 checks,
0 failures).

---

## 1. Supported now

**Measurement, on the Vietnam cohort (n = 2,708, 280 Xpert-positive, one image per
patient, all `pid` unique).**

| claim | value | source |
|---|---|---|
| our AUROC against negatives **without** self-reported previous TB | 0.7811 | `subgroup_metrics.csv` |
| our AUROC against negatives **with** self-reported previous TB | 0.6182 | same |
| **our degradation between the two** | **0.1629 [0.1372, 0.1890]** | `subgroup_difference_tests.csv` |
| Lunit 3.9.0.1 degradation, identical participants | 0.0978 [0.0770, 0.1196] | same |
| **difference in differences** | **+0.0651 [0.0399, 0.0900]** | same |
| our score separates previous-TB from other **negatives** | 0.6914 [0.6698, 0.7126] | same |

**Methodological claims that are safe.**

- The two subgroups **share all 280 positives**, so the interval for the difference comes
  from a stratified bootstrap reusing the same positive resample within each replicate
  (10,000 replicates, seed 20261021). An independent two-sample comparison would have
  been invalid.
- Vietnam is **one image per patient**, so patient-level and case-level resampling
  coincide. This is a property of this cohort, not a general one.
- Because AUROC is threshold-free, **a per-group threshold or recalibration cannot repair
  this gap.** That is a statement about what AUROC measures, not a novelty claim.
- Head training is **not bit-reproducible**: a refit of the same configuration gives
  scores differing by 0.85 logits on average, Spearman 0.998, with every AUROC within
  0.001. Rankings reproduce; the absolute score scale does not.

**Prior-art claims that are verified** (`LITERATURE_VERIFICATION_LOG.md` §1, two
independent full-text sources):

- Kagujje et al. 2022 report, at fixed thresholds, specificity of 24.0% with previous TB
  against 60.3% without for CAD4TB, and 22.2% against 61.8% for qXR.
- They report stratified AUROCs in Supplementary Table 2 and state in the main text that
  **CAD4TB discriminated better in participants without previous TB, while qXR performance
  was comparable between the two subpopulations.**
- Previous TB in that study was **self-reported and not record-verified**.

## 2. Exploratory — true of our data, not established as a finding

- **The subgroup was not prespecified.** It was selected after screening five clinical
  variables (`past_tb`, `dm`, symptom count, `contact`, `age_group`). The intervals are
  **not selection-corrected.**
- **Multiplicity is only roughly handled.** A Bonferroni correction across the five
  screened variables still leaves the bootstrap p below 0.0005, so the conclusion does not
  change — but the bootstrap p sits at the **1/10,000 resolution floor** of the procedure,
  six contrasts were reported from the selected variable, and **no selective-inference
  treatment was performed**. Present this as a rough check, never as a complete
  multiplicity treatment.
- **The cohort is exposed.** Vietnam informed the pooling configuration and its REPORT
  half has been scored in several runs. It is not independent validation.
- **The difference-in-differences is model- and comparator-specific.** It describes our
  prototype against one version of one commercial product on one cohort. **It requires
  independent replication.**
- Phase 1 diagnostic scores give 0.7811 / 0.6182; a Phase 2 refit of the same
  configuration gives 0.7822 / 0.6173. **Quote one phase or the other; never mix them
  inside a single contrast.**

## 3. Must not be claimed

| forbidden claim | why |
|---|---|
| that we discovered the previous-TB effect | Kagujje et al. published it in 2022, prespecified and peer-reviewed |
| that measuring AUROC is our contribution | they report stratified AUROCs too, in Supplementary Table 2 |
| that our result is the first ranking-level demonstration | **unverifiable** — Supplementary Table 2 was not retrieved after four attempts |
| that model-specific degradation is new | a 2020 five-algorithm evaluation reportedly found performance differed by prior-TB history across algorithms, and Kagujje et al. themselves found CAD4TB affected and qXR not. **A *Lancet Digital Health* 2024 paper may already report per-product prior-TB AUROCs — not retrieved** |
| that all commercial algorithms degrade alike | **directly contradicted**: qXR was comparable across subgroups in Kagujje et al. |
| that this is an AI-specific failure | **radiologists degrade too** — Qin et al. 2025 report significantly lower specificity in prior-TB participants for every reader group |
| that previous TB *causes* the degradation | observational association; history is self-reported and correlates with age, prior care-seeking, comorbidity and repeat imaging. No mediation or adjustment was done |
| that this subgroup explains the screening gap | our AUROC is 0.7811 **even without previous TB**, against the comparator's 0.8170 — a 0.036 deficit unrelated to this subgroup |
| that the prevalence difference explains anything | AUROC is invariant to class balance at fixed class-conditional distributions |
| any "first", "only" or "novel" framing | three searches and four retrievals is **not a systematic review**, and §4 of the log names a paper that could overturn the one distinctive-looking observation |
| that the system is clinically ready or validated | nothing prospective; nothing locked; no regulatory evaluation |
| that CIDRZ data access is likely or assured | no contact has been made; no agreement exists |

## 4. The defensible positioning, in one sentence

> Working from Kagujje et al.'s finding, we observed in an independent Vietnamese
> screening cohort that our research prototype's discrimination degrades markedly among
> self-reported previous-TB participants, and by more than a commercial comparator scored
> on the identical images — an exploratory, model-specific observation on an already
> exposed cohort that we would like to test independently.

Every hedge in that sentence is load-bearing.

## 5. Two caveats that cut in our favour, and must still be stated as caveats

- **Unknown histories.** `past_tb` has no unknown code, unlike `dm` which carries a `99`.
  If unknown histories were coded "no", the comparison stratum is contaminated and the
  **true gap would be larger**, not smaller. Direction knowable, size not. State it as a
  limitation, not as a reason the result is conservative.
- **Xpert Ultra trace results.** Persistent or trace positivity in previously treated
  patients would create false positives in the previous-TB stratum, pushing the measured
  gap in the **opposite** direction to what we observed. Whether Vietnam adjudicated trace
  results is not documented. This does not explain our finding but it does weaken the
  reference standard precisely where it matters.
