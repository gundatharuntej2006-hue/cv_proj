# Claim boundaries — revised after the targeted prior-art check

**Supersedes** `collaboration_package_20261021_1145/CLAIM_BOUNDARIES.md`, which was written
before Codlin et al. 2025 was identified. That file is left unmodified as the record of
what was believed then. **Two of its "supported now" claims move to "must not be claimed".**

---

## 1. What changed

| | before | after |
|---|---|---|
| Is the previous-TB subgroup question on this cohort open? | assumed open | **no — published** by Codlin et al. 2025, the paper that released this dataset |
| Is the model-vs-comparator difference supported? | yes, +0.0651 [0.0399, 0.0900] | **no — under the published construction it is +0.0481 [−0.0170, +0.1113]**, which includes zero |
| Is our cohort independent of the literature? | treated as an exposed-but-ours dataset | **it is a published benchmark test library** |

## 2. Still supported

**Our prototype's degradation, under the conventional (published) construction:**

| | past TB (n=952, 102 pos) | no past TB (n=1,756, 178 pos) | degradation |
|---|---|---|---|
| our prototype | 0.6392 [0.586, 0.692] | 0.7720 [0.740, 0.803] | **+0.1328 [0.0718, 0.1951]** |

The interval excludes zero under **both** constructions, so the statement *"our prototype
discriminates worse among people with previous TB"* survives. It is a replication of an
established effect in a new system, on an exposed and published cohort, with a
non-prespecified subgroup.

**Still supported, unchanged:**

- AUROC is threshold-free, so this gap cannot be repaired by recalibration or a
  subgroup-specific threshold.
- Head training is not bit-reproducible: refits differ by ~0.85 logits on average,
  Spearman 0.998, AUROCs within 0.001.
- All six Codlin et al. AUCs reproduce from our local copy at two decimal places, which
  **confirms cohort identity**.
- The two constructions answer different questions and are not interchangeable.

## 3. Now exploratory, and weaker than before

- **The model-versus-comparator contrast.** Point estimate +0.0481 under the published
  construction; **interval includes zero**. Describe it as *"a point estimate consistent
  with our prototype degrading more than the commercial comparator, not statistically
  supported"* — never as a finding.
- **Anything from construction B** (shared positives, our earlier run). It is a valid
  statistic answering a different question, but it is not what the literature reports, and
  its narrower intervals must not be quoted alongside published figures as though
  comparable.

## 4. Must not be claimed

Additions to the previous list are marked **NEW**.

| forbidden claim | why |
|---|---|
| that we discovered the previous-TB effect | Kagujje 2022, Qin 2024, Codlin 2025 |
| that measuring AUROC is our contribution | all three report AUROC |
| **NEW: that the previous-TB question is open on this cohort** | **Codlin et al. 2025 published it on exactly these 2,708 participants with exactly our comparator version** |
| **NEW: that our difference-in-differences is a finding** | **+0.0481 [−0.0170, +0.1113] under the published construction — includes zero** |
| **NEW: that +0.0651 [0.0399, 0.0900] can be quoted without its construction** | **it comes from the shared-positive construction, which is not what the literature uses and which narrows the between-stratum interval** |
| that model-specific degradation is new | Kagujje found CAD4TB affected and qXR not; Qin 2024 found "several products" worse; Codlin found version-to-version differences |
| that all algorithms degrade alike | contradicted directly |
| that this is AI-specific | radiologists degrade too (Qin 2025) |
| that previous TB *causes* the degradation | observational; self-reported history; no adjustment |
| that this subgroup explains the screening gap | 0.7720 even without previous TB, against the comparator's 0.8126 |
| any "first", "only" or "novel" framing | three papers found in a **targeted** check already cover it |
| that this is a complete literature review | three papers, five searches, several failed supplement retrievals |
| that the system is clinically ready | nothing prospective |
| **NEW: that CIDRZ is the natural collaborator for this question** | **the cohort belongs to Codlin et al. and the FIT group in Ho Chi Minh City** |

## 5. Unverified, and not to be characterised

| item | status |
|---|---|
| Kagujje et al. Supplementary Table 2 stratified AUROCs | **NOT RETRIEVED** — 4 attempts |
| Qin et al. 2024 appendix pp 20–24, per-product prior-TB AUCs | **NOT RETRIEVED** — PMC interstitial on `mmc1.pdf` |
| whether any paper reports a prototype-vs-commercial prior-TB DiD | **UNKNOWN** — not systematically searched |
| Xpert Ultra trace adjudication in the Vietnam cohort | **UNKNOWN** |
| exact enrolment/verification completeness in the Vietnam cohort | **UNKNOWN** |

## 6. The only defensible positioning left

> On the publicly released Ho Chi Minh City screening cohort of Codlin et al. (*PLOS
> Digital Health* 2025), whose previous-TB subgroup analysis for Lunit INSIGHT CXR is
> already published, we measured a research prototype on the same axis and observed a
> degradation of 0.133 (95% CI 0.072–0.195) in people with self-reported previous TB. This
> replicates, in a research system, an effect established for commercial products
> (Kagujje 2022, Qin 2024) and for human readers (Qin 2025). A comparison suggesting our
> prototype degrades more than the commercial comparator was not statistically supported
> (+0.048, 95% CI −0.017 to +0.111). The subgroup was not pre-specified and the cohort had
> already informed our configuration choices.

Every clause is load-bearing. **This is a replication with a null comparative result** —
publishable as part of a larger honest evaluation, not as a standalone contribution.

## 7. Provenance gap to fix

The project's documentation describes the Vietnam cohort by its Dryad DOI but **never
cites Codlin et al. 2025**, which is its source publication. Every description of the
cohort should cite it. This is a straightforward and currently unmet obligation.
