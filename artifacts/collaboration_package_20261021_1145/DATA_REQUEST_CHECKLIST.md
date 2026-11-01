# Data request checklist

For the replication described in `ONE_PAGE_CONCEPT_NOTE.md`. Every line is a requirement
or a question to settle — **none of it assumes any data will be made available.**

---

## 1. Cohort eligibility and design

| item | requirement | hard / soft |
|---|---|---|
| recruitment | consecutive, or a documented sampling frame, with a participant flow diagram and stated eligibility criteria | **hard** |
| setting | active case finding or routine screening. A facility-based symptomatic-only sample is acceptable **if documented as such** | **hard** |
| independence | not previously used to develop or tune our model — trivially satisfied, but record it | **hard** |
| analysable size | **2,500–3,500**, enrol ~3,500–4,500 allowing 15–25% attrition | **hard** |
| previous-TB fraction | **≥ 20% of negatives.** Below 15% the requirement rises past 3,400 analysable | **hard** |
| prevalence | recorded, not controlled. Below 5%, the operating-point endpoint must be demoted to secondary | soft |
| site diversity | more than one institution, or a different vendor/country from Vietnam, is worth more than extra volume | soft |

## 2. Reference standard

| item | requirement |
|---|---|
| modality | microbiological — Xpert Ultra or culture |
| coverage | **ideally all enrolled participants, independent of the AI output and of symptoms** |
| if partial | the triggering rule must be documented, and one of: a randomly sampled, fully tested screen-negative stratum with the fraction prespecified; or 6–12 month follow-up for incident TB. Otherwise sensitivity is reported as an upper bound, **in the abstract** |
| **trace adjudication** | **the rule for Xpert Ultra "trace" results must be stated and prespecified.** Persistent or trace positivity in previously treated patients affects precisely the subgroup under study |
| timing | interval between radiograph and specimen collection |

**Why coverage matters more than usual here:** if testing is triggered by an abnormal
radiograph, then in the previous-TB stratum — where films are abnormal for reasons other
than active disease — the triggering is differential *by subgroup*. That biases the
subgroup contrast itself, not merely the absolute estimates.

## 3. Previous-TB history

| item | requirement |
|---|---|
| presence | recorded for every participant | **hard** |
| **unknown category** | **an explicit "unknown/refused" code is required.** The Vietnam file has none, so it cannot be determined whether unknown histories were coded "no" | **hard** |
| ascertainment | self-report, clinic record, national register, or combination — **recorded per participant** | **hard** |
| verification | record-verified where possible; Kagujje et al. and the Vietnam cohort are both self-report only | soft but valuable |
| detail | date of previous treatment, completion/outcome, number of prior episodes | soft |

## 4. Images

De-identified **frontal** radiographs, DICOM preferred at original resolution; view
recorded (PA / AP / lateral); acquisition date; image-quality flags, with failures
**reported rather than silently dropped**; de-identification of both pixel data
(burnt-in text) and DICOM headers.

## 5. Identifiers

A **pseudonymous, stable participant identifier** sufficient for de-duplication and
patient-level partitioning, held only inside the agreed secure environment. No name, no
national ID, no date of birth beyond an age band. If a participant can contribute more
than one radiograph, that must be flagged so resampling is done by patient.

## 6. Metadata

| category | fields |
|---|---|
| acquisition | scanner make/model, site, protocol, software version |
| clinical | symptoms (cough, fever, weight loss, night sweats), TB contact, diabetes, HIV status where lawful |
| severity | smear grade or semi-quantitative Xpert — **the one variable no cohort we hold has** |
| demographic | age band and sex, where ethically and legally appropriate |
| dates | enrolment, imaging, specimen, result |
| comparator | any existing commercial CAD score already computed on the same images — valuable, not essential |

## 7. Exclusions and missingness

A documented exclusion log with reasons and counts; missingness per field; flags for
unreadable images and processing failures. **Missingness must be reported, not imputed
away**, and the previous-TB unknown stratum reported separately.

## 8. Consent, ethics and permitted use

- [ ] ethics / IRB approval covering **secondary AI analysis**, or a documented exemption
- [ ] consent scope — does it permit algorithm evaluation by a third party?
- [ ] data-sharing agreement, with publication and authorship terms agreed **in advance**
- [ ] allowed-use restrictions: evaluation only, no model training, no redistribution,
      retention period
- [ ] national or programme-level approvals beyond the research group
- [ ] storage location, encryption, access control, audit logging
- [ ] **explicit commitment that no patient-level data leaves the agreed environment and
      none is sent to any external service**
- [ ] named local clinical oversight

## 9. Preferred working arrangement

**Analysis runs inside the data holder's environment.** Measured requirement: peak 0.94 GB
VRAM, minutes of compute, no internet access needed at run time. We would supply a locked,
hashed model and scoring script; the output leaving the environment would be aggregate
metrics only.

This removes the main objection to sharing imaging data and should be offered first.

## 10. Minimum viable request

> Per participant: a de-identified frontal chest radiograph; the microbiological result;
> previous-TB history with an explicit unknown category and the ascertainment method; and
> a pseudonymous ID. Approximately 3,000 participants.

## 11. Known shortfalls to raise honestly

- The Kagujje et al. published cohort is **1,884**, below the 2,139 our calculation
  requires at a 24% previous-TB fraction. It would be powered only for a larger effect, or
  would need pooling with another site — **say this before they do.**
- No public dataset meeting these requirements was found in our searches; the Vietnam
  Dryad cohort may be the only open one of its kind.

## 12. Alternative sources, if this approach does not proceed

National TB programme ACF datasets; a prospective collaboration with a screening
programme; TB Portals (NIAID, registration-gated, likely treatment rather than screening
cohorts — unverified); carceral ACF studies (atypical population, image release
unestablished). **Do not approach several groups in parallel with near-identical
messages.**
