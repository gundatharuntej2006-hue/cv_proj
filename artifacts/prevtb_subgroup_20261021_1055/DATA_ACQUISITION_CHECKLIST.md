# Data acquisition checklist — previous-TB validation cohort

Scoped to the study in `NEXT_STUDY_PROTOCOL.md`. Everything here is a requirement or a
check, not a claim that any source is available.

---

## 1. Minimum cohort design

| requirement | specification | hard or soft |
|---|---|---|
| recruitment | consecutive, or a documented sampling frame with a participant flow diagram | **hard** |
| size | **2,500–3,500 analysable**, enrol ~3,500–4,500 for 15–25% attrition (derivation: protocol §5) | **hard** |
| previous-TB history | recorded for every participant, with an explicit **unknown** category and the ascertainment method stated | **hard** |
| previous-TB fraction | ≥ 20% of negatives; below 15% the study needs 3,400+ analysable | **hard** |
| reference standard | microbiological, ideally for all participants regardless of AI output or symptoms | **hard** |
| prevalence | recorded, not controlled; below 5% the operating-point co-primary must be demoted | soft |
| patient identifiers | pseudonymous and stable, so de-duplication is verifiable | **hard** |
| images | frontal, DICOM preferred at original resolution | **hard** |
| image view | recorded (PA / AP / lateral) | **hard** |
| scanner / vendor / site | recorded where permitted | soft |
| image quality flags | recorded, and failures reported rather than dropped | soft |
| symptoms and risk factors | cough, fever, weight loss, night sweats, contact, diabetes, HIV where lawful | soft |
| severity | smear grade or semi-quantitative Xpert | soft — but it is the one variable no held cohort has |
| age band and sex | where ethically and legally appropriate | soft |
| comparator scores | any commercial CAD score already computed on the same images | soft — enables the difference-in-differences secondary |

## 2. Candidate sources, in order

1. **Kagujje / Kerkhoff / Muyoyeta, CIDRZ, Lusaka.** Published the finding; 1,884
   analysed with 452 previous-TB participants and an Xpert reference. **Images were not
   publicly released**, so this requires a direct approach — see
   `COLLABORATION_OUTREACH_DRAFT.md`. Note their cohort is **below** the §1 size
   requirement for a 0.05 gap at a 24% previous-TB fraction (2,139 needed), so it would
   be powered only for a larger effect, or would need pooling with another site.
2. **National TB programme active case-finding datasets.** Largest, slowest; ethics
   approval plus a national data-sharing agreement.
3. **A prospective collaboration with an ongoing screening programme.** Best design;
   6–18 months.
4. **TB Portals (NIAID).** Registration-gated. Prior expectation is treatment and
   drug-resistance cohorts rather than screening. **Not verified.**
5. **Carceral ACF studies.** Design reportedly fits — all participants imaged — but the
   population is atypical and image release is unestablished. **Not verified.**

A search in the preceding run found **no public dataset meeting these requirements**. The
Vietnam Dryad cohort may be the only open one of its kind.

## 3. Per-candidate verification, BEFORE requesting access

- [ ] enrolment design: consecutive, symptom-triggered, or curated
- [ ] eligibility criteria and participant-flow numbers
- [ ] reference standard, and **whether applied to all participants**
- [ ] **Xpert "trace" adjudication rule** — this specifically affects the previous-TB
      stratum
- [ ] prevalence, n, n positive
- [ ] previous-TB proportion, and self-reported vs record-verified
- [ ] whether an **unknown** history category exists
- [ ] patient identifiers for de-duplication
- [ ] acquisition metadata
- [ ] image format, resolution, and whether released at all
- [ ] licence, consent scope, secondary-AI-use coverage
- [ ] whether the cohort could have been seen by RAD-DINO checkpoint selection — usually
      **unknown** for public cohorts, and must be written as such
- [ ] size against protocol §5: **reject cohorts too small rather than running an
      underpowered study**

## 4. Governance to settle before any transfer

- [ ] ethics / IRB approval covering secondary AI analysis, or documented exemption
- [ ] data-sharing agreement, with publication and authorship terms
- [ ] de-identification standard: DICOM headers and burnt-in text
- [ ] storage location, encryption, retention period
- [ ] **offer analysis inside the partner environment** — measured requirement is under
      1 GB VRAM and minutes of compute
- [ ] explicit commitment that no patient-level data leaves the agreed environment and
      none is sent to any external service
- [ ] local clinical oversight named

## 5. Minimum viable request

> Per participant: a de-identified frontal chest radiograph, the microbiological result,
> previous-TB history with an unknown category, and a pseudonymous ID. Approximately
> 3,000 participants. No demographics beyond age band and sex are required. Any existing
> commercial CAD score on the same images would be valuable but is not essential.

## 6. What not to do

- Do not contact several groups with near-identical messages simultaneously.
- Do not download restricted patient data automatically.
- Do not upload any patient-level data to an external service.
- Do not begin another model experiment while waiting — the preceding run's go/no-go
  stands.
- Do not spend TBX11K `internal_test` on this; it is unrelated and still unread.
