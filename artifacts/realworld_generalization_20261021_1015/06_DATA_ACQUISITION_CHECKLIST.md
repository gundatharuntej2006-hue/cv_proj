# Phase 3 — Data acquisition checklist

For an academic supervisor or a prospective data partner. Everything below is a request or
a check, not a claim that any source is available.

---

## 0. The one-paragraph case, for an email

> We have built and audited a chest-radiograph TB triage system on public data. On curated
> case-control benchmarks it reaches AUROC 0.95–0.97. On the one consecutive screening
> cohort we could obtain (Vietnam, n = 2,708, Xpert Ultra reference, publicly released on
> Dryad) the same system reaches 0.7242, and a certified commercial product reaches 0.7827
> on the identical images. Within that cohort, discrimination falls to 0.6182 among
> Xpert-negative participants with a history of previous TB, who make up 35% of the
> negatives — replicating Kagujje et al., *Clin Infect Dis* 2022. We have exhausted what
> one screening cohort can tell us and are seeking a second, from a different country and
> a different radiography vendor, to establish whether these findings generalise. We need
> images plus outcomes, not a trained model, and we are glad to work under any data-use
> agreement, including running the analysis inside your own environment.

## 1. Search status from this run

**No public dataset matching the requirements was found.** A web search for consecutive or
active-case-finding cohorts with a universal Xpert reference standard and released images
returned no qualifying open dataset. **The Vietnam Dryad cohort may be the only public one
of its kind**, which is itself worth stating in the paper.

Candidates and their verification status: `tables/data_source_shortlist.csv`. Searches were
run on 2026-10-21, were not exhaustive, and should be repeated.

## 2. Highest-value targets, in order

1. **Kagujje / Muyoyeta group, CIDRZ, Lusaka, Zambia.** They published the past-TB
   specificity finding (*Clin Infect Dis* 2022;76(3):e894–e901,
   DOI 10.1093/cid/ciac679) on 1,884 participants of whom 452 had previous TB, with an
   Xpert reference standard. They have already analysed this exact question with
   commercial CAD. **Images were not publicly released**, so this needs a direct approach —
   and it is the single best-matched target.
2. **National TB programme active-case-finding datasets.** Highest volume, longest lead
   time; requires ethics approval and a national data-sharing agreement.
3. **Prospective collaboration with an ongoing screening programme.** Best scientific
   design; needs a named clinical partner and typically 6–18 months.
4. **TB Portals (NIAID)**, tbportals.niaid.nih.gov. Registration-gated. Prior expectation
   is treatment and drug-resistance cohorts rather than screening, so verify the enrolment
   design before investing effort. **Not verified in this run.**
5. **Carceral ACF studies**, e.g. the Brazilian ultra-portable X-ray study. The design
   reportedly fits — all participants imaged irrespective of symptoms — but the population
   is atypical and image release is unestablished.

## 3. Per-candidate checklist, to complete BEFORE requesting access

For each source, record and cite:

- [ ] enrolment design: consecutive, symptom-triggered, or curated
- [ ] eligibility criteria and participant-flow numbers
- [ ] reference standard, and **whether it was applied to all participants**
- [ ] prevalence, n, n positive
- [ ] proportion with previous TB, and whether self-reported or record-verified
- [ ] patient identifiers available for de-duplication and patient-level splits
- [ ] acquisition metadata: device, site, view, date
- [ ] images: modality, format, resolution, and whether released at all
- [ ] licence, consent scope, and whether secondary AI use is covered
- [ ] whether the cohort could plausibly have been seen by RAD-DINO's checkpoint selection
      — for public cohorts the honest answer is usually **unknown**, and must be written
      as such
- [ ] size against `05_SAMPLE_SIZE_AND_POWER.md`: **reject cohorts too small to answer the
      question rather than running an underpowered study**

## 4. Governance, to settle before any transfer

- [ ] ethics / IRB approval covering secondary AI analysis, or a documented exemption
- [ ] data-sharing agreement, including publication and authorship terms
- [ ] de-identification standard for images (DICOM header scrubbing, burnt-in text)
- [ ] storage location, encryption, retention period
- [ ] whether analysis must run inside the partner's environment — **offer this
      proactively; it removes most objections, and this pipeline runs on a single laptop
      GPU in minutes (peak 0.94 GB VRAM, seven minutes for four arms × five seeds)**
- [ ] an explicit statement that no patient-level data will be sent to any external service

## 5. Minimum viable request

If a full cohort is unobtainable, the following still answers question 2 of the protocol
and is a much smaller ask:

> Per participant: a de-identified frontal chest radiograph, the microbiological result,
> previous-TB history, and a pseudonymous ID. Roughly 2,500 participants. No demographics
> beyond age band and sex are required.

## 6. What not to do while waiting

- **Do not run another architecture sweep.** Attention-MIL was measured and is worse, the
  pooling sweep is complete, subspace removal was rejected, and more TBX11K data is flat.
- Do not train on any evaluation cohort to manufacture the missing source labels.
- Do not download restricted patient data automatically, or upload any patient-level data
  to an external service.
- Do not spend TBX11K `internal_test` on an incidental question — it is the only unexposed
  partition left.
