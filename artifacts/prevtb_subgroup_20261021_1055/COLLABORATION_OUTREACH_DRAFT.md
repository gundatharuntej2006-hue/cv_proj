# Draft outreach — for user review only

**NOT SENT. No message has been or will be sent from this session.** No email address has
been looked up or invented; the user should obtain the correspondence address from the
published paper and verify who is the right contact.

**Before sending, please check:** the claims below against
`PREVIOUS_TB_SUBGROUP_REPORT.md`; that you are comfortable sharing the figures at this
stage; and that your institution's norms on unsolicited collaboration requests are met.
Consider having a supervisor co-sign or send it.

---

## Suggested subject

> Independent replication of the previous-TB CAD finding — possible collaboration?

## Draft

> Dear Dr Kagujje and colleagues,
>
> I am a final-year undergraduate working on chest-radiograph tuberculosis triage, and I
> am writing after reading your 2022 *Clinical Infectious Diseases* paper on CAD
> performance among people with a history of previous TB.
>
> We have been evaluating a research pipeline built on a frozen chest-radiograph
> foundation model. On curated public case-control benchmarks it reaches AUROC 0.95–0.97.
> On the one consecutive screening cohort we could obtain — the Vietnamese active
> case-finding dataset published on Dryad (n = 2,708, Xpert Ultra reference) — it reaches
> 0.724, while Lunit INSIGHT CXR scored on the identical images reaches 0.783.
>
> Prompted by your paper, we stratified that cohort by self-reported previous TB, which
> 35% of the Xpert-negative participants reported. Our system's AUROC was 0.781 against
> negatives without previous TB and 0.618 against those with it — a gap of 0.163
> (95% CI 0.137–0.189). The commercial comparator degraded on the same axis but less
> (0.817 to 0.719), and the difference between the two degradations was 0.065
> (95% CI 0.040–0.090).
>
> We want to be clear about what this is and is not. The phenomenon is yours: you reported
> it first, prespecified, and your finding that it was product-dependent — CAD4TB
> affected, qXR not — is the more informative result. Our subgroup analysis was
> exploratory rather than prespecified, it is a single cohort that had already informed
> earlier configuration choices in our work, and we make no claim that our approach is
> novel or validated.
>
> That is why we are writing. We would like to test whether the effect reproduces
> independently, and we are looking for a second, appropriately governed screening cohort
> with previous-TB history recorded. We have written a pre-registered protocol with a
> fixed primary endpoint, a minimum effect size set in advance, and a sample-size
> calculation (roughly 2,500–3,500 participants under our assumptions).
>
> If such a collaboration is of interest, we would be glad to work in whichever way suits
> you — including running the analysis entirely inside your own environment, since the
> pipeline needs under 1 GB of GPU memory and a few minutes of compute. We would not need
> to hold or move any patient-level data. We would of course expect any output to be
> authored jointly and governed by whatever agreement you consider appropriate, and we
> would welcome your view on the protocol regardless of whether data access is possible.
>
> We recognise this is an unsolicited request and that data sharing carries real
> obligations. A simple "not possible" is an entirely acceptable answer, and we would
> still be grateful for any pointer toward a cohort or group better placed to help.
>
> With thanks for your work and for your time,
>
> [Name]
> [Institution, programme]
> [Supervisor name and contact, if co-signing]

---

## Notes on the drafting

- **No entitlement.** The request is framed as a question, the refusal is pre-accepted,
  and an alternative (comment on the protocol) is offered that costs them little.
- **No novelty claim.** Their priority is stated plainly, including that their
  product-dependent finding is the more informative one.
- **Limitations disclosed up front** — exploratory, single cohort, previously exposed —
  rather than buried. A group that works on this will spot all three immediately, and
  naming them first is the only way to be taken seriously.
- **The offer to compute inside their environment** removes the most common objection to
  sharing imaging data and is true: peak VRAM measured at 0.94 GB.
- Figures quoted: 0.724, 0.783, 35%, 0.781, 0.618, 0.163 [0.137, 0.189], 0.817, 0.719,
  0.065 [0.040, 0.090] — all verified in `tables/subgroup_metrics.csv` and
  `tables/subgroup_difference_tests.csv`.

## Alternative recipients, if this one does not proceed

Named in `DATA_ACQUISITION_CHECKLIST.md` §2. Do not send a near-identical message to
several groups at once; if a second approach is made, say that a first was made and what
came of it.
