# Outreach email — DRAFT, NOT SENT

**No message has been sent. No email address has been looked up, inferred or invented.**
Obtain the corresponding author's address from the published paper, and confirm who the
appropriate contact is before writing.

**Before sending, check:** every figure against `CLAIM_BOUNDARIES.md`; that your
supervisor has seen it and ideally co-signs; and your institution's norms on unsolicited
collaboration requests. Attach `ONE_PAGE_CONCEPT_NOTE.md` rather than lengthening the
email.

**Do not attach or paste** patient-level data, images, file paths from this repository, or
per-case tables.

---

## Subject

> Previous-TB subgroup performance in a research CXR model — request for methodological feedback

---

## Draft

> Dear Dr Kagujje and colleagues,
>
> I am an undergraduate student working on chest-radiograph tuberculosis triage, writing
> after your 2022 *Clinical Infectious Diseases* paper on CAD performance among people
> with a history of previous TB. I am writing for methodological feedback rather than to
> request data, though I mention below why a collaboration might be of mutual interest.
>
> We have built a research prototype on a frozen chest-radiograph foundation model with a
> lightweight trained read-out. Prompted by your paper, we stratified the publicly
> released Vietnamese active case-finding cohort (Dryad, n = 2,708, Xpert Ultra reference,
> 280 positives) by self-reported previous TB, which 35% of the Xpert-negative
> participants reported.
>
> Our prototype's AUROC was 0.781 against negatives without previous TB and 0.618 against
> those with it — a degradation of 0.163 (95% CI 0.137–0.189). Lunit INSIGHT CXR 3.9.0.1,
> scored on the identical images, degraded less: 0.817 to 0.719, a difference of 0.098
> (0.077–0.120). The difference between those two degradations was 0.065 (0.040–0.090).
> Intervals come from a stratified bootstrap that reuses the same positive resample in
> each replicate, since both subgroups share all 280 positives.
>
> I want to be clear about what this is. The phenomenon is yours — you reported it first
> and prespecified, and your finding that CAD4TB and qXR behaved *differently* is the more
> informative result, since it shows the effect is not inevitable. Our subgroup analysis
> was exploratory rather than prespecified, it rests on a single cohort that had already
> informed earlier choices in our work, and we have not been able to retrieve your
> Supplementary Table 2, so we cannot tell whether the model-to-model contrast we observe
> is already covered by your stratified AUROCs or by the Qin et al. *Lancet Digital Health*
> analysis. We make no claim to novelty.
>
> Two things would be genuinely valuable to us, in increasing order of imposition.
>
> First, your view on whether the comparison is sound — in particular the shared-positive
> bootstrap, and whether a model-versus-comparator difference of this kind is informative
> at all given that reader performance also varies by previous-TB status.
>
> Second, if it were ever feasible, whether an independently governed screening cohort
> might allow this to be tested properly. We have written a pre-registered protocol with a
> fixed primary endpoint, a minimum effect size set in advance, and a sample-size
> calculation. On our assumptions it needs roughly 2,500–3,500 analysable participants
> with previous-TB history recorded, so we recognise your published cohort of 1,884 would
> be below that for the effect size we specified, and that any such cohort would involve
> permissions well beyond a single research group's gift.
>
> If anything did prove possible, we would be glad to work in whichever way suits you,
> including running the analysis entirely inside your own environment — the pipeline needs
> under 1 GB of GPU memory and a few minutes of compute, so no patient-level data would
> need to move. Any output would of course be jointly authored and governed by whatever
> agreement you consider appropriate.
>
> I appreciate that this is unsolicited and that requests of this kind carry real
> obligations. A brief "not possible" is an entirely acceptable reply, and feedback on the
> protocol alone would be worth a great deal to us.
>
> With thanks for your work and for your time,
>
> [Name]
> [Institution and programme]
> [Supervisor name and contact, if co-signing]
>
> *Attached: one-page concept note.*

---

## Notes on the drafting

- **The ask is feedback, not data.** The data question is secondary, hedged, and
  explicitly acknowledged as beyond one group's gift.
- **Their priority is stated plainly**, including that their product-dependent finding is
  the more informative one.
- **The inaccessible supplement is disclosed**, along with the possibility that our
  observation is already covered. A specialist would check this immediately; naming it
  first is the only credible option.
- **The size shortfall of their published cohort is raised by us**, not left for them to
  point out.
- **Compute-inside-their-environment is offered** and is true — measured peak VRAM 0.94 GB.
- **No clinical benefit is promised** and no novelty is claimed.
- Figures quoted — 0.781, 0.618, 0.163 (0.137–0.189), 0.817, 0.719, 0.098 (0.077–0.120),
  0.065 (0.040–0.090), 35%, 2,708, 280 — all verified in
  `tables/claim_verification.json`.

## If a second group is approached later

Alternatives are listed in `DATA_REQUEST_CHECKLIST.md`. Do not send near-identical
messages in parallel; if you approach a second group, say that a first approach was made
and what came of it.
