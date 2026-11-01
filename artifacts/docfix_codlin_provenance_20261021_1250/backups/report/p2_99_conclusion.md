# Conclusion

*Owner: Person 2. Numbers provisional pending the frozen split.*

---

## What was built

A complete, reproducible explainable tuberculosis screening pipeline: a quality and
out-of-distribution gate that abstains rather than guessing, a frozen-backbone classifier,
lesion localisation, calibrated three-way triage, a reviewing dashboard, and an HTTP service
a frontend can call. It runs end to end on a 6 GB laptop. Held-out evaluation is enforced by
a sealing procedure in code — write-once seals, signed single-use tokens held by someone
other than the person running the scoring, and an append-only ledger — rather than by a
promise in a document.

On the benchmark it reaches **AUC 0.9949**, within a whisker of the published state of the
art.

## What we learned, which is not the same thing

**The benchmark figure is substantially an artifact of the benchmark.** The same classifier
predicts tuberculosis at **0.9094 from the outer 12% of the image** — shoulders, collimation,
background, no lung tissue. Dilating the erased region to remove the apices costs 0.0039, the
model is indifferent to mirroring an anatomically asymmetric organ, and restricting it to the
lungs makes it *worse*. TBX11K collected cases and controls as separate groups, so acquisition
correlates with the label, and the model used that.

**The shortcut is dataset-specific, which is how we know the external results are real.**
Across four independent sites the border-only model scores 0.52–0.62 against 0.9094 at home,
while a lungs-only model tracks the full image within 0.015 everywhere. This dissociation is
a stronger argument for an external number than the external number alone, because it answers
the obvious objection — that the shortcut travelled too — with a measurement.

**But "external validation" conceals a ladder, and the field's standard rung is near the
top.** The same model scores 0.9382 on Shenzhen, 0.8183 on Montgomery, 0.7932 on DA/DB — and
**0.6151 on a consecutive screening cohort with a microbiological reference standard**,
against **0.7827 for a certified commercial product on identical images**. At 90% specificity
we detect 12% of cases where it detects 45%. We fail on radiographically obvious disease, not
only subtle disease.

Those four figures are one configuration (full image, global pooling, all negatives). Under
the configuration later adopted — lungs-only, healthy-only negatives, regional 12×12 max —
the rungs are **further apart, not closer**: Shenzhen 0.9690, Montgomery 0.9690, DA 0.9755,
DB 0.9497, against **0.7242** on the screening cohort. The ladder is the finding, and it
holds in both configurations.

**The cause is the negative class.** We assign a median score of 0.886 to screening-cohort
negatives against 0.953 to cases. TB-versus-healthy on the benchmark is AUC 1.0000 — a
completely saturated, uninformative metric — while TB-versus-sick is 0.9861. A screening queue
is the second case and worse: its negatives have healed tuberculosis, COPD and chronic change,
and its cases are found earlier. We trained a model to recognise abnormality and deployed it
where abnormality is the base rate.

**The safety net does not catch this.** Our out-of-distribution gate works well against the
inputs it was designed for — 100% of photographs and lateral films rejected at a 0.2%
false-flag cost — but it passes 99.2% of the cohort on which the classifier fails. The
screening cohort sits at the same measured distance as the site where we score 0.9382.
Distance-based gating detects appearance shift, not task shift. We had claimed, on three
sites, that it predicted transfer; the fourth site falsified our own claim and we report that.

## Other findings worth carrying forward

- **Active and latent tuberculosis are indistinguishable** to the model (AUC 0.55, chance),
  and 21% of its positive calls are healed old disease. No threshold separates them, so
  roughly one case in five sent for urgent assessment is not infectious.
- **Noise destroys the operating point without touching the AUC**: specificity falls from 95%
  to 26% while AUC only drops to 0.966 and sensitivity *rises*. Robustness claims must be
  made at a fixed threshold, because AUC is invariant to exactly the shift that breaks
  deployment.
- **A conformal threshold used as a target rather than a bound** was discarding 9.67% of sick
  patients and 3.39% of tuberculosis cases for no detection benefit at all.
- **The backbone is the only pipeline choice that measurably matters.** RAD-DINO beats a
  DenseNet-121 trained on 800k radiographs by ΔAUC 0.0409; crop margin, k, and head type make
  no measurable difference. Three of five comparative rows came back as "no difference", which
  is the useful part of the table.
- **The distributed TBX11K archive** contains 158 cross-split near-duplicate pairs, 576
  non-benchmark images reachable through one of its two list files, and 400 held-out test
  images that are relabelled copies of publicly available radiographs.

## What we would tell someone starting this project

**Measure the ladder, not a point.** A benchmark AUC and a curated external AUC are two
samples from the easy end. One consecutive cohort is worth more than three case-control sets,
and if none is available that should be stated as prominently as the number that is.

**Ablate the anatomy.** Mask the region of interest, re-run the same head, and report the
figure. On cached embeddings it costs minutes, and evaluating the ablation externally
separates a dataset-specific shortcut from a genuine signal. We would not have found any of
this without it.

**Compare against something real.** Scoring a commercial product on the same patients
converted "0.6151 seems low" into "0.6151 against 0.7827 on identical images" — the
difference between an ambiguous result and a diagnosis.

**Freeze the backbone.** Not as a compromise for limited hardware, but because cheap
experiments are what make this kind of dissection possible at all. Every ablation in this
report runs in minutes because the embeddings are cached.

## What this system is, honestly

Not a deployable screening tool. On the population such a tool would actually meet it detects
12% of tuberculosis at a specificity a clinic could tolerate, and a commercial product does
three times better on the same images. We would not install it anywhere.

It is a working, reproducible pipeline, and an unusually well-characterised one: we can say
what its headline number is made of, how much of that survives a change of hospital, where it
fails, and why. The commercial comparator's 0.7827 shows the task is attainable — so the gap
is ours to close, and we have localised where to start. That is a more useful place to end a
mini project than a number nobody had tested.

---

*Sources: `docs/p2_shortcut_probe.md`, `p2_shortcut_does_not_transfer.md`,
`p2_vietnam_result.md`, `p2_external_validation.md`, `p2_control_group.md`,
`p2_latent_vs_active_tb.md`, `p2_robustness.md`, `p2_backbone_baseline.md`,
`p2_ood_threshold_is_miscalibrated.md`, `p2_tbx11k_composition_corrected.md`,
`p2_d08_d09_recommendation.md`, `p2_sealed_scoring.md`. Artifacts under `artifacts/p2/`.*
