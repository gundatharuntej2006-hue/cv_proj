# One in five of the system's TB calls is old scarring, and it cannot tell

**Owner:** Person 2 · **Date:** 2026-10-16 · **Affects:** §6, §9 (triage), §12 limitations,
D11 · **Status:** measured on the provisional split, development + calibration

## The setup

`heads.binary_labels` maps **every** TB subtype to positive: `active_tb`, `latent_tb` and
`active_latent_tb` are all label 1. So the model is trained to call latent TB "TB", and every
sensitivity figure in the report is a blend of two clinically different findings:

- **active TB** — transmissible, needs treatment now. This is what a screening programme is
  for.
- **latent TB** — healed or old lesions, often calcified. Not infectious, not urgent.

Two questions follow, and they have different answers.

## 1. Is the blended sensitivity hiding a weakness? No.

Plausible worry: calcified old scars are high-contrast and obvious, so latent TB might be
much *easier* to detect, inflating the blended number above the system's real ability to find
active disease. Measured at the fitted operating point (`t_lower` = 0.9069):

| subtype | n | sensitivity | 95% CI |
|---|---|---|---|
| active_tb | 125 | **0.9280** | [0.8688, 0.9617] |
| latent_tb | 28 | 0.8929 | [0.7280, 0.9629] |
| active_latent_tb | 5 | 1.0000 | [0.4385, 1.0000] |
| **ALL TB (what the report quotes)** | 158 | **0.9280** | — |

Pooled over development and calibration for power. The intervals overlap comfortably, so
**no difference is demonstrated** — if anything latent TB is detected slightly *less* often.
The blended figure is representative, and the worry does not hold. Good.

(Split separately: development active 0.9574 / latent 0.9000; calibration active 0.9103 /
latent 0.8889. Calibration rows are in-sample for `t_lower`, which was fitted there.)

## 2. Can the system tell them apart? No — and that is the real finding.

Among TB-positive cases only, using the calibrated TB score to separate active from latent:

| split | AUC, active vs latent | n |
|---|---|---|
| development | **0.5532** | 47 active vs 10 latent |
| calibration | **0.5794** | 78 active vs 18 latent |

**Chance.** The score carries essentially no information about which kind of TB it is found,
and the median calibrated probability is almost identical for both (0.9998 active, 0.9989
latent). The model is not merely uncertain about the distinction — it does not represent it.

This is expected: nothing in the training signal asked it to, since both are label 1. It is
worth measuring anyway, because the obvious hope — that a second, higher threshold could
separate "treat now" from "old scarring" — is **not available**. There is no threshold on this
score that does it.

## 3. What it costs in practice

**33 of 158 TB-positive cases (21%) are latent or mixed.** At the fitted operating point the
system flags them as TB, with the same confidence as active disease and no signal that would
let a worklist rank them lower.

So roughly **one in five cases the system sends for urgent TB workup is non-infectious old
disease**. Every metric in the report counts those as true positives, correctly by the label
definition, but the clinic experiences them as unnecessary urgent referrals. That cost is
currently invisible in the evaluation, and it is a larger effect than most of the
differences the comparative experiments in §14 chase.

## 4. What to do about it

Three honest options; the project should pick one deliberately rather than leave the blend
unstated.

**(a) Report it, change nothing.** Add the subtype breakdown and the 21% figure to §6 and the
limitations, and state that the system detects "radiographic evidence of TB, active or
healed" rather than "active TB". This is accurate, costs nothing and is the minimum
acceptable step. TBX11K's own label set invites the conflation, so saying so is also a point
about the benchmark.

**(b) Three-class the TB head.** Train (healthy, sick_non_tb, active_tb, latent_tb) as four
classes, or a dedicated active-vs-latent head over TB positives, and report whether the
distinction is learnable at all from these embeddings. Cheap to try — the embeddings are
cached and a head trains in seconds — but **underpowered**: 125 active and 28 latent cases in
the splits available, so a negative result would not be conclusive and a positive one would
be fragile. Worth doing as an exploratory note, not as a decision.

**(c) Exclude latent TB from the positive class.** This would make every number a clean
statement about active disease. It is the most defensible scientifically and the most
disruptive: it changes D11, the label mapping, every reported figure, and Person 1's C01
category semantics. **Not recommended at this stage of the calendar**, and it should not be
done quietly if it is done at all.

**Recommendation: (a) now, (b) as a short exploratory experiment.** Option (c) only if the
team decides the project's claim should be about active TB specifically, and then explicitly,
with the decision dated.

## Caveats

- 28 latent cases pooled, so the latent sensitivity interval is wide [0.7280, 0.9629]; this
  rules out only large differences.
- Provisional split; `internal_test` untouched.
- Label quality is TBX11K's. Whether its `latent_tb` annotations are bacteriologically
  confirmed or radiological impressions affects what "cannot distinguish" means — the model
  may be failing to separate two classes that are not cleanly separable in the labels either.
  That is worth checking against the benchmark's documentation before (b) or (c).

Evidence: `artifacts/p2/dryrun/tb_subgroups.json`, reproduced by
`scripts/p2_20_tb_subgroups.py`.
