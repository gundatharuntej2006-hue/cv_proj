# The shortcut is real, and it does not travel — so the external number is the trustworthy one

**Owner:** Person 2 · **Date:** 2026-10-17 · **Resolves the open question in:**
`p2_shortcut_probe.md` §"External validation", `p2_external_validation.md` §4 ·
**Status:** exploratory (external sets must still be scored under seal)

## The question

Two established results that interacted badly:

- The classifier reads the border. On TBX11K, the outer 12% frame alone gives **AUC 0.9094**
  (`p2_shortcut_probe.md`).
- External validation on Shenzhen gives **AUC 0.9382** (`p2_external_validation.md`), read at
  the time as evidence that the system generalises.

But Shenzhen, like TBX11K, collected its TB cases and its normal controls as separate groups.
If those groups differ in *how* their images were acquired, the border trick could work there
too — and the 0.9382 would be the same shortcut succeeding twice, not disease recognition.
Reporting it as generalisation without checking would have been a serious overclaim.

## The test

A dissociation. Train a head on TBX11K **border-only** images and apply it unchanged to
Shenzhen and Montgomery **border-only** images. Identical head, seeds and procedure as the
full-image models; the only difference is which pixels exist.

| model | TBX11K development | Shenzhen | 95% CI | Montgomery | 95% CI |
|---|---|---|---|---|---|
| full image | 0.9949 | **0.9382** | [0.9189, 0.9552] | **0.8183** | [0.7308, 0.8986] |
| lungs only | 0.9889 | — | | — | |
| **border only** | **0.9094** | **0.5209** | [0.4759, 0.5642] | **0.6082** | [0.5147, 0.7005] |

**At Shenzhen the border model is at chance.** Its confidence interval contains 0.5. It
retains **5%** of the above-chance performance it had at home. The full model, on the same
662 images, still reaches 0.9382.

## What this establishes

**The 0.9382 cannot be explained by the border shortcut.** The shortcut channel is
demonstrably dead at Shenzhen, so whatever is driving the full model's external performance
is something else — and it is something that survived a change of hospital, country and
equipment. Combined with lungs-only reaching 0.9889 on TBX11K, the parsimonious reading is
that it is lung pathology.

This is **stronger evidence for the system than the 0.9382 was on its own.** An external AUC
can always be questioned as "maybe the shortcut followed it". That objection is now answered
with a measurement rather than an argument.

It also tells us precisely what the shortcut is: **a TBX11K-specific acquisition signature**.
The border weights learned at home encode how *that* dataset's two collection groups differ.
At a new hospital those differences are different, so the learned weights are meaningless —
exactly the behaviour a shortcut should show, and exactly why it inflates internal numbers
while contributing nothing externally.

**Montgomery retains more of it: 0.6082, keeping 26%**, with a lower bound of 0.5147 that
just clears chance. So a small residual border signal does exist there. Montgomery is 138
images, its interval is wide, and its full-model AUC is also the weakest at 0.8183 — but the
honest statement is that the shortcut is *entirely* dead at Shenzhen and only *mostly* dead
at Montgomery.

## Consequence: which number should the report lead with

This inverts the usual presentation.

| | value | status |
|---|---|---|
| TBX11K internal | 0.9949 | **inflated by a dataset-specific shortcut; do not lead with it** |
| Shenzhen external | 0.9382 | **clean; this is the number to report** |

The internal figure is the contaminated one and the external figure is the trustworthy one.
That is unusual and worth stating explicitly, because a reader's instinct is the reverse —
to treat the home number as solid and the external one as a stress test.

So the project's claim becomes: *a frozen radiology foundation model with a linear head
reaches AUC 0.9382 at an unseen hospital, and we can demonstrate that this figure is not an
artifact of dataset-specific acquisition cues, because the cue we identified is at chance
there.* That is a stronger and much more defensible claim than "0.9949 on TBX11K".

## The positive control (added 2026-10-17): the lungs do carry it

The open item above has now been run. All 800 external radiographs were segmented with the
**same** ChestX-Det PSPNet that built C05, masked by the same rule, and scored with a
lungs-only head trained on TBX11K:

| site | full image | **lungs only** | border only |
|---|---|---|---|
| Shenzhen | 0.9382 | **0.9284** [0.9077, 0.9465] | 0.5209 [0.4759, 0.5642] |
| Montgomery | 0.8183 | **0.8554** [0.7836, 0.9184] | 0.6082 [0.5147, 0.7005] |

**The argument is closed.** At Shenzhen, lung tissue alone reproduces essentially the whole
external result (0.9284 against the full image's 0.9382) while the border is at chance. The
external AUC is carried by the lungs.

**At Montgomery, lungs-only is *better* than the full image** — 0.8554 against 0.8183. Removing
everything outside the lung field *improves* performance at the more distant site. That is
what you would expect if the model's non-lung weights encode TBX11K-specific content: at a
near site it is neutral, at a distant site it actively misleads.

**The segmenter transferred perfectly: 0 of 800 external images failed the C05 sanity check.**
That was not guaranteed — the segmenter was only ever validated on TBX11K — and it matters
beyond this experiment, because Person 3's detector and the whole lung-crop path depend on it
working at a new site.

### This reopens D09, now on accuracy grounds rather than validity

`p2_d08_d09_recommendation.md` chose the full image over lung crops because accuracy was tied
internally and the full image was simpler. `p2_shortcut_probe.md` then argued for revisiting
that on validity grounds. The external measurement makes the case much stronger:

| | full image | lungs only |
|---|---|---|
| TBX11K development | 0.9949 | 0.9889 (−0.0060) |
| Shenzhen | 0.9382 | 0.9284 (−0.0098) |
| **Montgomery** | 0.8183 | **0.8554 (+0.0371)** |

Cropping costs ~0.006–0.010 at home and at a near site, and **gains 0.037 at a distant one**,
while structurally removing the shortcut's substrate. For a system whose entire purpose is
to be deployed somewhere other than where it was trained, that is a favourable trade.

**Recommendation: adopt the lung crop for the classification path**, and justify it by
external performance and shortcut removal rather than by internal accuracy, where it remains
slightly negative. This should be decided explicitly and dated, not quietly.

## Other caveats

- Exploratory. The reportable external number must still be a sealed one-shot run of the
  final model on the final C01 (`p2_sealed_scoring.md`).
- Two sites; Montgomery is only 138 images.
- "Border only" is one shortcut channel, the one we could identify and isolate. A different
  non-pathological cue living inside the lung field would not be caught by this test.
- D09 should still be reconsidered in favour of the lung crop (`p2_shortcut_probe.md`), but
  the argument weakens: since the shortcut does not transfer, cropping buys internal honesty
  rather than external accuracy.

Evidence: `artifacts/p2/shortcut/shortcut_transfer.json`, reproduced by
`scripts/p2_23_shortcut_transfers.py`.
