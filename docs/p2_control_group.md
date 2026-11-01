# Which negatives? TBX11K survives the hard control group — and so does the shortcut

**Owner:** Person 2 · **Date:** 2026-10-17 · **Affects:** §6, §12, and how the benchmark
should be described · **Status:** provisional split, development, all three maskings

## Why ask

Most published TB screening numbers compare TB against **healthy** chests. That is the easy
comparison and not the clinical one: a screening queue is full of people with *something*
wrong with them, and the real task is telling TB from pneumonia, fibrosis or a tumour.

A 2026 audit of medical vision-language models (arXiv 2609.21763) reports that swapping
healthy controls for sick ones **dropped AUROC by 0.075 to 0.306 across every model tested**,
and that a supervised model scoring 0.999 internally fell to 0.629 externally. If our 0.9949
were mostly "TB does not look like a normal chest", the same swap should expose it.

The test costs nothing: TBX11K already contains 3,800 healthy and 3,800 sick-but-not-TB
images from the same hospitals, scanners and pipeline. Only the control population changes.

## Result 1: the benchmark holds up, and that is to TBX11K's credit

Dedicated heads, each trained on its own contrast, development split, 2,000-resample
bootstrap:

| what the model can see | vs healthy | vs sick_non_tb | drop |
|---|---|---|---|
| **full image** | **1.0000** | **0.9861** [0.9706, 0.9978] | **0.0139** |
| lungs only | 0.9998 | 0.9841 [0.9684, 0.9963] | 0.0157 |
| border only | 0.9847 | 0.9119 [0.8703, 0.9480] | 0.0728 |

**The full model loses 0.0139 AUC when the controls become sick instead of healthy** — an
order of magnitude less than the 0.075–0.306 the audit paper found elsewhere.

That is a real point in TBX11K's favour and should be said plainly in the report: the
benchmark deliberately includes 3,800 sick-but-not-TB radiographs, so a model trained on it
is forced to separate TB from *other disease* rather than from normality. Datasets assembled
by gluing a TB folder onto a set of healthy normals cannot make that claim, which is
precisely why they collapse under this test.

So the "it only works against healthy controls" criticism, which was the most likely
objection to our headline number, **does not apply here.**

**One caveat the table makes obvious:** TB versus healthy is **AUC 1.0000, perfect
separation**, with a confidence interval of [1.0000, 1.0000]. A metric that saturates
completely cannot support any comparison at all. Every TB-versus-healthy number in this
project, and arguably in the literature, should be treated as uninformative.

## Result 2: the shortcut is not simply "TB films were taken unlike normal films"

The border-only model was the point of running this. If its 0.91 came from TB images having
been acquired differently from *normal* images, then against sick controls — who arrived
through the same general radiology route — it should have collapsed.

It did not. **Border-only scores 0.9119 against sick_non_tb controls**, down from 0.9847
against healthy but still far above chance. It loses 0.0728, five times the full model's
drop, so the healthy/sick distinction *is* part of what the border encodes — but only a small
part.

Two explanations remain, and this experiment cannot separate them:

1. **Collection campaign.** TB cases may have been gathered through a TB clinic or screening
   drive while both healthy and sick-non-TB images came from general radiology, leaving an
   acquisition signature that distinguishes TB from *both* control groups.
2. **A patient-level correlate visible in the frame.** The outer 12% contains shoulders and
   soft tissue, so it encodes **body habitus** — and weight loss is a classical feature of
   active tuberculosis. On this reading the border is not reading an artifact at all; it is
   reading cachexia, a genuine sign of disease that simply is not a lung finding.

Explanation 2 is worth taking seriously because it is clinically coherent and would be easy
to dismiss as a bug. It is also **not a defence of the model**: a screening tool that works
by noticing thin shoulders will fail on a well-nourished TB patient and will flag a thin
patient with no disease, and it is not what anyone believes they are deploying.

Either way, `p2_shortcut_does_not_transfer.md` shows the border signal **collapses to chance
(0.5209) at Shenzhen**. So whichever of the two it is, it is specific to the TBX11K
population and does not survive a change of hospital — which is consistent with both readings
(a different collection route, or a different population's baseline body habitus).

## What to report

- **State that the benchmark uses sick controls and survives them**, with the 0.0139 figure.
  It pre-empts the strongest available criticism of the headline number and distinguishes
  TBX11K from the composite datasets that dominate this field.
- **Report TB-versus-healthy as saturated** (1.0000) and decline to use it for any
  comparison.
- **Report the border result honestly in both directions**: it survives the control swap,
  which rules out the simplest explanation, and it dies at Shenzhen, which rules out the
  most damaging one.
- **A distinguishing experiment exists if anyone wants it:** regress the border score against
  a crude body-habitus proxy (shoulder width or soft-tissue area measured from the frame). If
  the border score tracks habitus, explanation 2 is supported. Cheap, and it would turn a
  "we cannot tell" into an answer.

## Caveats

- Development split, 59 TB cases, provisional C01; `internal_test` untouched.
- "Sick non-TB" is TBX11K's own category and its composition is not broken down by pathology,
  so we cannot say which diseases the model finds hard.
- The deployed head (trained on TB versus all negatives) scores slightly differently from the
  dedicated heads; both are reported in the artifact. The dedicated heads are the fairer
  comparison and are what the table above shows.

Evidence: `artifacts/p2/dryrun/control_group.json`, reproduced by
`scripts/p2_25_control_group.py`.
