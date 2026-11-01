# Limitations

## On the rights assessment

- CC0-1.0 resolves **copyright and database rights only**. It does not waive
  human-subjects obligations, does not constitute ethics approval, and does not imply
  re-identification risk is zero. These are de-identified radiographs of real patients and
  should be handled as sensitive whatever the licence says.
- The dataset **landing page still displays no licence**. The CC0 finding rests on the
  archive-wide policy plus the dataset-specific API field. Both are official, but a reader
  looking only at the landing page would not see it.
- `"versionStatus": "submitted"` on version 5 is recorded as an observation. No conclusion
  is drawn from it.
- Legal interpretation is not a competence this run can claim. The assessment is
  evidence-based and reversible if a contradictory notice appears.

## On the concept audit, if and when it runs

- **A similarity to the sentence "fibrosis" is not a fibrosis detection.** Every concept
  value is a proxy in a learned image-text space, labelled as such.
- **Running CLEAR does not validate CLEAR** for TB sequelae, and this project has not
  validated it.
- **3 of 12 preregistered concepts are excluded** for lack of an exact vocabulary entry,
  including **cavity**, which is clinically important in TB. That is a genuine coverage
  gap.
- **CLEAR's exposure to this cohort is unknown.** Its card names three training corpora
  and no evaluation set. If it has seen this cohort, associations measured here are
  exploratory and potentially exposed.
- **Vietnam TUNE is already exposed.** It informed the pooling configuration and has been
  analysed repeatedly. Nothing measured on it is independent validation.
- The previous-TB subgroup was **not prespecified** when first examined, and `past_tb` is
  **self-reported** history, not record-verified and not a radiologist annotation.
- **Association is not causation.** Even a strong, BH-surviving correlation between a
  concept similarity and the classifier score would not show the classifier uses that
  finding. Ruling that in needs an intervention, which is out of scope and unauthorised.
- One image per patient in this cohort, so case-level resampling equals patient-level.
  **That is a property of this cohort, not a general licence to resample by case.**

## On what this run did not do

- No model trained, fine-tuned, rescored, recalibrated or re-thresholded.
- **Vietnam REPORT never opened.** TBX11K `internal_test` never opened.
- RayDINO weights never downloaded, never re-hosted; its benchmark remains out of scope.
- Nothing uploaded anywhere, no email sent, no contact detail invented.
- No concept score used to modify, subtract from or suppress any TB score or referral.
- No per-case clinical or image-derived table placed in the shareable bundle.

## The honest summary

Two blockers were removed and one appeared. The audit is closer to running than it was,
and **it has still produced no concept score.** The reported screening AUROC is unchanged
at 0.7242.
