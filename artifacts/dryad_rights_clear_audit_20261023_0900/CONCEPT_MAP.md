# Concept map — preregistered concepts against CLEAR's vocabulary

Defined from the official asset **before** any association is computed. Machine-readable:
`configs/concept_indices.json`, `tables/concept_mapping.csv`.

---

## What CLEAR's vocabulary actually is

`mimic_concepts.csv` at pinned revision `142f169c8b006b0f28cf55c6a237c3003db5ed8f`:
**368,294 free-text observations extracted from radiology reports** — whole sentences, not
a curated finding label set. Columns `concept, concept_idx`. Examples:

> "et tube and left subclavian catheter are in standard placements"
> "no pathological parenchymal process"
> "unchanged evidence of a small pleural effusion"

**This matters.** "Fibrosis" is not one model output among a dozen. The model emits a
similarity against each of 368,294 sentences, and many sentences mentioning a finding
**negate** it.

## The mapping rule, fixed in advance

> **Use only the exact single vocabulary entry whose text equals the concept term.
> Concepts with no exact entry are EXCLUDED from the primary analysis and are not
> approximated by aggregating related sentences.**

Why: aggregating over the hundreds or thousands of sentences mentioning a term requires an
aggregation rule and a negation policy, neither specified by CLEAR's documentation nor
preregistered here. Choosing one after seeing associations would be exactly the post-hoc
selection the protocol forbids.

## Coverage, measured

| concept | matching entries | affirmative | negated | % negated | exact entry? | index | in primary set |
|---|---:|---:|---:|---:|:--:|---:|:--:|
| fibrosis | 677 | 562 | 52 | 7.7% | yes | 835 | **yes** |
| scarring | 4,204 | 3,769 | 63 | 1.5% | yes | 96,952 | **yes** |
| volume loss | 2,300 | 2,086 | 40 | 1.7% | yes | 368,164 | **yes** |
| calcified granuloma | 808 | 746 | 15 | 1.9% | yes | 209,383 | **yes** |
| calcification | 4,117 | 3,827 | 125 | 3.0% | yes | 77,874 | **yes** |
| focal opacity | 730 | 486 | 179 | 24.5% | yes | 281,312 | **yes** |
| consolidation | 13,977 | 8,860 | 3,993 | 28.6% | yes | 123,742 | **yes** |
| pleural thickening | 1,986 | 1,756 | 43 | 2.2% | yes | 341,471 | **yes** |
| bronchiectasis | 614 | 520 | 13 | 2.1% | yes | 57,870 | **yes** |
| **cavity** | 203 | 175 | 18 | 8.9% | **no** | — | **excluded** |
| **nodule** | 3,546 | 2,555 | 697 | 19.7% | **no** | — | **excluded** |
| **reticular opacity** | 40 | 40 | 0 | 0.0% | **no** | — | **excluded** |

**9 of 12 preregistered concepts map to an exact single vocabulary entry. None is absent
from the vocabulary entirely; three simply lack a bare-term entry and are excluded.**

The negation column is why the exact-entry rule matters: 28.6% of sentences mentioning
"consolidation" negate it, and 24.5% of those mentioning "focal opacity" do. Pooling them
would produce a score that rises both when a finding is present and when a radiologist
explicitly rules it out.

## Honest limits of this mapping

- **A similarity to the sentence "fibrosis" is not a fibrosis detection.** It is a
  proximity in a learned image-text space. It is a **proxy**, and it is labelled as one
  wherever it is reported.
- **CLEAR is not validated for TB sequelae in this project.** Running is not validating.
  No claim is made that any of these similarities measures the finding it is named after.
- **Nothing here licenses an inference about the TB classifier.** An association between a
  concept similarity and the classifier score would be diagnostic evidence, not proof that
  the classifier "uses fibrosis".
- **`past_tb` is clinical history**, not an image annotation, and no radiological finding
  may be inferred from it.
- The three excluded concepts include **cavity**, which is clinically important in TB.
  Its absence from the primary set is a real gap in coverage, not a neutral omission.
