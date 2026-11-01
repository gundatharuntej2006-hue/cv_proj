# What the CLEAR "consolidation" score actually measures

**Text-side forensics.** Everything in this document is a property of the pinned
vocabulary (`mimic_concepts.csv`, sha-256 `93e975b7…`) and the pinned concept embedding
tensor (`concept_embeddings_368294.pt`, sha-256 `7241c5b9…`) at revision `142f169c…`.
**No image, no classifier score and no patient record was read to produce any number
here**, so all of it is reproducible by anyone holding those two files.

---

## 1. Exact identity

| | |
|---|---|
| label used in every prior report | `consolidation` |
| **exact vocabulary phrase** | **`consolidation`** — the bare word, verbatim, nothing elided |
| `concept_idx` | **123742** |
| CSV row (0-based) | 123742 — index equals row; the file is in index order |
| entries with this exact text | **1** (unique) |
| representation | **a single phrase.** No prompt template, no expansion, no pooled group, no ontology mapping, no positive/negative pairing |
| embedding row used | 123742 of 368294, dim 768 |

The vocabulary has two columns, `concept` and `concept_idx`, and the index is the row
number. Nothing is inferred from a sort order. The same holds for all eleven audited
concepts — `tables/concept_mapping.csv`.

**But `consolidation` is not a curated finding label.** The vocabulary is **368,294
free-text sentences** drawn from MIMIC radiology reports. The entry `consolidation` is one
sentence in that corpus that happens to be a single word. It sits alongside 14,581 other
entries that contain the word inside longer sentences — `'no focal consolidation, pleural
effusion, pulmonary edema, or pneumothorax'`, `'dense right lung base consolidation'`,
`'previous left lower lobe consolidation has largely cleared'`.

## 2. Score direction and computation — verified from CLEAR's own code

Read from the installed pinned package (`clear` @ commit `57d2eae57dda…`), not
reconstructed:

| question | answer |
|---|---|
| larger = more similar? | **yes** |
| what quantity? | **cosine similarity.** The image feature from `model.encode_image(x)` is L2-normalised, and the concept embedding row is L2-normalised, then dotted |
| normalisation | both sides, explicitly, with `.clamp_min(1e-12)` |
| temperature / logit scale | **not applied.** The checkpoint carries `logit_scale`, but the audit path does not use it; a positive scalar is monotone and could not flip a sign in any case |
| contrastive difference, negative prompt, class contrast? | **none.** It is a plain single-anchor similarity |
| within-image normalisation? | **none** |
| probability or logit? | **neither** — a raw cosine in [-1, 1] |
| projection applied? | **yes** — `encode_image` returns the projected vector; the raw backbone output lives in a different space. The loader fails closed if `model.visual.projection` is absent |
| row/column order inferred from a CSV sort? | **no** — fixed integer indices throughout, and the image axis is keyed on patient id |

**There is no sign flip anywhere.** No intermediate table negates, reverses or rescales a
concept column; the audit writes the cosine straight into the correlation. A higher
consolidation score means *more* cosine similarity to the text `consolidation`.

So the negative correlation is not a transcription error, not a column-ordering error and
not a direction-of-score error. **Those three hypotheses are eliminated.**

## 3. The finding: the encoder is blind to negation

Using only phrases that already exist as exact vocabulary entries — nothing invented:

| concept | negated phrase | **cosine** | random-pair 95th pct |
|---|---|---|---|
| **volume loss** | `no volume loss` | **+0.980** | +0.820 |
| **calcification** | `no calcification` | **+0.961** | +0.830 |
| **focal opacity** | `no focal opacity` | **+0.951** | +0.831 |
| **pleural thickening** | `no pleural thickening` | **+0.915** | +0.755 |
| **scarring** | `no scarring` | **+0.906** | +0.770 |
| **fibrosis** | `no fibrosis` | **+0.844** | +0.631 |
| **consolidation** | `consolidation resolved` | +0.776 | +0.771 |
| **consolidation** | `no consolidation` | **+0.747** | +0.771 |
| **consolidation** | `no focal consolidation` | +0.736 | +0.771 |

`volume loss` and `no volume loss` are cosine **0.980** apart — that is, essentially the
same point in embedding space. Six of nine pairs sit far above the 95th percentile of
cosines to random vocabulary entries.

This is a **general property of the encoder, not a quirk of consolidation.** It affects
the project's headline concept too: `scarring` and `no scarring` are **+0.906**.

**Corroboration from real sentences.** Sampling 1,500 vocabulary entries that mention
consolidation and splitting them on negation cues, their cosine to the `consolidation`
anchor is:

| | n | median cosine |
|---|---|---|
| affirmative | 1,127 | +0.6047 |
| **negated** | 373 | **+0.6161** |

The negated sentences are, if anything, **closer** to the anchor than the affirmative ones
(gap −0.011). A similarity to this anchor carries **no information about whether the
finding is present or absent**.

## 4. Why consolidation specifically, and not the others

Negation blindness is uniform. What is *not* uniform is how often each concept appears
negated in the corpus the anchor was embedded from:

| concept | vocabulary entries containing the stem | of which negated | **negated fraction** |
|---|---|---|---|
| **consolidation** | 14,582 | 4,029 | **27.6%** |
| fibrosis | 1,201 | 76 | 6.3% |
| focal opacity | 46,100 | 2,634 | 5.7% |
| calcified granuloma | 1,248 | 42 | 3.4% |
| calcification | 7,535 | 219 | 2.9% |
| pleural thickening | 2,005 | 43 | 2.1% |
| bronchiectasis | 615 | 13 | 2.1% |
| volume loss | 2,300 | 39 | 1.7% |
| **scarring** | 4,752 | 76 | **1.6%** |
| degenerative changes | 2,505 | 36 | 1.4% |
| scoliosis | 1,275 | 14 | 1.1% |

**Consolidation is negated 27.6% of the time — 17 times scarring's rate and more than
four times the next concept.** That is not an accident of this vocabulary: *"no focal
consolidation"* is a stock phrase in the normal-chest-radiograph report template, while
nobody writes *"no scarring"* unless scarring was specifically in question.

**The two facts compose.** The anchor cannot separate presence from absence, and the
region of text space around this particular anchor is dominated by sentences that *deny*
the finding — the language of a normal chest. The consolidation axis therefore behaves
less like a consolidation detector and more like a **"this radiograph reads like a normal
chest" detector**, which would correlate *negatively* with a TB classifier score.

Supporting geometry: the anchor's cosine to `clear lungs` is **+0.597**, close to its
cosine to `scarring` (+0.587) and above its cosine to `pneumonia` (+0.625 — barely
higher). It is nearer to a normality phrase than a strict finding anchor should be.

**This is a hypothesis about mechanism, and §4 of `CORRELATION_REPRODUCTION_AND_ROBUSTNESS.md`
tests it on the image side** with the probes declared in advance in
`configs/concept_indices.json`.

## 5. An irony worth recording

The exact-single-entry rule was adopted in `dryad_rights_clear_audit_20261023_0900`
*precisely because* negation is common among consolidation sentences — that audit measured
28.6% by a different matching rule, closely agreeing with the 27.6% here, and concluded
that concepts should never be approximated by aggregating related sentences.

**The rule does not achieve what it was adopted for.** Avoiding aggregation prevents
explicitly averaging negated sentences into the anchor, but it cannot help when the
single-token anchor is *itself* sitting at cosine +0.75 from `no consolidation`. The
contamination is in the embedding, not in the aggregation.

## 6. Concept geometry

Text-side cosines among the eleven audited concepts
(`tables/concept_similarity_matrix.csv`). Consolidation's nearest neighbours:

| neighbour | cosine |
|---|---|
| **focal opacity** | **+0.847** |
| calcification | +0.793 |
| volume loss | +0.760 |
| pleural thickening | +0.696 |
| degenerative changes | +0.679 |

Consolidation is **not an independent axis**. It is strongly collinear with focal opacity
— unsurprising, since both describe airspace density — and the image-side Spearman
between the two similarity vectors was **+0.797** in the previous run, closely tracking the
text-side cosine of +0.847. **The image-side similarity structure largely inherits the
text-side geometry**, which is what one expects when every anchor is a single short phrase
in a shared embedding space.

This is also why "nine concepts" was never nine independent tests, and why
Benjamini-Hochberg across them is conservative for a correlated family rather than a
correction for nine independent hypotheses.

## 7. What this means for the word "consolidation"

The strongest defensible statement about any of these axes is:

> *Similarity to a short radiological text phrase is associated with the model score.*

Not *"the model responds to consolidation"*. Not *"consolidation is absent in high-scoring
cases"*. The axis is a **topic anchor**, not a finding detector, and for this particular
anchor the topic is substantially **the language of a normal chest report**.

Every concept name in this project's prior reports should be read with that caveat —
including `scarring`, whose anchor is cosine **+0.906** from `no scarring`.
