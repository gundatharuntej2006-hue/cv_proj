# The TBX11K label questions are answerable from the files we already have

**Owner:** Person 2 · **Date:** 2026-10-18 · **For:** Person 1 (Tharun), before emailing the
dataset authors · **Artifact:** `artifacts/p1/labels/tbx11k_image_labels_v1.csv`

A draft email to the TBX11K authors asked four questions. Three are answered by the
annotation files already in the archive; the fourth is a genuine ambiguity worth one short
message. The derived counts match the paper exactly, which is the check that the derivation
is right rather than merely plausible.

## 1. "An official image-level label file"

It exists: `annotations/json/TBX11K_trainval.json` (and the per-split files). It is COCO
format, so the image-level diagnosis is not in a `label` field — it is in `file_name`:

    {"id": 1, "file_name": "tb/tb0003.png", "width": 512, ...}

The folder prefix **is** the official three-class label: `health/` → healthy, `sick/` →
sick-but-not-TB, `tb/` → tuberculosis. All 8,400 trainval images carry one, with no gaps:
3,800 healthy, 3,800 sick_non_tb, 800 TB.

The TB **subtype** comes from the lesion box categories declared in the same file:

| id | name | meaning |
|---|---|---|
| 1 | ActiveTuberculosis | active |
| 2 | ObsoletePulmonaryTuberculosis | latent / healed |
| 3 | PulmonaryTuberculosis | unspecified TB |

An image whose boxes are all category 1 is active, all category 2 is latent, and a mix is
active-and-latent. This is the dataset's own definition, not an inference we are imposing.

## 2. The 473 / 104 / 23 discrepancy — resolved

Deriving subtypes from the boxes on the official **train** list gives:

| | from boxes | published |
|---|---|---|
| active | 473 | **473** ✓ |
| latent | 103 | **104** ✗ |
| active + latent | 23 | **23** ✓ |
| TB images with no box | 1 (`tb1199`) | — |

So the derivation agrees exactly except for one image, and the arithmetic closes it:
473 + 104 + 23 = 600, which is every TB image in the train split. There is no room for
`tb1199` to be anything else.

## 3. `tb1199` — it is latent, and we can be confident

It is the one TB image with no lesion bounding box, and by the count above it must be the
104th **latent** case. That is also clinically unsurprising: obsolete/healed tuberculosis
often shows as diffuse fibrosis or calcification rather than a discrete lesion worth boxing.

**This corrected an error in our own labels.** Our provisional split had `tb1199` as
`active_tb`, giving 631 active and 139 latent in trainval against the correct 630 and 140.
Impact is confined: the binary TB label is unchanged, so **no binary result in this project
moves**. The only affected analysis is the active-versus-latent breakdown
(`p2_latent_vs_active_tb.md`), where one case shifts between subgroups — well inside its
existing confidence intervals, and it slightly *strengthens* the finding that the two cannot
be separated. Fixed in `provisional_c01.csv`.

## 4. Taxonomy and licence

**Taxonomy** is declared in the `categories` block, quoted in §1 above.

**Licence is the one real question.** The COCO files declare, in a machine-readable
`licenses` field:

    {"name": "Attribution-NonCommercial-ShareAlike License",
     "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}

— that is **CC BY-NC-SA 2.0**, non-commercial and share-alike. The project website states
CC BY 4.0, which is materially more permissive. The README ships no licence statement at
all. For an academic mini-project the stricter reading is safe and costs us nothing, so this
blocks nothing; but it is worth one question if anyone writes.

## Recommendation

**Do not send the email as drafted.** Three of its four questions are answered above from the
shipped files, and asking them risks looking as though we had not read the annotations — the
authors' likely reply is "it is in the JSON". Waiting weeks for that would also stall the
split freeze for no reason.

If a message is sent, reduce it to the one open item:

> The COCO annotation files declare CC BY-NC-SA 2.0 in their `licenses` field while the
> project page states CC BY 4.0. Could you confirm which applies to academic use? We are
> proceeding under the stricter non-commercial terms in the meantime.

And optionally, as a courtesy rather than a request:

> For completeness, `tb1199` carries no lesion annotation; we have treated it as obsolete
> (latent) tuberculosis, since the published train counts of 473/104/23 total 600 and leave
> no alternative. Please correct us if that is wrong.

## The deliverable

`artifacts/p1/labels/tbx11k_image_labels_v1.csv` — 8,400 rows:

| column | meaning |
|---|---|
| `image_id` | e.g. `tb1199` |
| `official_split` | `train` or `val`, from the core TBX11K lists |
| `file_name` | path as recorded in the COCO file |
| `category` | `healthy`, `sick_non_tb`, `active_tb`, `latent_tb`, `active_latent_tb` |
| `label_basis` | `folder`, `lesion_box_categories`, or `deduced_from_published_counts` |

The `label_basis` column exists so that the single deduced label is visible rather than
indistinguishable from the 8,399 that are read directly. Train totals reproduce the paper's
473 / 104 / 23 exactly.

Note this covers the **core** lists only. The `all_*.txt` lists additionally reference 576
images under `imgs/extra/` (Shenzhen, Montgomery, DA, DB) which are not part of the benchmark
— see `p2_tbx11k_composition_corrected.md`.
