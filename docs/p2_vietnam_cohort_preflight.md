# Vietnam cohort: what the metadata says before a single image is downloaded

**Owner:** Person 2 · **Date:** 2026-10-19 · **Source:** Dryad `10.5061/dryad.5tb2rbpcv`,
CC0 · **Status:** CSV and README ingested; 20.85 GB of DICOM still to come

2,708 participants from community active-case-finding in Ho Chi Minh City, one chest
radiograph each, **Xpert MTB/RIF Ultra on every participant** as the reference standard.
280 positive (10.3%), 2,428 negative. The DICOMs arrive in 11 archives of 250 (the last has
208), named by `pid` range, and `pid` is the join key — the ranges in the filenames match the
CSV's `pid` span exactly, so the join is sound.

The 116 KB CSV turns out to answer three questions on its own.

## 1. The commercial comparator is real: Lunit INSIGHT at 0.76 / 0.78

Recomputed from the raw scores rather than taken from the paper:

| model version | AUC | 95% CI |
|---|---|---|
| Lunit INSIGHT CXR 3.1.0.0 | 0.7633 | [0.7357, 0.7893] |
| Lunit INSIGHT CXR 3.9.0.1 | 0.7827 | [0.7538, 0.8105] |

These reproduce the published figures, which is the check that the file is intact. **This is
a certified commercial product scoring 0.78 on a real screening cohort**, and it is the
number our system should be compared against here — not against the 0.99 that a curated
benchmark yields.

## 2. A scanner shortcut is impossible on this cohort — which is why it is the right test

The CSV labels the radiography system for every participant: 1/JPI Healthcare, 2/DRTECH,
3/Konica Minolta.

| system | n | TB rate |
|---|---|---|
| 1 | 408 | 0.1078 |
| 2 | 1,332 | 0.1134 |
| 3 | 968 | 0.0878 |

**Using the scanner itself as a predictor of TB gives AUC 0.4722 [0.4408, 0.5040] — chance.**

That is the property that makes this cohort uniquely valuable to us. In TBX11K, cases and
controls were collected as separate groups, so acquisition correlates with the label and a
border-only model reaches 0.9094. Here the machine carries no information about the outcome,
**so if our border-only model scores above chance on this cohort, it cannot be reading the
scanner.** It is the cleanest available test of the paper's central claim, and we can state
the result in either direction.

## 3. It bounds the body-habitus explanation

`p2_control_group.md` left two live explanations for the border signal — a collection-route
artifact, or a patient-level correlate such as cachexia visible in the shoulders and soft
tissue. The CSV carries self-reported weight loss, so the second has a measurable ceiling:

| predictor (no image at all) | AUC |
|---|---|
| cough | 0.5424 |
| weight loss | **0.5300** |
| contact with a TB case | 0.5198 |
| fever | 0.5145 |
| all seven symptoms, counted | 0.5645 [0.5295, 0.5998] |

**Weight loss alone separates TB at 0.5300 here.** It is self-reported and therefore a crude
proxy for what a radiograph would show, so this is not a hard ceiling — but it does mean the
cachexia explanation can only account for a small part of a 0.91 border signal, not the bulk
of it. Combined with §2, the collection-route explanation is the one that survives.

It is also worth noting for the report that **symptoms alone reach only 0.5645**. Symptom
screening is the standard of care in many settings; any imaging system is competing with
that floor, not with zero.

## What the images will add

Four experiments, already scripted for the other three sites:

1. **Full-image AUC** against the Lunit comparator on the same cohort. Expect 0.75–0.85;
   anything near 0.78 is a striking result for a frozen backbone plus a linear head.
2. **Border-only AUC** on a cohort where the scanner is uninformative — the decisive version
   of the shortcut test.
3. **Lungs-only AUC**, completing the dissociation at a fourth site.
4. **Per-scanner breakdown**, since `rad_system` lets us check transfer across three
   machines *within* one cohort — a sub-site analysis no other dataset here supports.

## Caveats

- HIV status was dropped by the depositors to limit indirect identifiers, and HIV strongly
  affects radiographic presentation. We cannot stratify by it.
- `age_group` is binary (18–54, ≥55) and `sex` is unlabelled as to which value is which.
- Negatives are Xpert-negative screening attendees, many with abnormal chests (old scarring,
  COPD). This is a screening cohort, not a curated normal-versus-TB set, and performance
  here will be well below benchmark figures **by design**.
- The README describes the archives as `.rar` while the download page lists `.zip`; whichever
  arrives, extraction may need an unrar tool.

Evidence: `data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv`, README from the same
deposit.
