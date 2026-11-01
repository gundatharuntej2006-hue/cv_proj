# -*- coding: utf-8 -*-
"""Generate PROPOSED_OUTREACH_REVISIONS.patch as a real unified diff.

Staged copies of the affected files are edited in a scratch directory and diffed against
the originals. **No original file is modified.**
"""
import difflib
import io
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
PKG = ROOT / "artifacts/collaboration_package_20261021_1145"
PREV = ROOT / "artifacts/prevtb_subgroup_20261021_1055"
STAGE = OUT / "staged"

DO_NOT_SEND = """> # ⛔ DO NOT SEND — SUPERSEDED 2026-10-21
>
> A targeted prior-art check found that **this cohort is the published test library of
> Codlin et al., *PLOS Digital Health* 2025;4(4):e0000813** (DOI
> 10.1371/journal.pdig.0000813), which **already reports the previous-TB subgroup analysis
> for Lunit INSIGHT CXR v3.9.0.1 on exactly these 2,708 participants** (past TB 0.73, no
> past TB 0.81). All six of its published AUCs reproduce from our local copy at two
> decimal places.
>
> Further, under that paper's construction — full stratification, where each stratum has
> its own positives — **our difference-in-differences is +0.0481 [−0.0170, +0.1113] and
> includes zero.** The +0.0651 [0.0399, 0.0900] quoted below comes from a shared-positive
> construction that narrows the between-stratum interval and is not what the literature
> reports.
>
> Both premises of this email therefore fail: the question is not open, and the
> comparative observation is not supported. **This draft needs rewriting, not editing.**
> See `prior_art_review_20261021_1200/PRIOR_ART_VERIFICATION.md` §5.

"""

CONCEPT_BANNER = """> # ⛔ SUPERSEDED 2026-10-21 — premise invalidated
>
> The "what we observed" section below is framed as an open question. It is not: the
> cohort is the published test library of **Codlin et al., *PLOS Digital Health*
> 2025;4(4):e0000813**, which already reports the previous-TB subgroup analysis for the
> same Lunit version. Under that paper's stratification our difference-in-differences
> includes zero (+0.0481 [−0.0170, +0.1113]).
>
> Retained only as a record. See
> `prior_art_review_20261021_1200/CLAIM_BOUNDARIES_REVISED.md`.

"""

CLAIMS_BANNER = """> # ⚠ SUPERSEDED in part — see `CLAIM_BOUNDARIES_REVISED.md` (2026-10-21)
>
> Two claims listed below as "supported now" have moved to "must not be claimed": the
> difference-in-differences (+0.0651), which includes zero under the published
> stratification, and the implicit premise that the previous-TB question on this cohort is
> open, which Codlin et al. 2025 settles. This file is kept unmodified as the record of
> what was believed before that paper was found.

"""

LITLOG_ADD = """
## 6. Added 2026-10-21 — the paper that supersedes this package

> **Codlin AJ, Vo LNQ, Dao TP, Forse RJ, Dang HTM, Nguyen LH, Nguyen HB, Dinh LV,
> Sidney Annerstedt K, Lundin J, Lönnroth K.** "Comparison of different Lunit INSIGHT CXR
> software versions when reading chest radiographs for tuberculosis." *PLOS Digital
> Health* 2025;4(4):e0000813. DOI 10.1371/journal.pdig.0000813. Published 15 Apr 2025.
> **RETRIEVED, full text, open access.**

This is the source publication for the Dryad dataset at `data/raw/vietnam/`
(DOI 10.5061/dryad.5tb2rbpcv). Ho Chi Minh City active case finding, 127 mobile screening
events, March 2019 – March 2021, **n = 2,708**, **10.3% Xpert Ultra positive**, **35.2%
previous TB**.

**It reports the previous-TB subgroup analysis we believed was open:** Lunit v3.1.0.0 vs
v3.9.0.1, past TB **0.67 vs 0.73** (p = 0.003), no past TB **0.81 vs 0.81** (p = 0.644),
overall 0.76 vs 0.78 (p = 0.029).

**All six values reproduce from our local copy at two decimal places**, confirming cohort
identity and revealing that the paper uses **full stratification** where our earlier
analysis shared positives across strata.
"""

PRIORART_ADD = """
## 6. Added 2026-10-21 — the cohort's own source publication

The Vietnam cohort used throughout this project is the **published test library of
Codlin et al., *PLOS Digital Health* 2025;4(4):e0000813** (DOI
10.1371/journal.pdig.0000813), released as Dryad 10.5061/dryad.5tb2rbpcv. That paper
already reports the previous-TB subgroup analysis for Lunit INSIGHT CXR v3.9.0.1 — the
comparator used here — on exactly these 2,708 participants.

Consequences for this document:

1. **The previous-TB question on this cohort is not open.** It is published, by the group
   that produced the data.
2. **Our analysis used a different construction.** Codlin et al. stratify every
   participant by `past_tb`; our earlier run shared all 280 positives across both strata.
   Under their construction our difference-in-differences is **+0.0481
   [−0.0170, +0.1113]** — it includes zero.
3. **Every description of this cohort in the project should cite Codlin et al. 2025.**
   None currently does.

Full detail: `prior_art_review_20261021_1200/PRIOR_ART_VERIFICATION.md`.
"""


def stage(src, rel, transform):
    dst = STAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(transform(text), encoding="utf-8")
    return text, dst.read_text(encoding="utf-8")


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    jobs = [
        (PKG / "OUTREACH_EMAIL_DRAFT.md",
         "collaboration_package_20261021_1145/OUTREACH_EMAIL_DRAFT.md",
         lambda t: DO_NOT_SEND + t),
        (PKG / "ONE_PAGE_CONCEPT_NOTE.md",
         "collaboration_package_20261021_1145/ONE_PAGE_CONCEPT_NOTE.md",
         lambda t: CONCEPT_BANNER + t),
        (PKG / "CLAIM_BOUNDARIES.md",
         "collaboration_package_20261021_1145/CLAIM_BOUNDARIES.md",
         lambda t: CLAIMS_BANNER + t),
        (PKG / "LITERATURE_VERIFICATION_LOG.md",
         "collaboration_package_20261021_1145/LITERATURE_VERIFICATION_LOG.md",
         lambda t: t.rstrip("\n") + "\n" + LITLOG_ADD),
        (PREV / "PRIOR_ART_COMPARISON.md",
         "prevtb_subgroup_20261021_1055/PRIOR_ART_COMPARISON.md",
         lambda t: t.rstrip("\n") + "\n" + PRIORART_ADD),
    ]

    chunks = []
    for src, rel, fn in jobs:
        before, after = stage(src, rel, fn)
        d = difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="a/artifacts/" + rel, tofile="b/artifacts/" + rel, n=3)
        chunks.append("".join(d))
        print("staged %s (+%d lines)" % (rel, len(after.splitlines()) - len(before.splitlines())))

    header = (
        "# Proposed revisions after the targeted prior-art check, 2026-10-21\n"
        "#\n"
        "# NONE OF THESE HAVE BEEN APPLIED. No original file was modified.\n"
        "# Apply from the repository root with:\n"
        "#     patch -p1 -i artifacts/prior_art_review_20261021_1200/"
        "PROPOSED_OUTREACH_REVISIONS.patch\n"
        "# Verified with `patch --dry-run -p1`, exit 0, on 2026-10-21.\n"
        "# (`git apply` rejects these paths because the targets are untracked.)\n"
        "# Or inspect the already-staged results under\n"
        "#     artifacts/prior_art_review_20261021_1200/staged/\n"
        "#\n"
        "# Rationale, per file:\n"
        "#  1 OUTREACH_EMAIL_DRAFT.md      DO NOT SEND banner. Both premises fail.\n"
        "#  2 ONE_PAGE_CONCEPT_NOTE.md     superseded banner; premise invalidated.\n"
        "#  3 CLAIM_BOUNDARIES.md          pointer to the revised boundaries.\n"
        "#  4 LITERATURE_VERIFICATION_LOG  add Codlin et al. 2025 and the reproduction.\n"
        "#  5 PRIOR_ART_COMPARISON.md      record the cohort's own source publication.\n"
        "#\n"
        "# These are minimal, additive annotations. They do NOT rewrite the outreach\n"
        "# materials, because the recommendation is not to send them at all.\n"
        "\n")
    (OUT / "PROPOSED_OUTREACH_REVISIONS.patch").write_text(
        header + "\n".join(chunks), encoding="utf-8")
    print("\nwrote PROPOSED_OUTREACH_REVISIONS.patch")


if __name__ == "__main__":
    main()
