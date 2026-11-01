# -*- coding: utf-8 -*-
"""Final cohort-description consistency scan over report/ and docs/.

Documentation-only. No training, scoring, model access or internal_test access.

The scan distinguishes two uses of "consecutive":
  * a DESCRIPTION OF THIS COHORT  -> an error, because recruitment was risk-based
    programmatic active case finding;
  * a GENERIC METHODOLOGICAL STATEMENT about consecutive sampling in general -> fine.

Classification is by proximity to cohort markers rather than by the word alone. Borderline
cases are reported for human review rather than silently resolved.
"""
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

DOCS = sorted(list((ROOT / "report").rglob("*.md")) + list((ROOT / "docs").rglob("*.md")))

COHORT_MARKERS = (r"(Vietnam|2,708|2708|screening attendees|screening cohort"
                  r"|screening test library)")
CITATION_BITS = ["10.1371/journal.pdig.0000813", "Codlin"]
MUST_CITE = ["docs/p2_vietnam_result.md", "docs/p2_external_validation.md",
             "report/paper_draft.md", "report/p2_00_abstract_intro.md",
             "report/slides/p2_slides_outline.md", "docs/p2_focal_pooling_final.md"]
AUTHORISED_THIS_RUN = ["report/slides/p2_slides_outline.md",
                       "docs/p2_focal_pooling_final.md"]

NOVELTY = [
    ("previous-TB effect framed as novel",
     r"(novel|first|new finding|we discover|unprecedented)[^.]{0,120}"
     r"(previous TB|past TB|prior TB)"),
    ("previous-TB effect generalised to all CAD products",
     r"(all|every) (commercial )?(CAD|AI|product|algorithm)s?[^.]{0,80}"
     r"(previous TB|past TB|prior TB)"),
    ("clinical readiness", r"clinically (validated|ready|proven)"),
    ("independent final validation of an exposed cohort",
     r"independent(ly)?[^.]{0,30}(final )?validat\w+[^.]{0,60}"
     r"(Vietnam|Shenzhen|Montgomery|DA/DB)"),
    ("causal scanner claim",
     r"scanner[^.]{0,60}(causes|caused|because of the scanner|is the cause)"),
    ("architectural novelty", r"(novel|new) (architecture|pooling|aggregat\w+|method)"),
]


def out(s):
    """The Windows console is cp1252; degrade rather than crash."""
    enc = sys.stdout.encoding or "cp1252"
    print(str(s).encode(enc, "replace").decode(enc))


def sentences(text):
    return re.split(r"(?<=[.!?])\s+|\s*\|\s*", " ".join(text.split()))


def main():
    cohort_hits, generic_hits, novelty_hits = [], [], []

    for p in DOCS:
        rel = p.relative_to(ROOT).as_posix()
        raw = p.read_text(encoding="utf-8")
        for s in sentences(raw):
            if "consecutive" not in s.lower():
                continue
            near = bool(re.search(COHORT_MARKERS, s, re.I))
            corrective = bool(re.search(
                r"not (an )?unselected|overstates|rather than unselected|"
                r"do not call it|word \"consecutive\"", s, re.I))
            rec = {"file": rel, "text": s.strip()[:200], "corrective_text": corrective,
                   "authorised_for_edit_this_run": rel in AUTHORISED_THIS_RUN}
            (cohort_hits if (near and not corrective) else generic_hits).append(rec)

        flat = " ".join(raw.split())
        for name, pat in NOVELTY:
            for m in re.finditer(pat, flat, re.I):
                novelty_hits.append({"file": rel, "claim": name,
                                     "excerpt": flat[max(0, m.start() - 70):m.end() + 70]})

    cit = []
    for rel in MUST_CITE:
        flat = " ".join((ROOT / rel).read_text(encoding="utf-8").split())
        flat = flat.replace("*", "").replace("> ", "")
        missing = [b for b in CITATION_BITS if b not in flat]
        cit.append({"file": rel, "missing": missing, "pass": not missing})

    base = ROOT / ("artifacts/docfix_codlin_provenance_20261021_1250/"
                   "tables/protected_artifact_hashes.csv")
    changed, nchecked = [], 0
    for r in csv.DictReader(io.open(base, encoding="utf-8-sig")):
        nchecked += 1
        f = ROOT / r["path"]
        if not f.exists():
            changed.append([r["path"], "MISSING"])
        elif hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256_before"]:
            changed.append([r["path"], "CHANGED"])

    res = {"cohort_consecutive_hits_for_review": cohort_hits,
           "generic_or_corrective_statements": generic_hits,
           "novelty_or_overclaim_hits": novelty_hits,
           "citation_presence": cit,
           "protected_artifacts": {"checked": nchecked, "changed": changed,
                                   "all_unchanged": not changed}}
    json.dump(res, io.open(RUN / "tables/final_consistency_scan.json", "w",
                           encoding="utf-8"), indent=2)

    out("A. 'consecutive' near a cohort marker -> needs review: %d" % len(cohort_hits))
    for r in cohort_hits:
        out("   %-36s [authorised: %s] %s"
            % (r["file"], r["authorised_for_edit_this_run"], r["text"][:100]))
    out("")
    out("B. generic or corrective statements, not errors: %d" % len(generic_hits))
    for r in generic_hits:
        out("   [%-10s] %-34s %s"
            % ("corrective" if r["corrective_text"] else "generic", r["file"],
               r["text"][:90]))
    out("")
    out("C. novelty / overclaim hits: %d" % len(novelty_hits))
    for r in novelty_hits:
        out("   %-34s %-44s %s" % (r["file"], r["claim"], r["excerpt"][:100]))
    out("")
    out("D. Codlin citation in cohort-describing files:")
    for c in cit:
        out("   %-40s %s %s" % (c["file"], "OK" if c["pass"] else "MISSING",
                                c["missing"] or ""))
    out("")
    out("E. protected artifacts: %d checked, %d changed" % (nchecked, len(changed)))
    for c in changed:
        out("   %s" % (c,))


if __name__ == "__main__":
    main()
