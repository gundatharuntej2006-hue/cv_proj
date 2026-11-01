# -*- coding: utf-8 -*-
"""Five consistency scans over report/ and docs/. Documentation only.

No training, scoring, model access or internal_test access.
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
AUTHORISED = {"docs/p2_vietnam_result.md", "report/paper_draft.md",
              "docs/p2_focal_pooling_final.md"}
COHORT = (r"(Vietnam|2,708|2708|screening attendees|screening cohort"
          r"|screening test library)")
MUST_CITE = ["docs/p2_vietnam_result.md", "docs/p2_external_validation.md",
             "report/paper_draft.md", "report/p2_00_abstract_intro.md",
             "report/slides/p2_slides_outline.md", "docs/p2_focal_pooling_final.md"]
OVERCLAIM = [
    ("clinical validation", r"clinically (validated|ready|proven)"),
    ("pooling improves every cohort",
     r"(improves|helps)[^.]{0,40}(every|all) (cohort|site|dataset)"),
    ("architectural novelty asserted",
     r"(is|are) (a )?(novel|new) (architecture|pooling|aggregat\w+|method)"),
    ("previous-TB effect framed as novel",
     r"(novel|first|unprecedented)[^.]{0,120}(previous TB|past TB|prior TB)"),
    ("comparative superiority over the commercial comparator",
     r"(better|superior|outperform\w*)[^.]{0,50}(Lunit|commercial)"),
]


def out(s):
    enc = sys.stdout.encoding or "cp1252"
    print(str(s).encode(enc, "replace").decode(enc))


def sents(t):
    return re.split(r"(?<=[.!?])\s+|\s*\|\s*", " ".join(t.split()))


def main():
    res = {}

    # 1. cohort-specific "consecutive"
    hits = []
    for p in DOCS:
        rel = p.relative_to(ROOT).as_posix()
        for s in sents(p.read_text(encoding="utf-8")):
            if "consecutive" not in s.lower():
                continue
            if not re.search(COHORT, s, re.I):
                continue
            if re.search(r"not (an )?unselected|overstates|rather than unselected|"
                         r"do not call it|word \"consecutive\"|if none is available|"
                         r"beats three|worth more than|At least one", s, re.I):
                continue
            hits.append({"file": rel, "text": s[:170],
                         "in_authorised_scope": rel in AUTHORISED})
    res["scan1_cohort_consecutive"] = hits

    # 2. citation presence
    cit = []
    for rel in MUST_CITE:
        flat = " ".join((ROOT / rel).read_text(encoding="utf-8").split())
        flat = flat.replace("*", "").replace("> ", "")
        miss = [b for b in ("10.1371/journal.pdig.0000813", "Codlin",
                            "10.5061/dryad.5tb2rbpcv") if b not in flat]
        cit.append({"file": rel, "missing": miss, "pass": not miss})
    res["scan2_citation"] = cit

    # 3. the 0.6744 / 0.6743 discrepancy
    base = []
    for p in DOCS:
        rel = p.relative_to(ROOT).as_posix()
        t = p.read_text(encoding="utf-8")
        for val in ("0.6744", "0.6742", "0.6743"):
            for m in re.finditer(re.escape(val), t):
                flat = " ".join(t.split())
                i = flat.find(val)
                base.append({"file": rel, "value": val,
                             "authoritative": val == "0.6743",
                             "in_authorised_scope": rel in AUTHORISED,
                             "context": flat[max(0, i - 70):i + 70]})
                break
    res["scan3_baseline_value"] = base

    # 4. mixed-denominator comparisons: a sentence holding both a full-cohort and a
    #    REPORT-half figure without naming the denominators
    mixed = []
    for p in DOCS:
        rel = p.relative_to(ROOT).as_posix()
        for s in sents(p.read_text(encoding="utf-8")):
            has_full = bool(re.search(r"0\.6151|0\.6743|0\.6744|0\.7827", s))
            has_rep = bool(re.search(r"0\.7242|0\.7862|0\.7237", s))
            if not (has_full and has_rep):
                continue
            names = bool(re.search(r"REPORT|full cohort|2,708|1,356|denominator|"
                                   r"evaluation set|different population", s, re.I))
            mixed.append({"file": rel, "text": s[:180], "denominators_named": names,
                          "in_authorised_scope": rel in AUTHORISED})
    res["scan4_mixed_denominators"] = mixed

    # 5. overclaims
    over = []
    for p in DOCS:
        rel = p.relative_to(ROOT).as_posix()
        flat = " ".join(p.read_text(encoding="utf-8").split())
        for name, pat in OVERCLAIM:
            for m in re.finditer(pat, flat, re.I):
                over.append({"file": rel, "claim": name,
                             "context": flat[max(0, m.start() - 80):m.end() + 80]})
    res["scan5_overclaims"] = over

    # protected artifacts
    bl = ROOT / ("artifacts/docfix_codlin_provenance_20261021_1250/"
                 "tables/protected_artifact_hashes.csv")
    changed, n = [], 0
    for r in csv.DictReader(io.open(bl, encoding="utf-8-sig")):
        n += 1
        f = ROOT / r["path"]
        if not f.exists():
            changed.append([r["path"], "MISSING"])
        elif hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256_before"]:
            changed.append([r["path"], "CHANGED"])
    res["protected_artifacts"] = {"checked": n, "changed": changed,
                                  "all_unchanged": not changed}

    json.dump(res, io.open(RUN / "tables/scan_results.json", "w", encoding="utf-8"),
              indent=2)

    out("1. cohort-specific 'consecutive' claims: %d" % len(hits))
    for h in hits:
        out("   [scope:%-5s] %-34s %s" % (h["in_authorised_scope"], h["file"],
                                          h["text"][:90]))
    out("")
    out("2. Codlin + both DOIs present in cohort-describing files:")
    for c in cit:
        out("   %-40s %s %s" % (c["file"], "OK" if c["pass"] else "MISSING",
                                c["missing"] or ""))
    out("")
    out("3. baseline-value occurrences (0.6743 authoritative): %d" % len(base))
    for b in base:
        out("   %-34s %s  authoritative=%-5s scope=%-5s %s"
            % (b["file"], b["value"], b["authoritative"], b["in_authorised_scope"],
               b["context"][:70]))
    out("")
    out("4. sentences mixing full-cohort and REPORT-half figures: %d" % len(mixed))
    for m in mixed:
        out("   %-34s denominators_named=%-5s %s" % (m["file"], m["denominators_named"],
                                                     m["text"][:85]))
    out("")
    out("5. overclaim hits: %d" % len(over))
    for o in over:
        out("   %-34s %-46s %s" % (o["file"], o["claim"], o["context"][:80]))
    out("")
    out("protected artifacts: %d checked, %d changed" % (n, len(changed)))


if __name__ == "__main__":
    main()
