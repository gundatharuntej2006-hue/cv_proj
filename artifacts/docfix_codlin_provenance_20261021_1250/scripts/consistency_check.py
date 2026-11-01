# -*- coding: utf-8 -*-
"""Phase 3: documentation consistency and claim checks. No training, scoring or model access.

Scans report/ and docs/ for the specific unsupported claims this task was asked to rule
out, verifies the citation and DOI strings, and confirms protected artifacts are unchanged.
"""
import csv
import hashlib
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

TRACKED = list((ROOT / "report").rglob("*.md")) + list((ROOT / "docs").rglob("*.md"))

# claims that must NOT appear in tracked documentation
BANNED = [
    ("patch tokens generally discarded",
     r"(discard\w*|throw\w* away)[^.]{0,80}patch token"),
    ("previous-TB effect framed as novel",
     r"(novel|first|new finding|we discover)[^.]{0,120}(previous TB|past TB|prior TB)"),
    ("our model degrades significantly more than Lunit",
     r"(significantly|statistically)[^.]{0,80}(more than|worse than)[^.]{0,40}Lunit"),
    ("claim of independent external validation of an exposed cohort",
     r"independent(ly)? (external )?validat\w+[^.]{0,60}(Vietnam|Shenzhen|Montgomery|DA/DB)"),
    ("clinical readiness claim",
     r"clinically (validated|ready|proven)"),
    ("comprehensive/systematic literature review claim",
     r"(comprehensive|systematic) (literature )?(review|search)"),
]

REQUIRED_CITATION_BITS = [
    "10.1371/journal.pdig.0000813",
    "10.5061/dryad.5tb2rbpcv",
    "Codlin",
    "PLOS Digital Health",
]

MUST_CITE = [
    "docs/p2_vietnam_result.md",
    "docs/p2_external_validation.md",
    "report/paper_draft.md",
    "report/p2_00_abstract_intro.md",
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    findings = []

    # ---- banned claims
    for p in TRACKED:
        txt = " ".join(p.read_text(encoding="utf-8").split())
        for name, pat in BANNED:
            for m in re.finditer(pat, txt, re.I):
                findings.append({"check": "banned_claim", "file": str(p.relative_to(ROOT)),
                                 "claim": name, "excerpt": txt[max(0, m.start() - 60):
                                                               m.end() + 60]})

    # ---- required citations present where the cohort is described
    cit = []
    for rel in MUST_CITE:
        # normalise whitespace: these documents hard-wrap, so a journal name can be
        # split across lines without being any less present
        txt = " ".join((ROOT / rel).read_text(encoding="utf-8").split())
        txt = txt.replace("*", "").replace("> ", "")
        missing = [b for b in REQUIRED_CITATION_BITS if b not in txt]
        cit.append({"file": rel, "missing": missing, "pass": not missing})

    # ---- the key qualifiers must be present where the citation is
    QUALIFIERS = {
        "statistical reproduction not ID matching":
            r"statistical reproduction|not (on )?patient-ID matching|rather than patient-ID",
        "DiD includes zero / inconclusive":
            r"\+0\.0481|includes zero|\[−0\.0170, \+0\.1113\]|\[-0\.0170, \+0\.1113\]",
    }
    qual = []
    v = (ROOT / "docs/p2_vietnam_result.md").read_text(encoding="utf-8")
    for name, pat in QUALIFIERS.items():
        qual.append({"qualifier": name, "present_in_p2_vietnam_result":
                     bool(re.search(pat, v, re.I))})

    # ---- "consecutive" claims about our own cohort
    consec = []
    for p in TRACKED:
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"[^.\n]{0,100}consecutive[^.\n]{0,100}", txt):
            s = " ".join(m.group(0).split())
            consec.append({"file": str(p.relative_to(ROOT)), "text": s})

    # ---- protected artifacts unchanged
    before = {r["path"]: r["sha256_before"] for r in csv.DictReader(
        io.open(RUN / "tables/protected_artifact_hashes.csv", encoding="utf-8-sig"))}
    changed = []
    for rel, h in before.items():
        f = ROOT / rel
        if not f.exists():
            changed.append((rel, "MISSING"))
        elif sha(f) != h:
            changed.append((rel, "CHANGED"))

    out = {"banned_claim_hits": findings,
           "citation_presence": cit,
           "qualifiers": qual,
           "consecutive_mentions": consec,
           "protected_artifacts": {"checked": len(before), "changed": changed,
                                   "all_unchanged": not changed}}
    json.dump(out, io.open(RUN / "tables/consistency_check.json", "w", encoding="utf-8"),
              indent=2)

    print("banned-claim hits: %d" % len(findings))
    for f in findings:
        print("  %-40s %s" % (f["claim"], f["file"]))
        print("      ...%s..." % f["excerpt"][:140])
    print("\ncitation presence:")
    for c in cit:
        print("  %-40s %s %s" % (c["file"], "OK" if c["pass"] else "MISSING",
                                 c["missing"] or ""))
    print("\nqualifiers in docs/p2_vietnam_result.md:")
    for q in qual:
        print("  %-40s %s" % (q["qualifier"], q["present_in_p2_vietnam_result"]))
    print("\n'consecutive' mentions remaining: %d" % len(consec))
    for c in consec:
        print("  %-34s %s" % (c["file"], c["text"][:110]))
    print("\nprotected artifacts: %d checked, %d changed"
          % (len(before), len(changed)))
    for c in changed:
        print("  ", c)


if __name__ == "__main__":
    main()
