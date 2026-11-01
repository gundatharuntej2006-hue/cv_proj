#!/usr/bin/env bash
# Rebuild paper.pdf from paper.tex (two pdflatex passes, MiKTeX) and verify against MANIFEST.sha256.
#   bash report/arxiv/build.sh            # build + verify
#   bash report/arxiv/build.sh --freeze   # build + rewrite the manifest (after an intentional edit)
set -euo pipefail
cd "$(dirname "$0")"
PDFL="${PDFLATEX:-C:/Users/atuld/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe}"
"$PDFL" -interaction=nonstopmode -halt-on-error paper.tex > build.log 2>&1
"$PDFL" -interaction=nonstopmode -halt-on-error paper.tex > build.log 2>&1
rm -f paper.aux paper.out
grep -n "Output written" build.log
if [ "${1:-}" = "--freeze" ]; then
    sha256sum paper.tex ../paper_draft_v2.md ../project_report_v2.md > MANIFEST.sha256
    echo "manifest rewritten (sources only; the PDF differs byte-wise between runs because of embedded timestamps)"
else
    sha256sum -c MANIFEST.sha256
fi
