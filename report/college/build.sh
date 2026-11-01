#!/usr/bin/env bash
# Rebuild the BAI506 P2 component report (two pdflatex passes, MiKTeX). Cover fields are the
# \newcommand block at the top of report.tex; logo.png and overlay_example.png sit beside it.
set -euo pipefail
cd "$(dirname "$0")"
PDFL="${PDFLATEX:-C:/Users/atuld/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe}"
"$PDFL" -interaction=nonstopmode -halt-on-error report.tex > build.log 2>&1
"$PDFL" -interaction=nonstopmode -halt-on-error report.tex > build.log 2>&1
rm -f report.aux report.out report.toc
grep -n "Output written" build.log
