#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper (one CPU core, about 10-15 minutes).
set -euo pipefail
cd "$(dirname "$0")/code"
python banknote_experiments.py              # E1-E5   -> results/
python paper_analysis.py lofo 2 3 4 5 6     # E6: leave-one-family-out for K = 2..6
python paper_analysis.py quadrant           # mechanism analysis (exact logits)
python make_figures.py ../paper             # figures/ and tables/, with text-collision checks
if command -v latexmk >/dev/null 2>&1; then
  (cd ../paper && latexmk -pdf -interaction=nonstopmode paper.tex)
else
  echo "latexmk not found: skipping the PDF build (paper/paper.pdf is included in the repository)"
fi
