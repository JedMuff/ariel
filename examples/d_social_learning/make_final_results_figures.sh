#!/bin/bash
#
# Rebuild the proposed results section's figures and tables
# (__data__/social/figures/overleaf_final_results/final_results_section.tex) from the final
# data sets in __data__/final_results:
#   1. final_results_data.py caches per-run curves and donor-offspring
#      distances for ARIEL ig10 and ig20 and for EvoGym, reps 0..REPS-1
#      ($OUT/cache/)
#   2. final_results_figures.py writes the figures ($OUT/{pdf,png}/), the
#      numbers the text quotes ($OUT/stats.csv) and the LaTeX tables
#      (overleaf_final_results/final_results_tables/)
#   3. build_overleaf.py copies the figures the .tex uses into
#      overleaf_final_results/figures/, and the folder is zipped to
#      overleaf_final_results.zip (Overleaf: New Project > Upload Project)
#
# Usage (from the repo root):
#   bash examples/d_social_learning/make_final_results_figures.sh
set -euo pipefail

REPS=${REPS:-10}
WORKERS=${WORKERS:-16}
ANALYSIS=examples/d_social_learning/analysis

for B in ig10 ig20; do
    uv run "$ANALYSIS/final_results_data.py" --domain ariel --budget "$B" --reps "$REPS" --workers "$WORKERS"
done
uv run "$ANALYSIS/final_results_data.py" --domain evogym --reps "$REPS" --workers "$WORKERS"
uv run "$ANALYSIS/final_results_figures.py"
uv run "$ANALYSIS/final_results_observations.py"
OVERLEAF=__data__/social/figures/overleaf_final_results
uv run "$ANALYSIS/build_overleaf.py" --overleaf-dir "$OVERLEAF"
rm -f "$OVERLEAF.zip"
(cd "$OVERLEAF" && zip -qr "../$(basename "$OVERLEAF").zip" .)
echo "Saved $OVERLEAF.zip"
