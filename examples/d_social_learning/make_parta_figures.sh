#!/bin/bash
#
# Rebuild all part-A figures from __data__/social/ariel:
#   __data__/social/figures/parta/report/  multi-panel overview + CSVs (parta_report.py)
#   __data__/social/figures/parta/paper/   standalone PDF/PNG publication figures (paper_figures.py)
#
# Novelty/fitness figures need novelty_history, which a re-sync from the
# cluster drops: run analysis/backfill_novelty.py first if you want them.
#
# Usage (from the repo root):
#   bash examples/d_social_learning/make_parta_figures.sh
set -euo pipefail

OUT=__data__/social/figures/parta

uv run examples/d_social_learning/analysis/parta_report.py --data-dir __data__/social/ariel \
    --run-novelty STRUCT --out-dir "$OUT/report"
uv run examples/d_social_learning/analysis/paper_figures.py \
    --report-dir "$OUT/report" --out-dir "$OUT/paper" --run-novelty STRUCT
