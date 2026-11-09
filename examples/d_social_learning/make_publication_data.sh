#!/bin/bash
#
# Package the final ARIEL and EvoGym data (reps 0-9, the runs behind the
# paper's figures) as three download tiers per domain, in
# __data__/publication/{ariel,evogym}/:
#   <domain>_1_core.zip              per-individual CSVs, runs.csv, derived/ figure inputs
#   <domain>_2_learning_curves.zip   per-individual CMA-ES learning curves
#   <domain>_3_raw_databases.zip     the original SQLite run databases
#   SHA256SUMS
# plus one __data__/publication/README.md (from
# examples/d_social_learning/publication/README.md). Run
# make_final_results_figures.sh first, so derived/ matches the figures.
# Tier 3 compresses ~100 GB and takes a few hours. The Zenodo record holds the
# zips in one flat folder, so its README lists them without the {ariel,evogym}/
# prefix.
#
# Usage (from the repo root):
#   bash examples/d_social_learning/make_publication_data.sh            # all tiers
#   TIERS="1 2" bash examples/d_social_learning/make_publication_data.sh
set -euo pipefail

TIERS=${TIERS:-1 2 3}
REPS=${REPS:-0-9}
WORKERS=${WORKERS:-16}

for DOMAIN in ariel evogym; do
    # shellcheck disable=SC2086
    uv run examples/d_social_learning/analysis/export_publication.py \
        --domain "$DOMAIN" --tiers $TIERS --reps "$REPS" --workers "$WORKERS"
done
