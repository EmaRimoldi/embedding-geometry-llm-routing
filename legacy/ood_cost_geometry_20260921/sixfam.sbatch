#!/bin/bash
#SBATCH --job-name=ocg-sixfam
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=ood_cost_geometry_20260921/logs/%x-%j.out
#SBATCH --error=ood_cost_geometry_20260921/logs/%x-%j.err
# Usage (from experiments4): sbatch --export=ALL,METHOD=heldout_shrink[,EXTRA="--seed-index 0"] ood_cost_geometry_20260921/sixfam.sbatch
set -euo pipefail
export RB_RUNTIME_ROOT=__PROJECT_SCRATCH__ E4_ROOT=__E4_ROOT__ OMP_NUM_THREADS=8 E4_PANEL=sprout-sixfam
cd "$E4_ROOT"
PY="$RB_RUNTIME_ROOT/venvs/env/bin/python"; W=ood_cost_geometry_20260921/scripts/run_heldout_sixfam.py
if [ "$METHOD" = "heldout_pairwise_pareto_ranker" ]; then "$PY" "$W" "$METHOD" --panel sprout-sixfam; else for g in 0 1 2 3 4 5; do "$PY" "$W" "$METHOD" --panel sprout-sixfam --family-index $g ${EXTRA:-}; done; fi
