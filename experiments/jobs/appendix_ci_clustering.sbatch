#!/bin/bash
#SBATCH --job-name=paper-ci-clustering
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-29%4
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/appendix_ci/clustering_assignments.py "$SLURM_ARRAY_TASK_ID"
