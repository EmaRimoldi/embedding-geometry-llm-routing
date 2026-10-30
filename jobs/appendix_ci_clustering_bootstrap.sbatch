#!/bin/bash
#SBATCH --job-name=paper-ci-clustering_bootstrap
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/appendix_ci/clustering_ci.py "$SLURM_ARRAY_TASK_ID"
