#!/bin/bash
#SBATCH --job-name=paper-ci-routing
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/appendix_ci/routing.py "$SLURM_ARRAY_TASK_ID"
