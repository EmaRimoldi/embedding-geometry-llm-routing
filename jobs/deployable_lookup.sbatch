#!/bin/bash
#SBATCH --job-name=paper-label-lookup
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/deployable_lookup.py --task "${SLURM_ARRAY_TASK_ID}"
