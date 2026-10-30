#!/bin/bash
#SBATCH --job-name=gte-l21-route
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --array=0-22%4
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/gte_l21/route.py --task "$SLURM_ARRAY_TASK_ID"
