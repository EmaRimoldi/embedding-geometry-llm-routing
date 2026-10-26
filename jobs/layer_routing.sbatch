#!/bin/bash
#SBATCH --job-name=paper-layer-routing
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/layer_sweep/run.py --pack "${1:-${SLURM_ARRAY_TASK_ID}}"
