#!/bin/bash
#SBATCH --job-name=paper-cost-reward-format
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/knn_cost_reward_format.py run --task "$SLURM_ARRAY_TASK_ID"
