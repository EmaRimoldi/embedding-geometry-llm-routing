#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=paper-family-utility
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
export CUDA_VISIBLE_DEVICES=""
datasets=(routerbench sprout)
"$PAPER_PYTHON" -u src/diagnostics/family_utility_transfer.py "${datasets[$SLURM_ARRAY_TASK_ID]}"
