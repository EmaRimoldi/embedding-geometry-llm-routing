#!/bin/bash
#SBATCH --job-name=gte-l21-extract
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
set -euo pipefail
source "$PWD/jobs/env.sh"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then dataset=routerbench; else dataset=sprout; fi
"$PAPER_PYTHON" -u src/diagnostics/gte_l21/extract.py --encoder gte-qwen2-7b --dataset "$dataset"
