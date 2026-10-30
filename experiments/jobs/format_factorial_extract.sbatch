#!/bin/bash
#SBATCH --job-name=paper-format-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
encoders=(gte-qwen2-7b qwen2.5-7b-instruct)
"$PAPER_PYTHON" -u src/diagnostics/format_factorial_extract.py "${encoders[$SLURM_ARRAY_TASK_ID]}"
