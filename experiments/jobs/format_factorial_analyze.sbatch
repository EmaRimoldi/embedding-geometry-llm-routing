#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=paper-format-analysis
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
encoders=(gte-qwen2-7b qwen2.5-7b-instruct)
"$PAPER_PYTHON" -u src/diagnostics/format_factorial_analyze.py "${encoders[$SLURM_ARRAY_TASK_ID]}"
