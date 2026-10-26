#!/bin/bash
#SBATCH --job-name=paper-geometry-plots
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
rep=(gte-qwen2-7b qwen2.5-7b-instruct/layer_21 qwen2.5-7b-instruct/layer_23 qwen2.5-7b-instruct/layer_25 minilm-l12)
"$PAPER_PYTHON" -u src/plot_geometry.py --encoder "${rep[$SLURM_ARRAY_TASK_ID]}"
