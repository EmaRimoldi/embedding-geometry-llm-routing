#!/bin/bash
#SBATCH --job-name=paper-nv-extract
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --gres=gpu:1
set -euo pipefail
source "$PWD/jobs/env.sh"
datasets=(routerbench sprout)
"$PAPER_PYTHON" -u src/nvidia/extract.py --dataset "${datasets[$SLURM_ARRAY_TASK_ID]}"
