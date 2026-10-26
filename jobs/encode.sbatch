#!/bin/bash
#SBATCH --job-name=paper-encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
encoders=(gte-qwen2-7b qwen2.5-7b-instruct minilm-l12)
encoder=${encoders[$SLURM_ARRAY_TASK_ID]}
for dataset in routerbench sprout; do
 "$PAPER_PYTHON" -u src/encode.py --encoder "$encoder" --dataset "$dataset"
done
