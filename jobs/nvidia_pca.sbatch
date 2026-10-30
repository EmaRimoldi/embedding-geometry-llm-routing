#!/bin/bash
#SBATCH --job-name=paper-nv-pca
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/nvidia/pca.py --task "$SLURM_ARRAY_TASK_ID"
