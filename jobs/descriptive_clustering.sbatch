#!/bin/bash
#SBATCH --job-name=paper-original-kmeans
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/reproduce_descriptive_clustering.py --task "$SLURM_ARRAY_TASK_ID"
