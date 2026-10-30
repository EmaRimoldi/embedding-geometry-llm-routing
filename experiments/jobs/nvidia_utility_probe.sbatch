#!/bin/bash
#SBATCH --job-name=nv-u-probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/nvidia_utility/probe.py --task "$SLURM_ARRAY_TASK_ID"
