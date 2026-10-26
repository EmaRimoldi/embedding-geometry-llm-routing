#!/bin/bash
#SBATCH --job-name=astra-reward
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reward-%A_%a.out
#SBATCH --error=logs/reward-%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export XDG_CACHE_HOME="$PWD/cache"
PY=python
"$PY" -u experiment.py --task "$SLURM_ARRAY_TASK_ID"
