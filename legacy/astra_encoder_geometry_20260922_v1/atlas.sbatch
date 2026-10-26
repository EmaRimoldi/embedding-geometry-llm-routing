#!/bin/bash
#SBATCH --job-name=astra-geo-atlas
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/atlas-%A_%a.out
#SBATCH --error=logs/atlas-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u geometry.py "$SLURM_ARRAY_TASK_ID"
