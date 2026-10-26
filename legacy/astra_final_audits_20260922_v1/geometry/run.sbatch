#!/bin/bash
#SBATCH --job-name=astra-geo-causes
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-16%2
#SBATCH --output=logs/geo-%A_%a.out
#SBATCH --error=logs/geo-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -u geometry/intervene.py "$SLURM_ARRAY_TASK_ID"
