#!/bin/bash
#SBATCH --job-name=astra-route-cross
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cross-%A_%a.out
#SBATCH --error=logs/cross-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u cross_space.py --kind "${STUDY_KIND:-kernel}" --seed "${STUDY_SEED:-42}" --task "$SLURM_ARRAY_TASK_ID"
