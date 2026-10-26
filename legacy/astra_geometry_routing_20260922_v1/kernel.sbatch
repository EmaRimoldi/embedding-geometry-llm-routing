#!/bin/bash
#SBATCH --job-name=astra-route-kernel
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kernel-%A_%a.out
#SBATCH --error=logs/kernel-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u nonlinear.py --kind kernel --seed 42 --task "$SLURM_ARRAY_TASK_ID"
