#!/bin/bash
#SBATCH --job-name=astra-base-cpu
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/basecpu-%A_%a.out
#SBATCH --error=logs/basecpu-%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u baselines.py --kind classical --task "$SLURM_ARRAY_TASK_ID"
