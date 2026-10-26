#!/bin/bash
#SBATCH --job-name=astra-base-mf
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/basemf-%A_%a.out
#SBATCH --error=logs/basemf-%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u baselines.py --kind mf --task "$SLURM_ARRAY_TASK_ID"
