#!/bin/bash
#SBATCH --job-name=astra-route-mf-scale
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mf-scale-%A_%a.out
#SBATCH --error=logs/mf-scale-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u mf_control.py "$SLURM_ARRAY_TASK_ID"
