#!/bin/bash
#SBATCH --job-name=astra-route-mlp
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mlp-%A_%a.out
#SBATCH --error=logs/mlp-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u nonlinear.py --kind mlp --seed "${STUDY_SEED:-42}" --task "$SLURM_ARRAY_TASK_ID"
