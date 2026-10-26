#!/bin/bash
#SBATCH --job-name=astra-mlp-packed
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mlp-pack-%A_%a.out
#SBATCH --error=logs/mlp-pack-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u code/run_mlp_pack.py --pack "$SLURM_ARRAY_TASK_ID"
