#!/bin/bash
#SBATCH --job-name=mlpt-verify
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
R=__PROJECT_SCRATCH__
export E4_ROOT=$R/experiments4 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $E4_ROOT/mlp_tuned_20260924_v1
PY=$R/venvs/env/bin/python
$PY -u code/verify.py
