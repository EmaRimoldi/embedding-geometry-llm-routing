#!/bin/bash
# One pack = one (panel, outer split, representation); its 6 configurations run in sequence. Restartable.
#SBATCH --job-name=mlpt
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
R=__PROJECT_SCRATCH__
export E4_ROOT=$R/experiments4 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $E4_ROOT/mlp_tuned_20260924_v1
PY=$R/venvs/env/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -u code/run_mlp.py --pack "$SLURM_ARRAY_TASK_ID"
