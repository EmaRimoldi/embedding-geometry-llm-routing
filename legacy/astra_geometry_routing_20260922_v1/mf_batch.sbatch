#!/bin/bash
#SBATCH --job-name=astra-route-mf-scale
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mf-batch-%A_%a.out
#SBATCH --error=logs/mf-batch-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# Four serial representation/normalization controls per held-out family.
# Uses one GPU and one scheduler entry per family, not four queued entries.
for study_variant in 0 1 2 3; do
  study_task=$((SLURM_ARRAY_TASK_ID * 4 + study_variant))
  python -u mf_control.py "$study_task"
done
