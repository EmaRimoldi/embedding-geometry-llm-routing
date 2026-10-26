#!/bin/bash
#SBATCH --job-name=astra-final-fixed
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fixed-%A_%a.out
#SBATCH --error=logs/fixed-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u code/run_fixed.py --task "$SLURM_ARRAY_TASK_ID" --kind "${ROUTER_KIND:?}"
