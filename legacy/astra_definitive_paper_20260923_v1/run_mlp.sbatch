#!/bin/bash
#SBATCH --job-name=astra-final-mlp
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mlp-%A_%a.out
#SBATCH --error=logs/mlp-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-/tmp}/astra-mlp-${SLURM_JOB_ID}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
trap 'rm -rf "$JOBLIB_TEMP_FOLDER"' EXIT
python -u code/run_mlp.py --task "$SLURM_ARRAY_TASK_ID" --jobs 4
