#!/bin/bash
#SBATCH --job-name=astra-linear-probe
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=linear_probe_v1/logs/probe-%A_%a.out
#SBATCH --error=linear_probe_v1/logs/probe-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u linear_probe_v1/code/run.py --task "$SLURM_ARRAY_TASK_ID"
