#!/bin/bash
# CPU: one pack of sweep groups from config/plan.json. Usage: sbatch --array=0-59%30 --export=ALL,NPACKS=60 slurm/sweep.sbatch
#SBATCH --job-name=ls-sweep
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
R=__PROJECT_SCRATCH__
export E4_ROOT=$R/experiments4 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $E4_ROOT/layer_sweep_20260924_v1
$R/venvs/env/bin/python -u code/sweep.py --plan config/plan.json --pack "$SLURM_ARRAY_TASK_ID" --npacks "${NPACKS:?}"
