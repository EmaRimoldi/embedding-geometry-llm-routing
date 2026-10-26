#!/bin/bash
#SBATCH --job-name=astra-final-geometry
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/geometry-%A_%a.out
#SBATCH --error=logs/geometry-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
panels=(routerbench-0shot sprout-sixfam)
reps=(gte reader)
python -u code/geometry_candidates.py --panel "${panels[$((SLURM_ARRAY_TASK_ID / 2))]}" --rep "${reps[$((SLURM_ARRAY_TASK_ID % 2))]}"
