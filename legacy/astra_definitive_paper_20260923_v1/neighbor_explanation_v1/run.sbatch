#!/bin/bash
#SBATCH --job-name=astra-neighbor-explain
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=neighbor_explanation_v1/logs/analysis-%A_%a.out
#SBATCH --error=neighbor_explanation_v1/logs/analysis-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
panels=(routerbench-0shot sprout-sixfam)
python -u neighbor_explanation_v1/code/run_analysis.py --panel "${panels[$SLURM_ARRAY_TASK_ID]}"
