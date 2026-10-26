#!/bin/bash
#SBATCH --job-name=astra-freeze
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep-%A_%a.out
#SBATCH --error=logs/prep-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export RB_RUNTIME_ROOT=__PROJECT_SCRATCH__
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache"
panels=(routerbench-0shot llmrouterbench-performance-cost)
"$RB_RUNTIME_ROOT/venvs/env/bin/python" -u prepare.py "${panels[$SLURM_ARRAY_TASK_ID]}"
