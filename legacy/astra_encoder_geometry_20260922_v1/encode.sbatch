#!/bin/bash
#SBATCH --job-name=astra-geo-encode
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/encode-%A_%a.out
#SBATCH --error=logs/encode-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE="$PWD/cache/modules-$SLURM_ARRAY_TASK_ID" XDG_CACHE_HOME="$PWD/cache/encode-$SLURM_ARRAY_TASK_ID"
reps=(gte reader)
python -u encode.py "${reps[$SLURM_ARRAY_TASK_ID]}"
