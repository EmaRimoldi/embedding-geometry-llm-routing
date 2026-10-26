#!/bin/bash
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python
case "$1" in
 prepare) "$PY" -u prepare.py ;;
 baselines) "$PY" -u baselines.py --task "$SLURM_ARRAY_TASK_ID" ;;
 kernel) "$PY" -u nonlinear.py --kind kernel --seed "$2" --task "$SLURM_ARRAY_TASK_ID" ;;
 report) "$PY" -u report.py ;;
 *) exit 2 ;;
esac
