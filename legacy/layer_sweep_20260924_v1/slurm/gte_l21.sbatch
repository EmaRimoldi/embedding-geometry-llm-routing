#!/bin/bash
# CPU: kNN on iid_0+iid_1 for GTE emitted, GTE L21 and Qwen L21 on both panels, then the comparison table.
#SBATCH --job-name=ls-gte-l21
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
R=__PROJECT_SCRATCH__
export E4_ROOT=$R/experiments4 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $E4_ROOT/layer_sweep_20260924_v1; PY=$R/venvs/env/bin/python
for panel in routerbench-0shot sprout-sixfam; do
  $PY -u code/sweep.py --panel $panel --reader gte --pool emitted --method knn --splits iid_0+iid_1
  $PY -u code/sweep.py --panel $panel --reader gte-qwen2-7b --layer 21 --pool last --method knn --splits iid_0+iid_1
  $PY -u code/sweep.py --panel $panel --reader qwen2.5-7b-instruct --layer 21 --pool last --method knn --splits iid_0+iid_1
done
$PY -u code/compare_gte_l21.py
