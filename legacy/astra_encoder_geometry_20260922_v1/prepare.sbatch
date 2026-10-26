#!/bin/bash
#SBATCH --job-name=astra-geo-prepare
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export RB_RUNTIME_ROOT=__PROJECT_SCRATCH__
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
"$RB_RUNTIME_ROOT/venvs/env/bin/python" -u prepare.py
