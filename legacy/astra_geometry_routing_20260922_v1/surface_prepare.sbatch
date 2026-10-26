#!/bin/bash
#SBATCH --job-name=astra-route-surface-prep
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/surface-prep-%j.out
#SBATCH --error=logs/surface-prep-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u surface.py prepare
