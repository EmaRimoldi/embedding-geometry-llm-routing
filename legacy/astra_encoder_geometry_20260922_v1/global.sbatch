#!/bin/bash
#SBATCH --job-name=astra-geo-global
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/global-%j.out
#SBATCH --error=logs/global-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u global_geometry.py
