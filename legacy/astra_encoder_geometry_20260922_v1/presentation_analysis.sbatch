#!/bin/bash
#SBATCH --job-name=astra-geo-present-analyze
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/presentation-analysis-%j.out
#SBATCH --error=logs/presentation-analysis-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u presentation_controls.py analyze
