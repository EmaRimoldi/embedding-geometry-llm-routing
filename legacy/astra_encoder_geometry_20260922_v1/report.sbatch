#!/bin/bash
#SBATCH --job-name=astra-geo-report
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=logs/report-%j.out
#SBATCH --error=logs/report-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$PWD/cache/matplotlib"
python -u report.py
