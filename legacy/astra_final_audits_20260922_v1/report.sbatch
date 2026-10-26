#!/bin/bash
#SBATCH --job-name=astra-audit-report
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/report-%j.out
#SBATCH --error=logs/report-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u report.py
