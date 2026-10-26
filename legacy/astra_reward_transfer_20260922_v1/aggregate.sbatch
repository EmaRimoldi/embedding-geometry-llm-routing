#!/bin/bash
#SBATCH --job-name=astra-aggregate
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate-%j.out
#SBATCH --error=logs/aggregate-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
python aggregate.py
