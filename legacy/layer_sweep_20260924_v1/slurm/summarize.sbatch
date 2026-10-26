#!/bin/bash
#SBATCH --job-name=ls-summary
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
R=__PROJECT_SCRATCH__
export E4_ROOT=$R/experiments4 PYTHONDONTWRITEBYTECODE=1
cd $E4_ROOT/layer_sweep_20260924_v1
$R/venvs/env/bin/python -u code/select_layers.py
$R/venvs/env/bin/python -u code/plot.py
