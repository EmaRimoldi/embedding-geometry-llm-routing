#!/bin/bash
#SBATCH --job-name=paper-reduced-preflight
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/baselines/prepare_mlp_reduced.py
