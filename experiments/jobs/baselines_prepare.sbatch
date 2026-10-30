#!/bin/bash
#SBATCH --job-name=paper-base-prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --signal=B:USR1@300
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/baselines/check.py
"$PAPER_PYTHON" -u src/baselines/prepare.py
