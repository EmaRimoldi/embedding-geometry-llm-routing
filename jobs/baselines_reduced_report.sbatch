#!/bin/bash
#SBATCH --job-name=paper-reduced-report
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/baselines/run_mlp_reduced.py --audit
"$PAPER_PYTHON" -u src/baselines/report_reduced.py
"$PAPER_PYTHON" -u src/baselines/report_ridge.py
