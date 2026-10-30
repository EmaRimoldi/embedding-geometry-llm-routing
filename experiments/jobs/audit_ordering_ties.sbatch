#!/bin/bash
#SBATCH --job-name=paper-tie-audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/audit_ordering_ties.py
"$PAPER_PYTHON" -u src/compare_historical_geometry.py
