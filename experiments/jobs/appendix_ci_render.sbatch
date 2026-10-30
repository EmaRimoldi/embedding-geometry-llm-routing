#!/bin/bash
#SBATCH --job-name=paper-ci-tables
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/appendix_ci/render_tables.py
