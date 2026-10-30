#!/bin/bash
#SBATCH --job-name=gte-l21-report
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log

set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/gte_l21/route.py --report
