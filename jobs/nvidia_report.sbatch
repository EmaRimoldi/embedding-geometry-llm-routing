#!/bin/bash
#SBATCH --job-name=paper-nv-report
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/nvidia/report.py
"$PAPER_PYTHON" -u src/nvidia/compute_report.py
