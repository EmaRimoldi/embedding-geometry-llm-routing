#!/bin/bash
#SBATCH --job-name=nv-u-audit-submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
array=$(sbatch --parsable --array=0-22%4 jobs/nvidia_utility_audit_task.sbatch)
array=${array%%;*}
"$PAPER_PYTHON" -B scripts/submit_utility_report.py --audit-array "$array"
