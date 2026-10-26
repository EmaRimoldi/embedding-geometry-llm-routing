#!/bin/bash
#SBATCH --job-name=paper-base-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --signal=B:USR1@300
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/baselines/worker.py cpu "${1:-${SLURM_ARRAY_TASK_ID}}" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
set +e
wait "$pid"
status=$?
# A caught signal interrupts wait; keep waiting for the current fit to save.
if kill -0 "$pid" 2>/dev/null; then
    wait "$pid"
    status=$?
fi
exit "$status"
