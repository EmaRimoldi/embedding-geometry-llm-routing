#!/bin/bash
#SBATCH --job-name=paper-mlp-reduced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --signal=B:USR1@600
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/baselines/run_mlp_reduced.py --worker "$1" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
set +e
wait "$pid"
status=$?
if kill -0 "$pid" 2>/dev/null; then wait "$pid"; status=$?; fi
exit "$status"
