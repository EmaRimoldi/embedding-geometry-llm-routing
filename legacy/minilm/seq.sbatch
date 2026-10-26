#!/bin/bash
#SBATCH --job-name=m-seq
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# minilm/ study: run the shell lines of $CMDFILE one after the other (each line may set E4_OUT=...; $PY = venv python).
set -euo pipefail
export RB_RUNTIME_ROOT=__PROJECT_SCRATCH__ E4_ROOT=__E4_ROOT__ OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PY="$RB_RUNTIME_ROOT/venvs/env/bin/python"; cd "$E4_ROOT"
echo "panel=$E4_PANEL rep=$E4_REP results=$E4_RESULTS_ROOT cmdfile=$CMDFILE"
while IFS= read -r line; do [ -z "$line" ] && continue; echo "== $(date +%H:%M:%S) $line"; eval "$line"; done < "$CMDFILE"
echo "== all done $(date +%H:%M:%S)"
