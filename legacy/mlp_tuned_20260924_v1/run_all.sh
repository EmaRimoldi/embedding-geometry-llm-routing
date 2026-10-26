#!/bin/bash
# Freeze LLMRouterBench splits (once), run the equivalence gate, then every pack, then assemble. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs
PY=python
[ -f inputs/llmrouterbench-performance-cost/splits.json ] || (cd code && $PY freeze_lrb.py)
N=$(cd code && $PY run_mlp.py --list | wc -l)
v=$(sbatch --parsable slurm/verify.sbatch)
a=$(sbatch --parsable --dependency=afterok:$v --array=0-$((N-1)) slurm/mlp.sbatch)
s=$(sbatch --parsable --dependency=afterany:$a slurm/assemble.sbatch)
echo "verify: $v  mlp packs 0-$((N-1)): $a  assemble: $s"
