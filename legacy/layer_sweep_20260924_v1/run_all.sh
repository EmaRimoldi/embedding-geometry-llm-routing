#!/bin/bash
# Submit the layer sweep: Qwen2.5-7B-Instruct last token, every layer, plus emitted GTE; held-out families.
# One array task per group. Stage A starts now (RouterBench + GTE; SPROUT groups exit if features are missing);
# stage B runs after the SPROUT extraction (finished or claimed groups are skipped). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs
PY=python
$PY code/tasks.py; N=$($PY -c "import json;print(len(json.load(open('config/plan.json'))['groups']))")
if [ -z "${EXTRACT_JOB:-}" ] && [ ! -f features/sprout-sixfam/qwen2.5-7b-instruct/meta.json ]; then
  EXTRACT_JOB=$(sbatch --parsable --export=ALL,READER=qwen2.5-7b-instruct,PANEL=sprout-sixfam slurm/extract.sbatch)
fi
a=$(sbatch --parsable --array=0-$((N-1)) --export=ALL,NPACKS=$N slurm/sweep.sbatch)
b=$(sbatch --parsable ${EXTRACT_JOB:+--dependency=afterok:$EXTRACT_JOB} --array=0-$((N-1)) --export=ALL,NPACKS=$N slurm/sweep.sbatch)
s=$(sbatch --parsable --dependency=afterany:$a:$b slurm/summarize.sbatch)
echo "extract: ${EXTRACT_JOB:-done}  sweep A: $a  sweep B: $b  summary: $s"
