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
report=$(sbatch --parsable --dependency=afterok:"$array" jobs/nvidia_utility_report.sbatch)
"$PAPER_PYTHON" - "$array" "$report" <<'PY'
import json,sys
from pathlib import Path
p=Path("results/nvidia_layer_validation/utility_v1/finished/audit_submission.json")
p.write_text(json.dumps(dict(audit_array=sys.argv[1],report=sys.argv[2]),indent=2)+"\n")
PY
