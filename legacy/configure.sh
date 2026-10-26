#!/usr/bin/env bash
# Fill the path placeholders left by anonymization.
# Usage: ./configure.sh [DATA_ROOT] [MODEL_ROOT]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${1:-$ROOT/external}"
MODELS="${2:-$ROOT/models}"
grep -rlZ -e '__E4_ROOT__' -e '__PROJECT_SCRATCH__' -e '__PROJECT_HOME__' -e '__SCRATCH__' -e '__HOME__' \
     -e '__POOL__' -e '__SOFTWARE_PATH__' -e '__CLUSTER_PATH__' \
     --include='*.py' --include='*.sh' --include='*.sbatch' --include='*.slurm' --include='*.json' \
     --include='*.yaml' --include='*.yml' "$ROOT" \
 | xargs -0 -r sed -i \
     -e "s#__E4_ROOT__#${ROOT}#g" -e "s#__PROJECT_SCRATCH__#${DATA}#g" -e "s#__PROJECT_HOME__#${DATA}#g" \
     -e "s#__SCRATCH__#${DATA}#g" -e "s#__HOME__#${HOME}#g" -e "s#__POOL__#${MODELS}#g" \
     -e "s#__SOFTWARE_PATH__#/usr#g" -e "s#__CLUSTER_PATH__#${DATA}#g"
echo "Paths configured (root: ${ROOT}). Set SLURM placeholders (__PARTITION__, __ACCOUNT__) in *.sbatch or run the Python entry points directly."
