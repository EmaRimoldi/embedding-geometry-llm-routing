#!/bin/bash
#SBATCH --job-name=paper-prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" src/prepare_data.py routerbench
"$PAPER_PYTHON" src/prepare_data.py sprout
