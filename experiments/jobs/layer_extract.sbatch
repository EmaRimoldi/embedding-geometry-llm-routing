#!/bin/bash
#SBATCH --job-name=paper-layer-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/layer_sweep/extract.py --dataset routerbench
"$PAPER_PYTHON" -u src/layer_sweep/extract.py --dataset sprout
