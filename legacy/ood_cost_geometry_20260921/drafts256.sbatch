#!/bin/bash
#SBATCH --job-name=ocg-drafts
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=ood_cost_geometry_20260921/logs/%x-%j.out
#SBATCH --error=ood_cost_geometry_20260921/logs/%x-%j.err
# Usage (from experiments4): sbatch --export=ALL,PANEL=<panel> ood_cost_geometry_20260921/drafts.sbatch
set -euo pipefail
export RB_RUNTIME_ROOT="__PROJECT_SCRATCH__" E4_ROOT="__E4_ROOT__"
export HF_HOME="$RB_RUNTIME_ROOT/huggingface-cache" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$E4_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$RB_RUNTIME_ROOT/venvs/env/bin/python" ood_cost_geometry_20260921/scripts/gen_drafts.py --panel "$PANEL" --max_new 256 --out drafts256
