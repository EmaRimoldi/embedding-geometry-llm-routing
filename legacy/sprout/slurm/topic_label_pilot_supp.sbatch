#!/bin/bash
#SBATCH --job-name=e4-sprout-topic-pilot2
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export RB_RUNTIME_ROOT="${RB_RUNTIME_ROOT:-__PROJECT_SCRATCH__}"
export HF_HOME="$RB_RUNTIME_ROOT/huggingface-cache"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$RB_RUNTIME_ROOT/experiments4/sprout"
mkdir -p logs topics
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$RB_RUNTIME_ROOT/venvs/env/bin/python" scripts/topic_label_pilot_supp.py
