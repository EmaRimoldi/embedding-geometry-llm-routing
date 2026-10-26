#!/bin/bash
#SBATCH --job-name=e4-twin-embed
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export RB_RUNTIME_ROOT="${RB_RUNTIME_ROOT:-__PROJECT_SCRATCH__}"
export E4_ROOT="$RB_RUNTIME_ROOT/experiments4"
export HF_HOME="$RB_RUNTIME_ROOT/huggingface-cache" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$E4_ROOT"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$RB_RUNTIME_ROOT/venvs/env/bin/python" results/hard_twins/embed_hard_twins.py
