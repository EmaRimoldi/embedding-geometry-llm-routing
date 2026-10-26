#!/bin/bash
#SBATCH --job-name=e4-imo-triplet
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export RB_RUNTIME_ROOT="__PROJECT_SCRATCH__"
export E4_ROOT="$RB_RUNTIME_ROOT/experiments4"
export HF_HOME="$RB_RUNTIME_ROOT/huggingface-cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$E4_ROOT"
"$RB_RUNTIME_ROOT/venvs/env/bin/python" results/hard_twins/imo_triplet/embed_imo_triplet.py
