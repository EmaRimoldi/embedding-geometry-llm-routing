#!/bin/bash
# GPU: all-layer prefill hidden states for one (reader, panel). Usage: sbatch --export=ALL,READER=...,PANEL=... slurm/extract.sbatch
#SBATCH --job-name=ls-extract
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
R=__PROJECT_SCRATCH__
export E4_ROOT=$R/experiments4 HF_HOME=$R/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd $E4_ROOT/layer_sweep_20260924_v1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$R/venvs/env/bin/python -u code/extract.py --reader "${READER:?}" --panel "${PANEL:?}" ${EXTRA:-}
