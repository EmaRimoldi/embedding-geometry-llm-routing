#!/bin/bash
#SBATCH --job-name=astra-graph-descriptions
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/graph-descriptions-%j.out
#SBATCH --error=logs/graph-descriptions-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u code/graph_descriptions.py
