#!/bin/bash
#SBATCH --job-name=m-embed
#SBATCH --partition=__PARTITION__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# minilm/ study: MiniLM-L12 embeddings of both panels' prompts and of the four probe collections.
set -euo pipefail
export RB_RUNTIME_ROOT="${RB_RUNTIME_ROOT:-__PROJECT_SCRATCH__}" E4_ROOT="__E4_ROOT__"
export HF_HOME="$RB_RUNTIME_ROOT/huggingface-cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS=8
PY="$RB_RUNTIME_ROOT/venvs/env/bin/python"; cd "$E4_ROOT"
$PY scripts/embed_gpu.py --panel routerbench-0shot --text-field prompt --models minilm-l12
$PY scripts/embed_gpu.py --panel llmrouterbench-performance-cost --text-field prompt --models minilm-l12
export E4_REP=computed/minilm-l12@prompt
$PY results/hard_examples/embed_hard_examples.py
$PY results/format_invariance/embed_format_invariance.py
$PY results/technique_twins/embed_technique_twins.py
$PY results/deformatting/build_and_embed_core.py
ls -la features/*/prompt/minilm-l12.npy results/*/embeddings/minilm-l12__*.npy results/deformatting/panel/minilm-l12.npy
