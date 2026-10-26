#!/bin/bash
#SBATCH --job-name=astra-geo-token-audit
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/presentation-token-%j.out
#SBATCH --error=logs/presentation-token-%j.err
set -euo pipefail
export E4_ROOT=__E4_ROOT__
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE="$PWD/real_presentation/cache/token-audit" XDG_CACHE_HOME="$PWD/real_presentation/cache/token-audit-xdg"
python -u presentation_token_audit.py
