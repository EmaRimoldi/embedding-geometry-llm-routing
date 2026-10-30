#!/bin/bash
#SBATCH --job-name=paper-proof-twins
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
source "$PWD/jobs/env.sh"
"$PAPER_PYTHON" -u src/diagnostics/proof_twins.py --preflight
"$PAPER_PYTHON" -u src/diagnostics/proof_twins.py --extract gte-qwen2-7b
"$PAPER_PYTHON" -u src/diagnostics/proof_twins.py --extract qwen2.5-7b-instruct
"$PAPER_PYTHON" -u src/diagnostics/proof_twins.py --report
