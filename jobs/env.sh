# Source from bash. Derive the root from this copy, never from an inherited path.
PAPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PAPER_ROOT
[[ -f "$PAPER_ROOT/configs/geometry.json" ]] || { echo "Invalid PAPER_ROOT" >&2; return 1; }
export HF_HOME="$PAPER_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_XET_CACHE="$HF_HOME/xet"
export XDG_CACHE_HOME="$PAPER_ROOT/.cache"
export TMPDIR="$PAPER_ROOT/.cache/tmp"
export MPLCONFIGDIR="$PAPER_ROOT/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$PAPER_ROOT/.cache/pycache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME" "$PAPER_ROOT/logs"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PAPER_PYTHON="${PAPER_PYTHON:-$PAPER_ROOT/.venv/bin/python}"
cd "$PAPER_ROOT"
