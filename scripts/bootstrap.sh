#!/usr/bin/env bash
# Create local environments from the experiment's pinned Python 3.11 dependencies.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
# Keep installer caches and temporary files inside this release copy.
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$ROOT/.cache/tmp"
export PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"
# Do not let inherited pip destination settings bypass the local environments.
unset PIP_TARGET PIP_PREFIX PIP_USER PYTHONUSERBASE
export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
PYTHON="${PYTHON:-python3.11}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11 (tested: 3.11.16)"'
case "${1:-all}" in
  all|compute|download) mode="${1:-all}" ;;
  *) echo "Usage: bash scripts/bootstrap.sh [all|compute|download]" >&2; exit 2 ;;
esac
if [[ "$mode" == all || "$mode" == compute ]]; then
    [[ -d .venv ]] || "$PYTHON" -m venv .venv
    .venv/bin/python -m pip install -r requirements.lock
    .venv/bin/python -m pip check
fi
if [[ "$mode" == all || "$mode" == download ]]; then
    [[ -d .venv-download ]] || "$PYTHON" -m venv .venv-download
    .venv-download/bin/python -m pip install -r reproducibility/environment/download-requirements.lock
    .venv-download/bin/python -m pip check
fi
"$PYTHON" -B scripts/initialize_release.py --stage layout
