# Reviewer smoke tests

Run all commands in this guide from the repository root.

The current repository additionally executes Avengers-Pro and KM-Prox in the CPU
baseline harness (260 score checks). The 156-check counts below describe the
original conference reviewer run, whose reports are preserved unchanged. See
[REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md) for current executions.

These checks exercise installed software and small real execution paths. Synthetic outputs are explicitly labelled and never replace the paper evidence.

## Installation

From the repository root, run:

```bash
bash scripts/bootstrap.sh all
.venv/bin/python -m pip check
.venv-download/bin/python -m pip check
```

The release was tested with a newly created Python 3.11 environment and packages downloaded from public PyPI, without access to the original environment's site packages. Both dependency checks passed. The system still needs a compatible Python interpreter, network access for installation, and appropriate compute resources.

## CPU checks

Run from the repository root, inside a CPU allocation on clusters:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -B -m pytest -q tests
.venv/bin/python -B scripts/smoke_baselines.py
.venv/bin/python -B scripts/smoke_nvidia.py
.venv/bin/python -B scripts/smoke_orchestration.py
```

- Geometry and pipeline tests: 10 passed, including the small-reference-pool self-exclusion regression.
- Baselines: actual kNN, Family Adaptation and Ridge CLI fit/predict/save paths on synthetic IID/OOD data; 156 independently reconstructed utility scores across all 13 RouterBench penalties, 16 MLP equivalence cases, and 36 direct-utility Ridge checks.
- NVIDIA: actual per-target probe, reference ternary search, fitting core, prediction, checkpoint reload and ensemble assembly. Twenty toy networks are trained across two penalties and ten seeds; the best five are independently verified. The toy run uses four epochs and is not a speed or performance estimate for the paper.
- Orchestration: 17 categories, including 59 launchers using inert scheduler/Python commands. This checks command construction without submitting 59 experiments.

Generated results live under `results/reproduced` or `.cache`, and are excluded from the release ZIP.

## PCA100 preprocessing

Run `.venv/bin/python -B scripts/smoke_pca.py` in a CPU allocation. The actual production PCA CLI passed ten checks on each of the full-SVD and randomized branches, always retaining 100 components. Poisoning held-out vectors left scaler/PCA arrays and source projections bit-identical; saved-transform reconstruction error was zero. Cache reuse and deliberate stale/checksum mismatch rejection passed. See [REVIEWER_SMOKE_PCA.md](REVIEWER_SMOKE_PCA.md).

## GPU check

With one allocated GPU and pinned model assets already downloaded:

```bash
.venv/bin/python -B scripts/smoke_encoders.py
.venv/bin/python -B scripts/smoke_nvidia.py --device cuda
```

An optional `--model-root /path/to/embedders` uses an existing read-only asset cache. The smoke script runs the unchanged encoder on the four shipped illustrative prompts. MiniLM, GTE, and raw Qwen layers 21/23/25 must all produce finite vectors with matching query, vector and code hashes.

This test passed on one NVIDIA L40S using newly installed dependencies and existing pinned weights. The GPU NVIDIA test additionally passed twenty fit/reload checks and verified unrestricted negative outputs and that changing held-out evaluation costs does not change routing decisions.

## Public download and complete preparation check

The actual downloader fetched and verified seven pinned assets (62 selected files). Dataset allowlists were unchanged; model-weight patterns were omitted in a private test fixture, while tokenizers and model code were downloaded publicly. Both actual dataset-preparation commands then ran in the fresh compute environment:

```bash
.venv/bin/python -B src/prepare_data.py routerbench
.venv/bin/python -B src/prepare_data.py sprout
```

RouterBench produced 34,771 rows (34,659 in the shared MiniLM subset); SPROUT produced 43,887 (41,324 shared). All eight output-file hashes and all sixteen panel arrays exactly matched the frozen canonical data, including dtypes. The public download smoke did not download model weights. The separate encoder smoke used pinned cached weights.

## Interpretation

These are working smoke paths, not a repeat of the full study. Full-scale training, all per-query audits and bootstrap resampling remain separate workflows. Read [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md) for statistical scope and the recorded optional rank-interval tie-rule inconsistency.
