# Routing Theory: experiment reproduction

This repository now uses the final conference supplement (archive SHA-256
`043676877fed7b9ecf4f93da0ba428a5f8dbd0025212873fcebc94470623494e`). The obsolete
September export was removed from the active tree; Git history preserves it.
The original conference file manifest is retained separately from the current
repository manifest, so original results and later code fixes remain distinguishable.

## Checks added to this repository

```bash
python scripts/verify_release.py
python scripts/check_paper_results.py
```

The paper check reconstructs saved aggregates and verifies **5,493 numeric values
and settings covering all 44 tables** against the pinned final manuscript snapshot.
It checks printed precision, full penalty grids and known-metadata/oracle labels.
Supplied confidence endpoints are compared, not resampled by this compact command.

Fresh public data preparation has also been rerun: all eight output-file hashes
match the canonical cohorts (34,771 RouterBench and 43,887 SPROUT queries).
CPU and CUDA component checks passed, including actual encoder extraction on four
illustrative prompts and SharedTrunkNet toy fitting. These checks **do not constitute
a repeat of every full-scale model fit or confidence-interval bootstrap**. See
`docs/REPRODUCTION_AUDIT.md` for exactly what ran, fixes and remaining compute.

For CPU development on Python 3.11, create a virtual environment. On Linux,
install the CPU PyTorch wheel before the remaining pinned dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environment/cpu-requirements.lock
python -B -m pytest -q tests
```

On macOS, install `torch==2.5.1` from the ordinary package index instead of the
CPU wheel index. Full Linux/CUDA reproduction retains `requirements.lock`.

To independently download, checksum and prepare public data and tokenizers without
downloading the large model weights:

```bash
python -B scripts/check_public_inputs.py --download --prepare
python -B scripts/reproduce_outcome_controls.py
```

The latter recomputes all 598 single-model/oracle control scores from the prepared
outcomes and checks 3,588 saved metric values. Use a fresh reproduction copy for
preparation. Full model extraction still requires all model weights; after the
production asset download, verify their pinned LFS checksums with
`python scripts/verify_model_weights.py`.

This package contains the source, frozen configurations and compact evidence for the routing study. It supports two separate workflows:

1. **Check the supplied evidence and regenerate numerical summary tables**, without models, datasets or a GPU.
2. **Reproduce the experiments from public assets**, using the pinned revisions and the full execution instructions.

## Quick evidence check

From this directory, with Python 3.11:

```bash
python scripts/verify_release.py
python scripts/reproduce_results.py
```

The second command reconstructs the supported aggregate tables from the bundled records and writes derived outputs to `results/reproduced/`. Read `docs/RESULT_INDEX.md` for the exact coverage and distinction between reconstructed results and supplied summary-only evidence. It does not retrain routers or independently regenerate all predictions.

## Render saved appendix tables

With the compute dependencies installed:

```bash
python -B scripts/render_saved_tables.py
```

This optional command regenerates four table families in isolated output folders.
See `docs/SAVED_TABLES.md`. It does not recompute bootstrap intervals.

## Full experiment reproduction

Read `docs/RUN_COMMANDS.md` and `environment/`. Public dataset/model revisions are in `manifests/assets.json`. Frozen settings and row splits are in `configs/`.

The full pipeline requires Linux, the pinned Python environment, CUDA-capable hardware for encoder extraction and neural router fitting, substantial storage, and either configurable Slurm launchers or the documented Python commands. No GPUs are needed for the compact evidence check. Downloaded model weights, raw datasets, all fitted checkpoints and large embedding arrays are deliberately omitted.

## Methods and results

- The primary representations are GTE-Qwen2-7B-Instruct and raw Qwen2.5-7B-Instruct hidden state 21; GTE internal-layer and Qwen layer-sweep controls are separate.
- kNN-100 and validated kNN are separate methods. The fixed cost-penalty grids, splits and all reported outcomes are retained.
- The final NVIDIA utility adaptation is `src/nvidia_utility/` with `configs/nvidia_utility_v1.json`. It uses per-target/per-penalty Ridge validation and the reference ternary layer search, followed by SharedTrunkNet ensembles.
- `src/nvidia/` retains the earlier common-layer and fixed-layer utility controls. Their CPU inference measurements apply to those controls, not to the final per-target pipeline.
- Read `docs/METHODS.md` and `docs/RESULT_INDEX.md` before interpreting a comparison.

## Package contents

`src/` source; `configs/` frozen protocols/splits; `jobs/` configurable Slurm examples; `tests/` synthetic tests; `environment/` dependency specifications; `manifests/` public asset revisions; `results/` compact evidence; `release_provenance/` checksums; `scripts/` installation, verification and reproduction helpers.

Historical scheduling logs, private research notes, machine-specific caches and obsolete manuscript drafts are excluded. This code supplement does not contain or replace the submitted paper PDF.

## Updating the repository

After changing code, configuration or documentation, run the affected tests and
the paper-value check, then refresh the current file manifest and code archive:

```bash
python scripts/build_submission.py
python scripts/verify_release.py
```

Commit the regenerated manifest with the changes. The deterministic export is
`dist/routing_reproduction.zip`. Original conference fingerprints and saved
experimental evidence remain separate from newly executed results.

## Provenance and limitations

Original-run source hashes and release hashes are distinct. Portability and anonymization changes are described in `docs/RELEASE_CHANGES.md`. Published numerical outcomes are not retuned or selected by this packaging step. Fresh runs produce their own provenance; do not relabel historical results as having been generated by the release sources.

The release verification report describes exactly which checks ran. A successful compact evidence check is not a claim that every GPU experiment was rerun. See `docs/VALIDATION.md`.

Third-party code notices are retained. See `THIRD_PARTY.md` for asset and licensing information. The package does not assert a new license grant over third-party assets or its authors' original code.

## Reviewer smoke tests

See [reviewer smoke tests](docs/REVIEWER_SMOKE_TESTS.md) for small CPU/GPU execution checks, installation evidence and their scope. These runs do not replace the saved paper evidence.
