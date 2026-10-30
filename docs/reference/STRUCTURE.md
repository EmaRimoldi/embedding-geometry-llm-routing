# Repository structure and retention

The root uses familiar research-software names and separates responsibilities. All commands and paths in the current manuals are relative to the repository root.

## Primary directories

| Directory | Responsibility | Start here |
| --- | --- | --- |
| [src/](../../src/) | Scientific algorithms, data processing and numerical report producers | [Experiment map](EXPERIMENTS.md) |
| [configs/](../../configs/) | Fixed protocols, hyperparameters, row splits and manuscript references | [Methods](METHODS.md) |
| [scripts/](../../scripts/) | Setup, validation, comparison and distribution tooling | [Execution guide](../guides/RUN_COMMANDS.md) |
| [jobs/](../../jobs/) | Slurm resource declarations and launchers | [Slurm conventions](../guides/RUN_COMMANDS.md#slurm-conventions) |
| [tests/](../../tests/) | Regression, component and distribution checks | [Development checks](../validation/VALIDATION.md#development-checks) |
| [docs/](../README.md) | Setup guides, scientific reference, audits and release instructions | [Manual index](../README.md) |
| [reproducibility/](../../reproducibility/README.md) | Environments, pinned manifests, frozen evidence and provenance | [Reproducibility index](../../reproducibility/README.md) |

This follows the separation of source, tests, documentation and automation described in the [Good Research Code Handbook](https://goodresearch.dev/setup.html#create-a-project-skeleton) and [pyOpenSci structure guide](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html). These are conventions to adapt to a project, not a mandatory universal folder count. This repository is a research workflow with direct Python and Slurm entrypoints.

## Source map

| Location | Implementation |
| --- | --- |
| [src/baselines/](../../src/baselines/) | kNN, Family Adaptation, clustering, Ridge, MLP and baseline reports |
| [src/nvidia_utility/](../../src/nvidia_utility/) | Final per-target layer selection, utility training and independent audit |
| [src/nvidia/](../../src/nvidia/) | Shared PCA preprocessing and earlier common-layer/fixed-L21 controls |
| [src/layer_sweep/](../../src/layer_sweep/) | Full-layer routing and layer-selection analysis |
| [src/diagnostics/](../../src/diagnostics/) | Controlled-format, lookup, transfer and uncertainty analyses |
| [src/geometry.py](../../src/geometry.py) and [src/report_geometry.py](../../src/report_geometry.py) | Geometry measurements and numerical aggregation |

The final utility pipeline and the earlier NVIDIA controls use different selection protocols. Their outputs are not interchangeable.

## Root files

[README.md](../../README.md) is the project entry point and ends with the paper citation. [requirements.lock](../../requirements.lock) pins compute dependencies; the [Makefile](../../Makefile) provides root commands. [LICENSE](../../LICENSE) exposes licensing terms. Development instructions are in the [validation guide](../validation/VALIDATION.md#development-checks), and third-party notices are in [docs/THIRD_PARTY.md](../THIRD_PARTY.md).

The hidden `.github/` directory contains CI and issue templates. `.gitignore` excludes downloaded data, model weights, environments, caches and generated results from Git.

## Retention policy

| Material | Location | Tracked? |
| --- | --- | --- |
| Scientific source and protocols | `src/`, `configs/` | Yes |
| Automation and tests | `scripts/`, `jobs/`, `tests/` | Yes |
| Documentation and presentation assets | `docs/` | Yes |
| Pinned environment and asset records | `reproducibility/environment/`, `reproducibility/manifests/` | Yes |
| Frozen original evidence | `reproducibility/evidence/`, `reproducibility/evidence_manifest.json` | Yes; preserved historical bytes |
| Source identities and scoped checks | `reproducibility/provenance/` | Yes |
| Four illustrative prompts | `reproducibility/examples/proof_twins.jsonl` | Yes |
| Raw and prepared benchmark data | `datasets/` | No; download and prepare from pinned public assets |
| Model weights | `embedders/` | No |
| Embeddings, PCA arrays and checkpoints | `artifacts/` | No |
| New scores, predictions, bootstraps and reconstructed reports | `results/` | No |
| Environments, caches, setup state and logs | `.venv*`, `.cache/`, `.setup/`, `logs/` | No |
| Built distribution ZIPs | `dist/` | No; published versions are retained separately |

The [execution guide](../guides/RUN_COMMANDS.md) lists producers and prerequisites for runtime paths. The initializer materializes the bundled illustrative prompts in the runtime dataset directory and refuses to replace changed bytes. New model execution never imports frozen scores as measurements.

Historical manifests retain the original paths and hashes. Current verification maps original `results/evidence/` names to `reproducibility/evidence/`; the current export manifest records this checkout's paths. Source-path changes are recorded separately from frozen outputs. Earlier tags and release assets remain unchanged.

## Validation and distribution

`make check-docs` checks local links, anchors, reachability and versioned command paths. CI also runs regression tests, synthetic baseline/NVIDIA/PCA/orchestration checks, and numerical table reconstruction. The [export allowlist](../../scripts/verify_release.py) excludes generated data and weights.

Run `make manifest` after changes to refresh current file fingerprints, then `make verify`. `make manifest` does not alter a ZIP. Reserve `make package` for an intentional new release, following the [release guide](../releases/RELEASE.md); require the Git index and current export manifest to contain the same files.

The [layout validation record](../../reproducibility/provenance/current_checks/repository_layout.json) records the earlier directory reorganization, clean-export checks and source-path adaptations.
