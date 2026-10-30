# Scientific workspace

This directory contains the study's code, fixed protocols, launchers, tests and compact evidence. The [repository README](../README.md) and [manual](../docs/README.md) are the main entry points.

## Choose a working directory

From the repository root, use `make verify`, `make test` or `make smoke` as convenience commands. For the scientific command examples and Slurm jobs, enter this workspace first:

```bash
cd experiments
```

Paths such as `src/`, `configs/`, `jobs/`, `datasets/`, `results/` and `release_provenance/` in the scientific manuals are relative to this directory. This keeps the internal scientific layout and frozen input identities consistent.

## Workspace map

| Directory | Content |
| --- | --- |
| [src/](src/) | Scientific implementations and numerical report producers |
| [configs/](configs/) | Fixed protocols, row splits and paper references |
| [jobs/](jobs/) | Slurm launchers; submit them from this workspace |
| [scripts/](scripts/) | Setup, checks, comparison, reporting and packaging tools |
| [tests/](tests/) | Regression and component tests |
| [environment/](environment/) and [requirements.lock](requirements.lock) | Pinned dependency environments and runtime provenance paths |
| [manifests/](manifests/) | Public asset revisions, weight checksums and license records |
| [results/evidence/](results/evidence/) | Frozen original scores, summaries and audit evidence |
| [release_provenance/](release_provenance/) | Original fingerprints, current export manifest and scoped checks |

Downloaded datasets, weights, generated embeddings, model checkpoints and new result directories are local runtime material and stay outside Git. The [execution guide](../docs/guides/RUN_COMMANDS.md) gives their producers and prerequisites; the [retention policy](../docs/reference/STRUCTURE.md) lists their locations.

The top-level license and machine-readable citation are [LICENSE](../LICENSE) and [CITATION.cff](../CITATION.cff).
