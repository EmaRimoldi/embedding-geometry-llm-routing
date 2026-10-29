# Repository map and retention policy

The [documentation index](../README.md) is the manual's entry point. The repository
separates scientific code and fixed inputs, executable tooling, reference evidence,
and generated runtime material.

## Scientific implementation

| Location | Responsibility |
| --- | --- |
| [src/baselines/](../../src/baselines/) | kNN, Family Adaptation, clustering, Ridge, MLP and baseline reports |
| [src/nvidia_utility/](../../src/nvidia_utility/) | Final per-target layer selection, utility training and independent audit |
| [src/nvidia/](../../src/nvidia/) | Shared PCA preprocessing and earlier common-layer/fixed-L21 controls |
| [src/layer_sweep/](../../src/layer_sweep/) | Full-layer routing and layer-selection analysis |
| [src/diagnostics/](../../src/diagnostics/) | Controlled-format, lookup, transfer and uncertainty analyses |
| [src/geometry.py](../../src/geometry.py) and [src/report_geometry.py](../../src/report_geometry.py) | Geometry measurements and numerical aggregation |
| [configs/](../../configs/) | Frozen scientific protocols and canonical row splits |

The final per-target utility pipeline and the earlier NVIDIA controls have
different selection protocols; their outputs are not interchangeable.

## Documentation layout

| Directory | Content |
| --- | --- |
| [docs/guides/](../guides/) | Setup, execution, input preparation and resource planning |
| [docs/reference/](./) | Methods, experiment/result maps, table production and this repository map |
| [docs/validation/](../validation/) | Current audit, conceptual limitations and historical reviewer checks |
| [docs/releases/](../releases/) | Published-release scope, archival and new-version instructions |
| [docs/demo/](../demo/README.md) | Presentation assets, source provenance and rebuild instructions |

## Versioned and generated material

| Location | Purpose | Versioned? |
| --- | --- | --- |
| [README.md](../../README.md), `LICENSE`, `CITATION.cff`, `THIRD_PARTY.md` | Introduction, reuse terms, citation and attribution | Yes |
| `Makefile`, `CONTRIBUTING.md`, `.gitignore` | Convenience commands, issue/change guidance and exclusions | Yes |
| `src/` | Scientific implementations and numerical report producers | Yes |
| `configs/` | Frozen protocols, hyperparameters, row splits and paper references | Yes |
| `scripts/` | Setup, integrity checks, fresh CPU diagnostics and export | Yes |
| `jobs/` | Portable Slurm launchers with original resource requests | Yes |
| `tests/` | Regression and distribution checks | Yes |
| `requirements.lock`, `environment/` | Compute, CPU, download and optional demo environments | Yes |
| `manifests/` | Pinned upstream assets, weight fingerprints and license provenance | Yes |
| `datasets/` | Raw and prepared public datasets | Downloaded locally; only the four proof-example prompts are included |
| `results/evidence/` | Frozen per-fold scores, numerical summaries and original audit reports | Yes |
| `results/release_evidence_manifest.json` | Fingerprints and provenance of frozen evidence | Yes |
| `results/reproduced/` | Newly reconstructed reports and CPU diagnostic summaries | No |
| Other `results/` subdirectories | New full-workflow predictions, reports and bootstrap arrays | No |
| `docs/` | Indexed guides, references, validation, releases and presentation assets | Yes |
| `release_provenance/` | Original fingerprints, release manifest and compact check records | Yes |
| `.github/workflows/` | CPU CI | Yes |
| `.zenodo.json` | Metadata prepared for later archival | Yes; its presence does not create a DOI |
| `embedders/`, `artifacts/` | Downloaded weights, generated embeddings/PCA and checkpoints | No |
| `.venv*`, `.cache/`, `.setup/`, `logs/`, `dist/` | Local environments, temporary workspaces, setup state, logs and built ZIP | No |

The directory names used by scientific code are retained so input fingerprints,
launch commands and historical evidence paths remain traceable. Downloaded inputs
and new outputs stay separate from the frozen evidence.

Keep small configuration files, split indices, source dependencies and reference
scores in Git. Distribute large reusable arrays through a versioned archive with
checksums when available. Temporary fixtures and caches can be removed after their
checks finish. Recreate them from the included commands when needed.

The deterministic export includes only the allowlisted files in
`scripts/verify_release.py`; adding a new root file may require updating that list.
`make manifest` refreshes and checks the current manifest without changing a ZIP.
`make package` additionally builds a new archive and is reserved for intentional
new-version packaging. The Git index and release-manifest file set must agree
before tagging a release. Published tags and release assets remain historical.
