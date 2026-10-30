# Repository map and retention policy

The repository root has two main directories: [experiments/](../../experiments/README.md)
for the scientific workspace and [docs/](../README.md) for the manual. Its four
visible entry files are `README.md`, `Makefile`, `CITATION.cff` and `LICENSE`.
Contribution instructions and third-party notices are in
[docs/CONTRIBUTING.md](../CONTRIBUTING.md) and [docs/THIRD_PARTY.md](../THIRD_PARTY.md).

Scientific paths below, including `src/`, `configs/` and `results/`, are relative
to `experiments/`. Top-level Make targets invoke the appropriate tools without
requiring the reader to navigate those internal folders.

## Scientific implementation

| Location | Responsibility |
| --- | --- |
| [src/baselines/](../../experiments/src/baselines) | kNN, Family Adaptation, clustering, Ridge, MLP and baseline reports |
| [src/nvidia_utility/](../../experiments/src/nvidia_utility) | Final per-target layer selection, utility training and independent audit |
| [src/nvidia/](../../experiments/src/nvidia) | Shared PCA preprocessing and earlier common-layer/fixed-L21 controls |
| [src/layer_sweep/](../../experiments/src/layer_sweep) | Full-layer routing and layer-selection analysis |
| [src/diagnostics/](../../experiments/src/diagnostics) | Controlled-format, lookup, transfer and uncertainty analyses |
| [src/geometry.py](../../experiments/src/geometry.py) and [src/report_geometry.py](../../experiments/src/report_geometry.py) | Geometry measurements and numerical aggregation |
| [configs/](../../experiments/configs) | Frozen scientific protocols and canonical row splits |

The final per-target utility pipeline and the earlier NVIDIA controls have
different selection protocols; their outputs are not interchangeable.

## Documentation layout

| Directory | Content |
| --- | --- |
| [docs/guides/](../guides) | Setup, execution, input preparation and resource planning |
| [docs/reference/](.) | Methods, experiment/result maps, table production and this repository map |
| [docs/validation/](../validation) | Current audit, conceptual limitations and historical reviewer checks |
| [docs/releases/](../releases) | Published-release scope, archival and new-version instructions |
| [docs/demo/](../demo/README.md) | Presentation assets, source provenance and rebuild instructions |

## Versioned and generated material

Locations in this table are relative to the repository root.

| Location | Purpose | Versioned? |
| --- | --- | --- |
| [README.md](../../README.md), `LICENSE`, `CITATION.cff`, `Makefile` | Root overview, license, citation and convenience commands | Yes |
| [CONTRIBUTING.md](../CONTRIBUTING.md), [THIRD_PARTY.md](../THIRD_PARTY.md), `.gitignore` | Contribution guidance, attribution and runtime exclusions | Yes |
| `experiments/src/` | Scientific implementations and numerical report producers | Yes |
| `experiments/configs/` | Frozen protocols, hyperparameters, row splits and paper references | Yes |
| `experiments/scripts/` | Setup, integrity checks, fresh CPU diagnostics and export | Yes |
| `experiments/jobs/` | Portable Slurm launchers with original resource requests | Yes |
| `experiments/tests/` | Regression and distribution checks | Yes |
| `experiments/requirements.lock`, `experiments/environment/` | Compute, CPU, download and optional demo environments | Yes |
| `experiments/manifests/` | Pinned upstream assets, weight fingerprints and license provenance | Yes |
| `experiments/datasets/` | Raw and prepared public datasets | Downloaded locally; only the four proof-example prompts are included |
| `experiments/results/evidence/` | Frozen per-fold scores, numerical summaries and original audit reports | Yes |
| `experiments/results/release_evidence_manifest.json` | Fingerprints and provenance of frozen evidence | Yes |
| `experiments/results/reproduced/` | Newly reconstructed reports and CPU diagnostic summaries | No |
| Other `experiments/results/` subdirectories | New full-workflow predictions, reports and bootstrap arrays | No |
| `docs/` | Indexed guides, references, validation, releases and presentation assets | Yes |
| `experiments/release_provenance/` | Original fingerprints, release manifest and compact check records | Yes |
| `.github/workflows/` | CPU CI | Yes |
| `.zenodo.json` | Metadata prepared for later archival | Yes; its presence does not create a DOI |
| `experiments/embedders/`, `experiments/artifacts/` | Downloaded weights, generated embeddings/PCA and checkpoints | No |
| `experiments/.venv*`, `experiments/.cache/`, `experiments/.setup/`, `experiments/logs/`, `.cache/`, `dist/` | Local environments, temporary workspaces, setup state, logs and built ZIP | No |

The internal directory names used by scientific code are retained within
`experiments/` so input fingerprints,
launch commands and historical evidence paths remain traceable. Downloaded inputs
and new outputs stay separate from the frozen evidence.

Keep small configuration files, split indices, source dependencies and reference
scores in Git. Distribute large reusable arrays through a versioned archive with
checksums when available. Temporary fixtures and caches can be removed after their
checks finish. Recreate them from the included commands when needed.

The deterministic export includes only the allowlisted files in
`experiments/scripts/verify_release.py`; adding a new root file may require updating that list.
`make manifest` refreshes and checks the current manifest without changing a ZIP.
`make package` additionally builds a new archive and is reserved for intentional
new-version packaging. The Git index and release-manifest file set must agree
before tagging a release. Published tags and release assets remain historical.
