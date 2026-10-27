# Repository map and retention policy

| Location | Purpose | Versioned? |
| --- | --- | --- |
| `README.md`, `LICENSE`, `CITATION.cff`, `THIRD_PARTY.md` | Introduction, reuse terms, citation and attribution | Yes |
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
| `docs/` | Data guide, experiment map, methods, audit and presentation assets | Yes |
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
`make package` refreshes the manifest. The Git index and release-manifest file set
must agree before tagging a release.
