# v1.0.0 — Experiment code and reproducibility materials

This release connects the scientific implementations, pinned public inputs, frozen
splits and numerical evidence to runnable reproduction and verification commands.

- CC BY 4.0 for original project material, matching the paper's arXiv license;
  NVIDIA reference sources, fonts, datasets and models retain their upstream terms.
- A consolidated data guide, all-44-label experiment map, scheduler resource
  inventory, repository map and contribution guidance.
- `make verify`, `make controls`, `make diagnostics`, `make tables`, `make test`,
  `make smoke` and `make package` wrap the documented commands.
- Fresh complete production IID metadata lookups and the MMLU topic diagnostic,
  including its paired bootstrap, match all 24,209 compared numerical values
  with maximum absolute error zero.
- Saved evidence reconstruction covers 23,740 fold records and 4,031 aggregate rows;
  the paper-reference gate checks 5,493 quantities/settings across 44 table labels.
- Input integrity, estimator tie handling, safe initialization, table equality
  checks and production component paths have regression/CI coverage.
- The attached deterministic ZIP has a SHA-256 sidecar and an internal manifest.
  Citation and Zenodo metadata identify software version 1.0.0.

The numerical reference is Overleaf snapshot
`6e3bea88ec5d65de3587c0719025702e110ac3e0`. Full-scale router training, all 2,990
final SharedTrunkNet network fits and every bootstrap have not all been independently
rerun. CUDA tests to date cover components and illustrative encoder inputs. Large
datasets, weights, embeddings, PCA arrays, checkpoints and per-query bootstrap
arrays are downloaded or regenerated through the execution guide; this release
does not contain a separately published mirror of those arrays.

No Zenodo software DOI is claimed. The repository's existing access settings are
retained. See `docs/REPRODUCTION_AUDIT.md`, `docs/EXPERIMENTS.md`, `docs/DATA.md`
and `docs/RELEASE.md` in the tagged source for scope and instructions.
