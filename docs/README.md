# Documentation

Start at the [repository README](../README.md) for the paper, quickstart and citation. Top-level `make` commands run from the repository root. Scientific commands run from `experiments/` (`cd experiments`); bare paths such as `src/`, `jobs/` and `results/` in scientific manuals are relative to that workspace.

## Guides: run the workflow

| Document | Use it to |
| --- | --- |
| [Execution guide](guides/RUN_COMMANDS.md) | Set up environments and follow preparation, extraction, fitting, reporting and bootstrap stages in order |
| [Data and model access](guides/DATA.md) | Obtain pinned public inputs, verify hashes and understand the canonical cohorts |
| [Resources](guides/RESOURCES.md) | Plan CPU/GPU allocations, memory, storage and timing interpretation |

## Reference: understand the material

| Document | Use it to |
| --- | --- |
| [Experiment map](reference/EXPERIMENTS.md) | Map manuscript tables to producers, prerequisites, commands and outputs |
| [Methods](reference/METHODS.md) | Understand representations, splits, routing objectives and evaluation |
| [Result index](reference/RESULT_INDEX.md) | Locate included evidence and the inputs needed for fresh results |
| [Saved numerical tables](reference/SAVED_TABLES.md) | Regenerate the supplied tables from frozen reports |
| [Repository map](reference/STRUCTURE.md) | Locate implementations and distinguish versioned material from generated runtime files |

## Validation: know what has been checked

| Document | Scope |
| --- | --- |
| [Reproduction audit](validation/REPRODUCTION_AUDIT.md) | Current executed checks, corrections, observed differences and incomplete coverage |
| [Conceptual audit](validation/CONCEPTUAL_AUDIT.md) | Statistical assumptions, information regimes and interpretation limits |
| [Reviewer validation summary](validation/VALIDATION.md) | Historical conference-package validation |
| [Reviewer smoke tests](validation/REVIEWER_SMOKE_TESTS.md) | Historical component, download and preparation checks |
| [Reviewer NVIDIA audit](validation/REVIEWER_NVIDIA_AUDIT.md) | Scope of the synthetic per-target utility check |
| [Reviewer orchestration check](validation/REVIEWER_SMOKE_ORCHESTRATION.md) | Provenance initialization and inert-scheduler checks |
| [Reviewer PCA check](validation/REVIEWER_SMOKE_PCA.md) | Full-SVD/randomized preprocessing and held-out poisoning checks |

The frozen numerical evidence is in [results/evidence](../experiments/results/evidence). Current scoped check records are in [release_provenance/current_checks](../experiments/release_provenance/current_checks); historical reviewer records are in [release_provenance/reviewer_smoke](../experiments/release_provenance/reviewer_smoke).

## Releases and presentation

| Document | Purpose |
| --- | --- |
| [Release and archival guide](releases/RELEASE.md) | Retrieve and verify published assets or intentionally package a new version |
| [v1.0.0 release notes](releases/RELEASE_v1.0.0.md) | Scope of the original published release |
| [Conference-package changes](releases/RELEASE_CHANGES.md) | Historical portability and packaging changes |
| [README demo](demo/README.md) | Sources and regeneration commands for the animation and comparison figures |

See [contributing](CONTRIBUTING.md) for the development workflow and [third-party notices](THIRD_PARTY.md) for attribution. `make check-docs` validates local links, heading anchors, documentation reachability and literal versioned paths used in the manual and job launchers.
