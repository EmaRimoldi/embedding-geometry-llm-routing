# Release validation

## Development checks

Run commands from the repository root. After code changes, run `make test` and the relevant synthetic component checks in the pinned compute or CPU environment described in the [setup guide](../guides/RUN_COMMANDS.md#platform-and-setup). `COMPUTE_PYTHON` can select another installed interpreter.

Before committing:

```bash
make check-docs
make manifest
make verify
git diff --check
```

Commit edited files and the current export manifest together. `make manifest` refreshes file fingerprints without modifying a ZIP. Reserve `make package` for intentional new versions, following the [release guide](../releases/RELEASE.md). Preserve earlier tags, release assets and historical evidence.

For demonstrated bugs, add a meaningful regression and explain effects on data selection, splits, seeds, penalties, aggregation or uncertainty. New runs belong in ignored runtime directories with new provenance. Report issues with the commit, command, environment, relevant input hashes, expected result and observed result; remove credentials and machine-specific details from logs.

## Executed reviewer checks

The original ZIP was extracted to a fresh directory. New compute and download environments were installed from public PyPI with Python 3.11.16; neither used the original environment's site packages. Both `pip check` commands passed.

| Check | Observed result |
|---|---|
| Public downloads | Seven pinned assets, 62 selected files verified; model weights excluded from this download test |
| Complete data preparation | RouterBench 34,771 and SPROUT 43,887 rows; eight output hashes and sixteen panel arrays exactly match frozen data |
| Geometry/pipeline tests | 10 passed, including small-pool self-exclusion cases |
| Real baseline CLI paths on synthetic IID/OOD data | 156 reconstructed scores across all 13 RouterBench penalties |
| MLP / direct-utility Ridge equivalence | 16 / 36 cases passed |
| Final utility NVIDIA smoke, CPU and CUDA | 20 trained networks and 20 checkpoint reloads per device; Ridge, search, seed retention and cost-once checks passed |
| Canonical encoder extraction on one L40S | MiniLM, GTE and Qwen layers 21/23/25; four supplied prompts each; finite vectors and matching hashes |
| Orchestration | 17 categories passed, including 59 launchers with inert scheduler commands |
| PCA100, full-SVD and randomized solvers | 10 checks each; held-out poisoning changes no source transforms; zero reconstruction error |
| Source preservation | All 182 original source/configuration/document fingerprints, sizes and modification times unchanged |

The GPU allocation completed in 42 seconds. The combined CPU baseline/NVIDIA/orchestration allocation completed in 52 seconds. Public-data preparation and exact comparison completed in 89 seconds. These are smoke allocation durations, not full-study compute or comparable router latency.

## Compact evidence and tables

The compact route checks 1,758 evidence files and reconstructs 23,740 saved fold records into 4,031 aggregate rows. Utility identities and saved aggregate comparisons use relative tolerance 1e-10 and absolute tolerance 1e-12. Four saved-report renderer families produce seven LaTeX tables, previously checked byte for byte against the supplied tables.

Main-comparison outputs now distinguish ordinary routers, known-metadata diagnostics and realized-oracle controls explicitly. All 1,300 penalty rows and 100 mean rows retain the previous fields, ordering and numerical values.

## Scope and caveats

These checks include real downloads, preparation, encoding and small fits. They do not repeat all full-scale GPU experiments, independently recompute every saved per-query prediction, or repeat bootstrap resampling from omitted arrays. The GPU extraction test reused pinned cached weights; complete model-weight downloading was not repeated. The synthetic NVIDIA smoke executes the actual fitting core and assembly, with fixture adapters; it does not test the entire production worker loop.

Read [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md) for conceptual limitations and the optional family-rank interval tie-rule inconsistency. Static identity/credential-pattern scans and checksums are useful checks, not a guarantee about every indirect identifier or all scientific correctness.

Commands and detailed coverage are in [REVIEWER_SMOKE_TESTS.md](REVIEWER_SMOKE_TESTS.md), [REVIEWER_NVIDIA_AUDIT.md](REVIEWER_NVIDIA_AUDIT.md), and [REVIEWER_SMOKE_ORCHESTRATION.md](REVIEWER_SMOKE_ORCHESTRATION.md). Machine-readable evidence is under `reproducibility/provenance/reviewer_smoke/`. Exact exported bytes are listed in `reproducibility/provenance/release_manifest.json`; the archive checksum is supplied beside the ZIP.
