# Render tables from saved reports

Commands for the current checkout run from the [scientific workspace](../../experiments/README.md). From the repository root, enter it with `cd experiments`.

From the scientific workspace, using its Python environment:

```bash
python -B scripts/render_saved_tables.py
```

This optional command renders four selected table families from the bundled reports. It checks that each renderer finishes and that all seven numerical table outputs match the frozen evidence byte-for-byte. A nonempty but changed output fails verification. It does not recompute predictions, bootstrap intervals, training results, or all figures in the paper. NumPy is needed by the appendix table helper; the other renderers use the Python standard library.

Each invocation creates a fresh workspace under `results/reproduced/saved_tables/run-*/`. It copies only the listed renderers, their direct helpers/configurations, and the required saved report inputs into that workspace. Subprocesses use the current interpreter with `-B`; their caches and temporary files stay in the workspace. Source files, configurations, bundled evidence, and previous rendering runs are left intact.

| Family | Bundled input coverage | Produced tables |
|---|---|---|
| `appendix_ci` | Routing reports for both benchmarks and IID/OOD; geometry reports for both; baseline and interval settings | Main appendix interval macros and complete penalty tables, plus audit JSON |
| `sharedtrunknet` | SharedTrunkNet interval report; audited layer selections; both frozen split configurations | SharedTrunkNet comparison and layer-selection tables |
| `cost_reward` | Cost/reward/format report and completion metadata for both benchmarks | Cost/reward/format diagnostic tables |
| `family_utility` | Family-utility audit, summary, and per-penalty CSV | Source-family retrieval rank tables |

Inputs retain their original result-relative locations under `results/evidence/`. In the temporary workspace, they are staged below `results/` so the existing renderer source can run unchanged. Expected output paths and every staged input are recorded in the run's `report.json`, together with SHA-256 hashes and log locations.

Statuses are explicit: `passed` means rendering succeeded; `skipped` means no evidence for a family was bundled; `missing` means only part of the required input set was available; `failed` means a renderer failed or omitted an output. A missing or failed family returns a nonzero exit code. A skipped family is not a validation success. The report does not certify the completeness of the paper or correctness of the underlying experiment.

The four families need about 25 MB of staged input, mostly the two frozen split configurations. The manuscript-only figure builder has been removed. Generated LaTeX retains the renderer's original labels and formatting; compilation is outside the experimental reproduction requirements.
