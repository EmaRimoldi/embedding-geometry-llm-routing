# Fresh experiment reproduction

Commands for the current checkout run from the [scientific workspace](../../experiments/README.md). From the repository root, enter it with `cd experiments`.

These commands are for a **new reproduction copy**, separate from the saved evidence
under `results/evidence/`. They regenerate runtime results; they do not import the
saved paper scores as new measurements. The shipped evidence is immutable input to
the lightweight report route documented in the main README.

## Platform and setup

The study used Linux x86-64 and Python **3.11.16**. Use Python 3.11, with venv/pip
available. `requirements.lock` pins the compute environment, including PyTorch
2.5.1/CUDA 12.4 dependencies; `environment/download-requirements.lock` pins the
separate download CLI environment. GPU extraction uses CUDA, BF16 for the two 7B
encoders, and full untruncated prompts (up to 8192 tokens). MiniLM uses FP32 and a
separate compatible subset. Do not reduce token limits to make a smaller GPU fit.

From the scientific workspace:

```bash
bash scripts/bootstrap.sh
source jobs/env.sh
.venv-download/bin/python -B src/download_assets.py
.venv/bin/python -B src/prepare_data.py routerbench
.venv/bin/python -B src/prepare_data.py sprout
.venv/bin/python -B -m pytest -p no:cacheprovider tests
```

Downloads use the revisions and patterns in `manifests/assets.json` and verify
remote checksums. The downloader enables network access explicitly; compute
launchers default to offline model use. If an asset requires accepting an upstream
license, do so through the upstream provider before download. Never include access
tokens in the supplementary archive.

Expect tens of GB for model downloads alone and more than 100 GB for the complete
experiment archive. Resource declarations in `jobs/*.sbatch` retain the original
CPU count, host memory and walltime requests. Adapt queue/account/GPU-type settings
to the cluster, but preserve algorithms, task ranges, seeds and scientific settings.

## Slurm conventions

Run every `sbatch` from the scientific workspace (`experiments/`), after `source jobs/env.sh`.
Launchers use their working directory (Slurm's default submission directory), so
do not override `--chdir` to another directory. They do not trust a stale
`PAPER_ROOT` or inherited `SLURM_SUBMIT_DIR` from an older allocation. The release
uses `logs/%x-%A_%a.log`; the layout initializer creates `logs/` **before**
submission. Set local partition/account defaults via `SBATCH_PARTITION` and
`SBATCH_ACCOUNT` (or command-line options). Automatic worker continuations inherit
these environment variables. Example queue names are deliberately not hardcoded.

GPU jobs each request one GPU. **Do not have more than two active GPU tasks across
stages.** Submit stages sequentially after their prerequisites complete; a %2
throttle on two separate arrays would otherwise permit four simultaneous GPUs.
The commands below are a stage list, not a single script to paste asynchronously.
CPU stages may overlap only when their stated inputs are complete.

`PAPER_PYTHON` can override the compute interpreter. Run ordinary Python scripts
inside a scheduler allocation; never run heavy fits on a login node. The GPU
worker entrypoints use Slurm IDs, continuation submission and automatic audit
submission, so use their supplied Slurm launchers.

## 1. Canonical embeddings and geometry

```bash
sbatch jobs/encode.sbatch
# Wait for all three encoder tasks to finish; each task handles both datasets.
.venv/bin/python -B src/launch_geometry.py
sbatch jobs/geometry_batch.sbatch
# Wait for all 20 batches.
sbatch jobs/report.sbatch
sbatch jobs/plots.sbatch
```

`run_geometry_batch.py` covers task indices 0 through the end exactly once with
20 disjoint workers. Task 0 was previously executed outside that launcher; the
release includes it in the complete batch schedule.

The direct equivalent for geometry batch `i` is:
`python src/run_geometry_batch.py i`. Each batch invokes `src/run_task.py TASK`.
The configurations store the original outer populations, folds and seeds.

For the separately reported clustering analyses:
```bash
sbatch --array=0-17%4 jobs/descriptive_clustering.sbatch
sbatch jobs/additional_clustering.sbatch
```
Wait for geometry before these follow-up analyses; the additional-clustering array
requests GPUs and must share the same overall two-GPU limit.

## 2. Baselines

After prepared panels and canonical GTE/L21 embeddings are complete:

```bash
.venv/bin/python -B src/baselines/check.py
.venv/bin/python -B src/baselines/prepare.py
.venv/bin/python -B scripts/initialize_release.py --stage reduced
.venv/bin/python -B src/baselines/prepare_mlp_reduced.py
.venv/bin/python -B src/baselines/run_ridge.py --check
sbatch --array=0-45%4 jobs/baselines_cpu.sbatch
sbatch --array=0-22%4 jobs/baselines_ridge.sbatch
sbatch jobs/baselines_mlp_reduced.sbatch 0
sbatch jobs/baselines_mlp_reduced.sbatch 1
```

The 46 baseline tasks are panel/split/representation combinations. Ridge packs
pair the GTE and L21 representation tasks. Reduced MLP workers cover even/odd
indices and retain the approved two/three-hidden-layer, 100-unit architectures.
Do **not** launch `baselines_mlp.sbatch` for the paper's reduced MLP: that launcher
is the historical full-validation variant, retained for provenance.

`prepare.py` creates runtime `results/baselines/_run/source_hashes.json` from the
release sources. The reduced initializer updates only the copied config's source
hash metadata, then the original synthetic preflight verifies the method.
`run_ridge.py --check` creates `ridge_sources.json` after independent direct
utility-equivalence tests. These fresh hashes are distinct from original-run
hashes in saved evidence.

After all baseline jobs finish:
```bash
.venv/bin/python -B src/baselines/run_mlp_reduced.py --audit
.venv/bin/python -B src/baselines/report_reduced.py
.venv/bin/python -B src/baselines/report_ridge.py
.venv/bin/python -B scripts/submit_utility_report.py
```

Compare the newly generated main-baseline records with the pinned original
scores and selections:

```bash
python -B scripts/compare_baseline_results.py --fresh-results results
# Equivalent: make compare-baselines
```

This standard-library command covers 296 prescribed result records: the six main
method groups across all 46 tasks, plus 20 IID metadata-lookup reports. It compares
scores, candidate validation values, selected settings, cohort sizes and split
identities. Floats use the predeclared relative/absolute tolerances `1e-10`/`1e-12`;
counts, candidate grids and discrete selected settings, including KM-Prox inverse
temperature, must match exactly. Every nonidentical
compared value is retained, including differences within tolerance.

For a valid invocation, exit 0 requires complete coverage and matching scientific
fields; exit 2 means coverage remains incomplete; exit 1 reports differences or
invalid input files. Invalid command-line arguments also use the usual exit 2.
Registered source/config hash transitions remain explicit differences and are
recognized only when the new source bytes match the existing transition registry.
The default report is `results/reproduced/fresh-baselines.json`. Comparing supplied
files does not attest that new fitting occurred, compare omitted query-level
predictions, or establish paper printed-precision agreement. Keep the new run's
execution provenance separately. The command refuses to use `results/evidence/`
as the fresh results directory.

## 3. Candidate layers and PCA

The earlier common-layer NVIDIA implementation also provides the shared
fold-specific PCA cache needed by the final per-target utility implementation.
Initialize it before extracting missing candidate layers:

```bash
.venv/bin/python -B scripts/initialize_release.py --stage nvidia
.venv/bin/python -B src/nvidia/prepare.py
sbatch --array=0-1%2 jobs/nvidia_extract.sbatch
# Wait for both datasets' extraction.
sbatch --array=0-27%4 jobs/nvidia_pca.sbatch
```

The canonical extraction reuses already completed L21/L23/L25 vectors.
PCA100 is fitted separately on the frozen source subsets; its component count
does not eliminate the need for those separate fits. The original preflight
generates `results/nvidia/provenance/source_hashes.json`. Its new config hash
records the release-only provenance update and is used by newly generated PCA
cache identities.

## 4. Final per-target NVIDIA utility pipeline

After all required PCA tasks finish:

```bash
.venv/bin/python -B scripts/initialize_release.py --stage utility
sbatch jobs/nvidia_utility_preflight.sbatch
# Wait for passed preflight.
sbatch --array=0-22%4 jobs/nvidia_utility_probe.sbatch
# Wait for all 23 source-only probe selections.
sbatch jobs/nvidia_utility_train.sbatch 0
sbatch jobs/nvidia_utility_train.sbatch 1
```

Both workers finishing triggers the independent audit array. To submit
the audit explicitly after complete training:
`sbatch jobs/nvidia_utility_audit.sbatch`.

The aggregate comparison waits for both the utility audit array and the complete
baseline report (`_summary_reduced/status.json` and `scores.csv`). The audit
launcher and reduced-baseline report launcher both call
`scripts/submit_utility_report.py`; their calls are serialized to submit one report.
It uses an `afterok` dependency while the registered audit array is unfinished;
if accounting already verifies all 23 audit tasks as successful, it submits
without recreating a dependency on a retired job ID. Slurm limits how long
completed IDs remain usable for [new dependencies](https://slurm.schedmd.com/sbatch.html#OPT_dependency).
Missing or partial baseline
summaries are recorded as `waiting_for_baselines` in
`results/nvidia_layer_validation/utility_v1/finished/audit_submission.json`.
If generating the baseline report manually, run the helper as shown above after
the report completes. Repeated helper calls are safe and do not repeat training.
To rerun an already submitted aggregate report explicitly, use
`sbatch jobs/nvidia_utility_report.sbatch` after checking both prerequisites.

This implementation uses the **per-target Ridge probe and reference ternary
search**, concatenated PCA100 blocks and the reference SharedTrunkNet with MSE
utility targets. Its files live in `src/nvidia_utility/` and
`results/nvidia_layer_validation/utility_v1/`. The older implementation in
`src/nvidia/` selects one common layer by source validation utility and has a
separate fixed-L21 control. These are different comparisons. Do not label the old
implementation or its inference benchmark as measurements of the final pipeline.

Required upstream executable sources are included at
`src/nvidia_utility/vendor/`, with published hashes, commit, URLs and license.
The wrapper changes the BCE objective to MSE in memory; vendor bytes are preserved.

## 5. Full layer sweep

After the complete raw layer extraction and baseline preparation:
```bash
.venv/bin/python -B scripts/initialize_release.py --stage layer-sweep
sbatch --array=0-27%4 jobs/layer_routing.sbatch
# Wait for all packs.
sbatch jobs/layer_report.sbatch
# Wait for the report job to create summary/per_split.csv.
.venv/bin/python -B src/diagnostics/audit_layer_source_selection.py
```
Run all packs in a new study. Do not copy the historical resume range 6–27.

## 6. Controlled-format and source-family diagnostics

Controlled format, after RouterBench preparation:
```bash
.venv/bin/python -B scripts/initialize_release.py --stage format
.venv/bin/python -B src/diagnostics/format_factorial_prepare.py
sbatch --array=0-1%2 jobs/format_factorial_extract.sbatch
# Wait for both encoders, then:
sbatch --array=0-1%2 jobs/format_factorial_analyze.sbatch
# Wait for both analyses, then:
sbatch jobs/format_factorial_report.sbatch
# Wait for the report job to create comparison.json.
.venv/bin/python -B src/diagnostics/format_factorial_audit.py
```

The initializer creates relative, internal runtime symlinks for unchanged
canonical encoding. No symlink points outside the release. The intervention
crosses prompt structure with requested answer encoding within MMLU; topic is
observed, not experimentally manipulated.

Family-relative-utility diagnostic, after the **complete GTE OOD geometry**:
```bash
.venv/bin/python -B scripts/initialize_release.py --stage family-utility
sbatch --array=0-1%2 jobs/family_utility_transfer.sbatch
# Wait for both datasets:
sbatch jobs/family_utility_transfer_report.sbatch
```

Optional four-prompt illustration:
```bash
.venv/bin/python -B scripts/initialize_release.py --stage proof-twins
sbatch jobs/proof_twins.sbatch
```
The exact four input prompts are included. This is an embedding-only comparison;
it measures neither generation cost nor cross-encoder cosine calibration.

## 7. Historical common-layer NVIDIA and CPU latency controls

Only to reproduce those explicitly labeled historical controls, after PCA:
```bash
sbatch jobs/nvidia_train.sbatch 0
sbatch jobs/nvidia_train.sbatch 1
# After all 23 folds and the routing audit complete:
.venv/bin/python -B scripts/initialize_release.py --stage inference
sbatch jobs/nvidia_inference_cpu.sbatch
# Wait for measurements, then:
sbatch jobs/nvidia_inference_audit.sbatch
```
The CPU inference benchmark uses precomputed raw embeddings, four threads,
batches 1/128 and identical queries for all methods. It excludes encoder inference
and does not measure the final per-target NVIDIA pipeline. Accounting scripts
that read historical Slurm IDs/submission metadata describe the original study;
fresh scheduler accounting requires records from the new scheduler. In particular,
`results/nvidia/submission.json` must contain the new `training_workers` IDs and
any available `preflight`, `timing_preflight`, `extraction_array` and `pca_array`
IDs. Without that file, the timing report retains measured phase times and marks
scheduler accounting unavailable; it does not query unrelated jobs or invent
allocated CPU/GPU totals.

## 8. Additional diagnostics and uncertainty

Run direct CPU commands below inside a CPU allocation, from the scientific workspace.
Wait for each prerequisite before submitting its consumer. Runtime arrays and
predictions must come from the new run, not the compact `results/evidence/` export.

GTE hidden-layer-21 comparison, after canonical inputs, models and baseline setup:

```bash
sbatch jobs/gte_l21_extract.sbatch
# Wait for both extraction tasks.
sbatch jobs/gte_l21_route.sbatch
# Wait for all 23 routing tasks.
sbatch jobs/gte_l21_report.sbatch
```

Reward/cost prediction and known-format restrictions, after all GTE/L21 baseline
predictions are complete:

```bash
sbatch jobs/knn_cost_reward_format.sbatch
# Wait for both benchmark tasks.
sbatch jobs/knn_cost_reward_format_report.sbatch
# Wait for both reports.
.venv/bin/python -B src/diagnostics/render_knn_cost_reward_format.py
```

Predicted-label lookup uses canonical GTE embeddings and the ten IID tasks:

```bash
.venv/bin/python -B src/diagnostics/submit_deployable_lookup.py
```

The helper submits the fixed array and a dependent report job. For the separate
IID disagreement analysis, wait for all five GTE IID kNN and known-label lookup
prediction files on both benchmarks, then run:

```bash
.venv/bin/python -B src/diagnostics/iid_lookup_disagreement.py
```

The native MMLU-topic diagnostic uses prepared outcomes and frozen IID splits,
without encoder extraction:

```bash
.venv/bin/python -B src/diagnostics/mmlu_topic_preference.py
```

For a complete CPU rerun with automatic comparison to saved results, including
all IID known-metadata lookups and this MMLU diagnostic, use
`.venv/bin/python -B scripts/reproduce_cpu_diagnostics.py`. It creates an isolated
workspace and verifies canonical input hashes before execution.

Appendix uncertainty calculations require completed fresh predictions and
query-level geometry/rank arrays:

```bash
sbatch jobs/appendix_ci_routing.sbatch
sbatch jobs/appendix_ci_geometry.sbatch
sbatch jobs/appendix_ci_clustering.sbatch
# Wait for all clustering assignment jobs before its bootstrap stage.
sbatch jobs/appendix_ci_clustering_bootstrap.sbatch
sbatch jobs/appendix_ci_rank.sbatch
# Final and historical SharedTrunkNet predictions must all be complete first.
sbatch jobs/appendix_ci_sharedtrunknet.sbatch
# Wait for every required report before rendering its tables.
sbatch jobs/appendix_ci_render.sbatch
.venv/bin/python -B src/diagnostics/appendix_ci/render_sharedtrunknet.py
# Requires completed OOD kNN routing bootstrap arrays as well as predictions.
sbatch jobs/per_family_knn_ci.sbatch
```

The family-rank bootstrap uses the corrected average-rank tie convention described
in [REPRODUCTION_AUDIT.md](../validation/REPRODUCTION_AUDIT.md). Original evidence is historical;
the new convention can affect auxiliary intervals with ties. The observed point
estimates remain unchanged.

## Other entrypoints and audit scope

The launch scripts preserve their resource requests and Python entrypoints.
Additional appendix uncertainty calculations are in `jobs/appendix_ci_*.sbatch`;
they require the completed routing/geometry outputs identified in those scripts.
`src/diagnostics/submit_deployable_lookup.py` submits only the fixed ten IID
lookup tasks and their report; it no longer edits private planning documents.

Initialization is explicit and repeatable when manifests agree. It refuses to
overwrite differing runtime hashes or rebind reduced/NVIDIA configs after result
or even partial fit/PCA output files exist. Missing assets must be generated in the order above. It never
relaxes a frozen utility data/split hash to accept a different dataset.

Initial packaging checks validated source/shell syntax, internal link targets,
vendor hashes and small synthetic tests. Subsequent reviewer checks installed
fresh environments, downloaded pinned data and tokenizers, reproduced complete
data preparation, extracted embeddings for four supplied prompts, fitted small
CPU/GPU router fixtures and exercised actual PCA100 preprocessing. See
`docs/validation/VALIDATION.md` for measured coverage. The subsequent full-input campaign has
since completed final per-target model training and checkpoint audits, with
historical-result differences retained in
[REPRODUCTION_AUDIT.md](../validation/REPRODUCTION_AUDIT.md). Other full-scale stages and
bootstrap resampling remain incomplete; full-study numerical equivalence has
not been established.


## Reviewer orchestration smoke (no downloads or compute jobs)

```bash
python3.11 -B scripts/smoke_orchestration.py
```

This uses temporary miniature fixtures under `.cache/`, unchanged method
placeholders and inert Python/Slurm command stubs. It checks all initializer stages,
missing-prerequisite failures, preservation of input hashes and existing metadata,
partial-checkpoint guards, safe repeated calls, internal runtime links, actual
launcher shell behavior and documented task ranges. It does not establish
numerical equivalence of a new full experiment run. See
`docs/validation/REVIEWER_SMOKE_ORCHESTRATION.md` for the findings.


## Actual PCA100 CPU smoke

Inside a small CPU allocation, run `.venv/bin/python -B scripts/smoke_pca.py`.
This executes the unchanged production preprocessing CLI on synthetic rows,
retains 100 components, poisons only held-out vectors to verify source isolation,
and checks saved transforms, hashes, cache reuse and expected rejections. See
`docs/validation/REVIEWER_SMOKE_PCA.md` for commands, measured outcomes and scope.
