# Compute and storage requirements

Use the pinned Linux x86-64/Python 3.11 environment for complete experiment runs.
The numerical-evidence route needs only the Python standard library. CPU component
checks use `environment/cpu-requirements.lock`; full encoder/training jobs use the
CUDA environment in `requirements.lock`.

## Storage and GPU memory

The verified model weights total about 45.8 GB. Public benchmark payloads total
about 680 MB; metadata, tokenizer/cache duplicates and prepared panels add space.
Budget more than 100 GB for all layers, PCA caches, fitted models, predictions and
bootstrap outputs. This is a planning allowance, not a measured minimum.

Slurm `--mem` requests below are host RAM, not GPU VRAM. A universal minimum VRAM
has not been established by these checks. The 7B encoders use BF16 and complete
prompts up to 8192 tokens, so memory depends on model and prompt batch. Preserve
the frozen batch/token settings; select adequate hardware instead of truncating
inputs. At most two GPU tasks may be active across all stages combined.

## Measured records

| Measurement | Recorded value | Scope |
| --- | ---: | --- |
| Final utility SharedTrunkNet allocated GPU time | 2.7894 GPU-hours | Original incremental final run; excludes cached extraction/PCA |
| Same run allocated CPU time | 11.7344 CPU-core-hours | Includes recorded preflight/probe/train/audit/report jobs, including a failed preflight |
| Same run training elapsed wall time | 6,605 seconds | Recorded training wall accounting; distinct from summed per-network timers |
| Final utility networks | 2,990 fits | 23 outer evaluations × 13 penalties × 10 seeds; five seeds retained per ensemble |

Source: [compute_and_reversals.json](../results/evidence/nvidia_layer_validation/utility_v1/summary/compute_and_reversals.json).
Earlier CPU inference timings apply to the historical common-layer controls,
exclude encoder runtime and must not be attributed to the final per-target pipeline.
[VALIDATION.md](VALIDATION.md) records earlier component/preparation durations;
fresh CPU diagnostic execution times are recorded in
[v1_cpu_diagnostics.json](../release_provenance/current_checks/v1_cpu_diagnostics.json).
No complete fresh wall-time or monetary estimate for every study stage is asserted.

## Scheduler requests

The following values are extracted from the included launchers. Walltime is a
scheduler limit per job, not a measured runtime; arrays and worker continuations
can create many jobs. Queue waits, GPU type, filesystem speed and concurrency
affect elapsed reproduction time. See [RUN_COMMANDS.md](RUN_COMMANDS.md) for task
ranges supplied on the command line and dependency order.

| Launcher | CPU cores | Host RAM | GPUs/job | Walltime limit | Embedded array |
| --- | ---: | --- | ---: | --- | --- |
| [additional_clustering.sbatch](../jobs/additional_clustering.sbatch) | 8 | 48G | 1 | 06:00:00 | `0-9%2` |
| [appendix_ci_clustering.sbatch](../jobs/appendix_ci_clustering.sbatch) | 8 | 16G | 0 | 01:00:00 | `0-29%4` |
| [appendix_ci_clustering_bootstrap.sbatch](../jobs/appendix_ci_clustering_bootstrap.sbatch) | 8 | 16G | 0 | 01:00:00 | `0-5%3` |
| [appendix_ci_geometry.sbatch](../jobs/appendix_ci_geometry.sbatch) | 4 | 16G | 0 | 01:00:00 | `0-1%2` |
| [appendix_ci_rank.sbatch](../jobs/appendix_ci_rank.sbatch) | 8 | 16G | 0 | 01:00:00 | `0-1%2` |
| [appendix_ci_render.sbatch](../jobs/appendix_ci_render.sbatch) | 2 | 8G | 0 | 00:30:00 | `supplied by caller / single job` |
| [appendix_ci_routing.sbatch](../jobs/appendix_ci_routing.sbatch) | 8 | 24G | 0 | 01:00:00 | `0-3%2` |
| [appendix_ci_sharedtrunknet.sbatch](../jobs/appendix_ci_sharedtrunknet.sbatch) | 4 | 8G | 0 | 00:20:00 | `supplied by caller / single job` |
| [audit_absolute_difficulty.sbatch](../jobs/audit_absolute_difficulty.sbatch) | 2 | 16G | 0 | 00:20:00 | `supplied by caller / single job` |
| [audit_features.sbatch](../jobs/audit_features.sbatch) | 2 | 8G | 0 | 00:30:00 | `supplied by caller / single job` |
| [audit_ordering_ties.sbatch](../jobs/audit_ordering_ties.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [baselines_cpu.sbatch](../jobs/baselines_cpu.sbatch) | 8 | 48G | 0 | 12:00:00 | `supplied by caller / single job` |
| [baselines_mlp.sbatch](../jobs/baselines_mlp.sbatch) | 4 | 32G | 1 | 06:00:00 | `supplied by caller / single job` |
| [baselines_mlp_reduced.sbatch](../jobs/baselines_mlp_reduced.sbatch) | 4 | 32G | 1 | 06:00:00 | `supplied by caller / single job` |
| [baselines_prepare.sbatch](../jobs/baselines_prepare.sbatch) | 4 | 16G | 0 | 00:30:00 | `supplied by caller / single job` |
| [baselines_reduced_preflight.sbatch](../jobs/baselines_reduced_preflight.sbatch) | 2 | 16G | 0 | 00:30:00 | `supplied by caller / single job` |
| [baselines_reduced_report.sbatch](../jobs/baselines_reduced_report.sbatch) | 2 | 16G | 0 | 00:30:00 | `supplied by caller / single job` |
| [baselines_report.sbatch](../jobs/baselines_report.sbatch) | 4 | 16G | 0 | 00:30:00 | `supplied by caller / single job` |
| [baselines_ridge.sbatch](../jobs/baselines_ridge.sbatch) | 8 | 32G | 0 | 12:00:00 | `supplied by caller / single job` |
| [deployable_lookup.sbatch](../jobs/deployable_lookup.sbatch) | 4 | 24G | 0 | 02:00:00 | `supplied by caller / single job` |
| [descriptive_clustering.sbatch](../jobs/descriptive_clustering.sbatch) | 8 | 16G | 0 | 01:00:00 | `supplied by caller / single job` |
| [encode.sbatch](../jobs/encode.sbatch) | 4 | 48G | 1 | 06:00:00 | `0-2%2` |
| [family_utility_transfer.sbatch](../jobs/family_utility_transfer.sbatch) | 8 | 16G | 0 | 01:00:00 | `supplied by caller / single job` |
| [family_utility_transfer_report.sbatch](../jobs/family_utility_transfer_report.sbatch) | 4 | 16G | 0 | 00:15:00 | `supplied by caller / single job` |
| [format_factorial_analyze.sbatch](../jobs/format_factorial_analyze.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [format_factorial_extract.sbatch](../jobs/format_factorial_extract.sbatch) | 4 | 48G | 1 | 02:00:00 | `supplied by caller / single job` |
| [format_factorial_prepare.sbatch](../jobs/format_factorial_prepare.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [format_factorial_report.sbatch](../jobs/format_factorial_report.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [geometry.sbatch](../jobs/geometry.sbatch) | 4 | 48G | 1 | 01:30:00 | `supplied by caller / single job` |
| [geometry_batch.sbatch](../jobs/geometry_batch.sbatch) | 4 | 48G | 1 | 06:00:00 | `0-19%2` |
| [gte_l21_extract.sbatch](../jobs/gte_l21_extract.sbatch) | 4 | 48G | 1 | 06:00:00 | `0-1%2` |
| [gte_l21_report.sbatch](../jobs/gte_l21_report.sbatch) | 8 | 48G | 0 | 06:00:00 | `supplied by caller / single job` |
| [gte_l21_route.sbatch](../jobs/gte_l21_route.sbatch) | 8 | 48G | 0 | 06:00:00 | `0-22%4` |
| [knn_cost_reward_format.sbatch](../jobs/knn_cost_reward_format.sbatch) | 4 | 16G | 0 | 01:00:00 | `0-1%2` |
| [knn_cost_reward_format_report.sbatch](../jobs/knn_cost_reward_format_report.sbatch) | 4 | 16G | 0 | 01:00:00 | `0-1%2` |
| [layer_extract.sbatch](../jobs/layer_extract.sbatch) | 4 | 48G | 1 | 06:00:00 | `supplied by caller / single job` |
| [layer_report.sbatch](../jobs/layer_report.sbatch) | 4 | 16G | 0 | 00:30:00 | `supplied by caller / single job` |
| [layer_routing.sbatch](../jobs/layer_routing.sbatch) | 8 | 32G | 0 | 12:00:00 | `supplied by caller / single job` |
| [nvidia_comparison_design_audit.sbatch](../jobs/nvidia_comparison_design_audit.sbatch) | 1 | 2G | 0 | 00:05:00 | `supplied by caller / single job` |
| [nvidia_extract.sbatch](../jobs/nvidia_extract.sbatch) | 4 | 48G | 1 | 06:00:00 | `supplied by caller / single job` |
| [nvidia_finalize.sbatch](../jobs/nvidia_finalize.sbatch) | 4 | 8G | 0 | 00:20:00 | `supplied by caller / single job` |
| [nvidia_inference_audit.sbatch](../jobs/nvidia_inference_audit.sbatch) | 4 | 32G | 0 | 02:00:00 | `supplied by caller / single job` |
| [nvidia_inference_cpu.sbatch](../jobs/nvidia_inference_cpu.sbatch) | 4 | 32G | 0 | 02:00:00 | `supplied by caller / single job` |
| [nvidia_pca.sbatch](../jobs/nvidia_pca.sbatch) | 16 | 32G | 0 | 12:00:00 | `supplied by caller / single job` |
| [nvidia_preflight.sbatch](../jobs/nvidia_preflight.sbatch) | 2 | 8G | 0 | 00:30:00 | `supplied by caller / single job` |
| [nvidia_report.sbatch](../jobs/nvidia_report.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [nvidia_train.sbatch](../jobs/nvidia_train.sbatch) | 4 | 24G | 1 | 06:00:00 | `supplied by caller / single job` |
| [nvidia_utility_audit.sbatch](../jobs/nvidia_utility_audit.sbatch) | 1 | 2G | 0 | 00:10:00 | `supplied by caller / single job` |
| [nvidia_utility_audit_task.sbatch](../jobs/nvidia_utility_audit_task.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [nvidia_utility_preflight.sbatch](../jobs/nvidia_utility_preflight.sbatch) | 4 | 16G | 0 | 00:30:00 | `supplied by caller / single job` |
| [nvidia_utility_probe.sbatch](../jobs/nvidia_utility_probe.sbatch) | 4 | 16G | 0 | 02:00:00 | `supplied by caller / single job` |
| [nvidia_utility_report.sbatch](../jobs/nvidia_utility_report.sbatch) | 4 | 4G | 0 | 00:30:00 | `supplied by caller / single job` |
| [nvidia_utility_train.sbatch](../jobs/nvidia_utility_train.sbatch) | 4 | 24G | 1 | 06:00:00 | `supplied by caller / single job` |
| [per_family_knn_ci.sbatch](../jobs/per_family_knn_ci.sbatch) | 2 | 8G | 0 | 00:10:00 | `supplied by caller / single job` |
| [plots.sbatch](../jobs/plots.sbatch) | 4 | 20G | 0 | 01:00:00 | `0-4%2` |
| [prepare.sbatch](../jobs/prepare.sbatch) | 4 | 24G | 0 | 01:00:00 | `supplied by caller / single job` |
| [proof_twins.sbatch](../jobs/proof_twins.sbatch) | 4 | 48G | 1 | 00:30:00 | `supplied by caller / single job` |
| [report.sbatch](../jobs/report.sbatch) | 4 | 24G | 0 | 02:00:00 | `supplied by caller / single job` |
