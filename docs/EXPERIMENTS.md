# Experiment-to-command map

This map covers all 44 table labels in the pinned paper-reference registry,
[configs/paper_reference.json](../configs/paper_reference.json), whose source is
Overleaf snapshot `6e3bea88ec5d65de3587c0719025702e110ac3e0`. It distinguishes
numeric checks against that reference from new experimental execution.

## Start with the intended verification level

| Level | Command from repository root | What is actually computed |
| --- | --- | --- |
| Reference evidence | `make verify` | File integrity, 23,740 saved fold records, 4,031 rebuilt aggregate rows and 5,493 paper checks; no new fitting or resampling |
| Public outcome controls | `make controls` | 598 new single-model/oracle scores and 3,588 comparisons from canonical public outcomes |
| Complete CPU diagnostics | `make diagnostics` | 20 new IID metadata lookup reports and complete MMLU topic analysis, including 1,000 paired bootstrap draws per penalty |
| Numerical table production | `make tables` | Seven tables regenerated from saved reports and compared byte-for-byte |
| CPU regression/components | `make test` and `make smoke` | Regression cases and synthetic production fit/PCA/orchestration paths |
| Full study | [RUN_COMMANDS.md](RUN_COMMANDS.md) | New embeddings, fitted routers, predictions, diagnostics and bootstrap arrays |

`controls` and `diagnostics` require the canonical prepared inputs and the compute
environment. They do not treat frozen scores as fresh measurements. Full-study
stages must run after their prerequisites, in a new reproduction copy.

## Scientific stages

Outputs below are fresh runtime paths. Their saved counterparts live under
`results/evidence/`, with the leading `results/` replaced by `results/evidence/`.
Queue requests, measured accounting and their limits are in [RESOURCES.md](RESOURCES.md).

| Experiment | Prerequisites | Producers | Fresh output | Exact commands |
| --- | --- | --- | --- | --- |
| Inputs and protocol | Prepared cohorts and fixed protocols | `src/prepare_data.py`; `configs/mlp_reduced.json` | `datasets/*/prepared/`; frozen configs | [Data](DATA.md) and [setup](RUN_COMMANDS.md#platform-and-setup) |
| Main routing comparisons | Canonical GTE/L21 embeddings and fixed splits | `jobs/baselines_cpu.sbatch`, `jobs/baselines_ridge.sbatch`, reduced MLP workers | `results/baselines/_summary_reduced/` and Ridge summaries | [Baseline commands](RUN_COMMANDS.md#2-baselines) |
| Geometry and locality | All canonical encoder cohorts and geometry task configuration | `jobs/geometry_batch.sbatch`, then `jobs/report.sbatch` | `results/geometry/_summary/` and per-task metrics | [Geometry commands](RUN_COMMANDS.md#1-canonical-embeddings-and-geometry) |
| Clustering recovery | Completed geometry and matched encoder subsets | `jobs/descriptive_clustering.sbatch`; `jobs/additional_clustering.sbatch` | Encoder/subset clustering metrics and `additional_clustering_kmeans/` | [Clustering commands](RUN_COMMANDS.md#1-canonical-embeddings-and-geometry) |
| Family utility retrieval rank | Complete GTE OOD geometry and public outcome panels | `jobs/family_utility_transfer.sbatch`, then report launcher | `results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer/` | [Diagnostic commands](RUN_COMMANDS.md#6-controlled-format-and-source-family-diagnostics) |
| Layer sweep and nested selection | All Qwen raw layers 15–28, prepared baselines | `jobs/layer_routing.sbatch`, `jobs/layer_report.sbatch`, selection audit | `results/layer_sweep/summary/` | [Layer commands](RUN_COMMANDS.md#5-full-layer-sweep) |
| Final per-target SharedTrunkNet | Layer extraction, source-fitted PCA100 and utility preflight | `jobs/nvidia_utility_probe.sbatch`, two training workers, audit/report | `results/nvidia_layer_validation/utility_v1/` | [PCA](RUN_COMMANDS.md#3-candidate-layers-and-pca) and [final training](RUN_COMMANDS.md#4-final-per-target-nvidia-utility-pipeline) |
| Reward/cost and format controls | Completed GTE/L21 baseline predictions and canonical inputs | `jobs/knn_cost_reward_format.sbatch`, then report and table producer | `results/diagnostics/knn_cost_reward_format/` | [Appendix and diagnostic commands](RUN_COMMANDS.md#8-additional-diagnostics-and-uncertainty) |
| Routing uncertainty | Completed router predictions for all fixed outer splits and penalties | `jobs/appendix_ci_routing.sbatch`, then table renderer | `results/diagnostics/appendix_confidence_intervals/routing/` | [Bootstrap commands](RUN_COMMANDS.md#8-additional-diagnostics-and-uncertainty) |
| Predicted-label lookup | GTE embeddings and canonical IID splits | `src/diagnostics/submit_deployable_lookup.py` | `results/diagnostics/deployable_lookup/` | [Lookup commands](RUN_COMMANDS.md#8-additional-diagnostics-and-uncertainty) |
| IID lookup disagreement | Completed GTE IID kNN and lookup predictions | `src/diagnostics/iid_lookup_disagreement.py` | `results/diagnostics/iid_lookup_disagreement/` | [Lookup commands](RUN_COMMANDS.md#8-additional-diagnostics-and-uncertainty) |
| SharedTrunkNet uncertainty | Final utility and both earlier SharedTrunkNet control predictions | `jobs/appendix_ci_sharedtrunknet.sbatch`; SharedTrunkNet table renderer | `results/diagnostics/appendix_confidence_intervals/sharedtrunknet/` | [Control](RUN_COMMANDS.md#7-historical-common-layer-nvidia-and-cpu-latency-controls) and [bootstrap commands](RUN_COMMANDS.md#8-additional-diagnostics-and-uncertainty) |
| Per-family routing uncertainty | Completed OOD kNN predictions and original routing bootstrap arrays | `jobs/per_family_knn_ci.sbatch` | `results/diagnostics/per_family_knn_ci/v1/` | [Bootstrap commands](RUN_COMMANDS.md#8-additional-diagnostics-and-uncertainty) |

The controlled-format factorial, proof-twin illustration, MMLU native-topic
analysis, GTE L21 diagnostic, geometry/clustering/rank uncertainty and earlier
CPU inference controls are additionally described in the execution guide and
[RESULT_INDEX.md](RESULT_INDEX.md). They retain their distinct information regimes;
known-metadata diagnostics and realized oracles are not deployable prompt-only routers.

## Table-label coverage

Use stable manuscript labels when identifying a result; table numbers may change
between manuscript versions. The registry stores the expected numeric values and
their source selectors.

| Experiment | Pinned manuscript table labels |
| --- | --- |
| Inputs and protocol | `tab:sprout-prices`, `tab:mlp-settings` |
| Main routing comparisons | `tab:ood-encoder-means`, `tab:ood-mean` |
| Geometry and locality | `tab:difficulty-locality`, `tab:full-gte-family-locality`, `tab:semantic-family-locality`, `tab:family-concentration`, `tab:answer-format` |
| Clustering recovery | `tab:subject-recovery` |
| Family utility retrieval rank | `tab:utility-family-rank` |
| Layer sweep and nested selection | `tab:layer-sweep`, `tab:layer-iid` |
| Final per-target SharedTrunkNet | `tab:nvidia-utility-comparison` |
| Reward/cost and format controls | `tab:answer-format-control`, `tab:reward-cost-prediction`, `tab:answer-format-utility`, `tab:cost-reward-errors`, `tab:cost-reward-relative`, `tab:cost-reward-mean-ci`, `tab:cost-reward-differences`, `tab:cost-reward-routerbench`, `tab:cost-reward-sprout` |
| Routing uncertainty | `tab:ci-routerbench-iid-1`, `tab:ci-routerbench-iid-2`, `tab:ci-routerbench-iid-3`, `tab:ci-routerbench-ood-1`, `tab:ci-routerbench-ood-2`, `tab:ci-routerbench-ood-3`, `tab:ci-sprout-iid-1`, `tab:ci-sprout-iid-2`, `tab:ci-sprout-iid-3`, `tab:ci-sprout-ood-1`, `tab:ci-sprout-ood-2`, `tab:ci-sprout-ood-3` |
| Predicted-label lookup | `tab:lookup-deployability` |
| IID lookup disagreement | `tab:iid-disagreement-summary`, `tab:iid-disagreement-penalties` |
| SharedTrunkNet uncertainty | `tab:nvidia-mean-ci`, `tab:sharedtrunknet-differences`, `tab:sharedtrunknet-routerbench`, `tab:sharedtrunknet-sprout` |
| Per-family routing uncertainty | `tab:family-routing-routerbench`, `tab:family-routing-sprout` |

## Completion criteria

A completed run must retain all prescribed cohorts, outer splits, candidates,
seeds and cost penalties. Compare dataset identities and fixed splits first,
then per-fold metrics, aggregate weighting and interval definitions. New training
may have floating-point variation; report it against a stated tolerance rather
than rounding it away. Exact-input fingerprints and frozen-table checks remain exact.

The compact result checker uses relative tolerance `1e-10` and absolute tolerance
`1e-12` for saved-score reconstruction. The paper gate separately uses printed
precision. Fresh CPU diagnostics use the same numerical tolerances and exact
integer/string/shape checks. These tolerances do not certify a full CUDA training
rerun; its numerical comparison and scientific conclusions require their own audit.

A paper-reference pass establishes consistency of saved evidence. Component
execution establishes that tested paths run. Full-scale training and every paired
bootstrap have not all been independently repeated; see [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md).
