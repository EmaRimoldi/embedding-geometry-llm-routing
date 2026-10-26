# Result evidence and reconstruction

This release contains a compact snapshot of saved experiment results under `results/evidence/`. It includes completed outcomes, all recorded cost penalties, negative results, incomplete-state indicators, uncertainty reports and numerical audits. Evidence is isolated from the runtime result directories so a full rerun does not mistake exported summaries for completed model fits.

Run from the release root:

```bash
python3 -B scripts/reproduce_results.py --verify-only
python3 -B scripts/reproduce_results.py --plots
```

Only Python's standard library is required. The first command checks every exported file against `results/release_evidence_manifest.json` and writes nothing. The second regenerates aggregate CSVs and two SVG overview charts in `results/reproduced/`. `--output PATH` selects another output directory. The SVGs are new views of the saved scores; they are not the historical manuscript figures.

The reconstruction checks 23,740 saved fold-level records across seven result groups, reproduces 4,031 aggregate rows, and checks utility = reward - penalty × cost. It preserves query-count weighting, incomplete-group flags and all thirteen penalties. Reconstructed numbers must match saved aggregates within relative tolerance 1e-10 and absolute tolerance 1e-12. The final per-model SharedTrunkNet fold scores are additionally rebuilt from the 23 exported `result.json` records and compared with their saved CSV and report.

These checks re-aggregate saved scores. They do not independently reconstruct predictions from prompts or checkpoints, refit models, or resample confidence intervals. Those operations require the full workflow and omitted numerical arrays. Saved audit reports describe checks performed in the original experiment; their presence is distinct from a check rerun by this release script.

## Canonical comparisons

**MLP:** `baselines/_summary_reduced/` contains the completed reduced-grid MLP comparison. `baselines/_summary/` retains the original broader run, including its partial status; do not substitute incomplete original MLP aggregates for the completed reduced comparison. Ridge is stored separately under `baselines/_summary/ridge_*.csv`.

**Final SharedTrunkNet:** `nvidia_layer_validation/utility_v1/` is the final per-target, per-penalty utility adaptation. Each candidate model selects a Qwen layer from 15–28 using five-fold source-query Ridge(alpha=1) MSE and the released ternary search. Source-trained scaler/PCA100 transforms are fixed during probing. Selected model blocks are concatenated and the shared network uses continuous-utility MSE with five of ten seeds retained. Its saved summary reports 23 outer evaluations, 299 fold/penalty scores and 2,990 network fits. The final definition is in `configs/nvidia_utility_v1.json` and `src/nvidia_utility/`.

**Earlier SharedTrunkNet controls:** `nvidia/` contains fixed-L21 and one-common-selected-layer variants. Their definition is `configs/nvidia.json` and `src/nvidia/`. They remain reported controls, with their own selection, scores and compute accounting. They are not the final per-target selection procedure. The final appendix report at `diagnostics/appendix_confidence_intervals/sharedtrunknet/` compares all three variants.

**Inference timing:** `nvidia/inference_cpu_v1/audits/` contains the latest saved audit of the earlier NVIDIA pipeline. These CPU measurements must not be attributed to the final `utility_v1` pipeline. `utility_v1/summary/compute_and_reversals.json` contains that final run's incremental compute and penalty reversals; cached extraction/PCA costs are explicitly distinguished. Hardware types and numerical resource totals are retained while machine identifiers and scheduler details are redacted.

## Study-to-evidence map

Paths in this table are relative to `results/evidence/`. A row marked "saved evidence" is available for inspection and integrity verification; its underlying statistical computation is not rerun by the compact reconstruction command.

| Study | Evidence path | Compact reconstruction |
| --- | --- | --- |
| Original routing run, including partial MLP status | `baselines/_summary/{per_split.csv,scores.csv,status.json}` | Rebuilds all aggregate rows, including explicit incomplete groups |
| Completed reduced MLP comparison and all baseline methods | `baselines/_summary_reduced/` | Rebuilds all aggregate rows |
| Ridge | `baselines/_summary/{ridge_per_split.csv,ridge_scores.csv,ridge_status.json}` | Rebuilds all aggregate rows |
| Geometry, neighbor composition, native group comparisons and difficulty | `geometry/_summary/` | Rebuilds `pooled_query_weighted.csv` from `neighborhoods.csv`; other CSVs/audits are saved evidence |
| Descriptive and additional ordinary-k-means clustering | Encoder/dataset/subset directories under `geometry/`; `geometry/_summary/additional_clustering_kmeans/` | Saved metric records and all available cluster-recovery scores |
| Upper-layer routing and source-selected layers | `layer_sweep/summary/` and `layer_sweep/qwen2.5-7b-instruct/` | Rebuilds aggregate fixed-layer and nested-selection scores |
| Final per-target SharedTrunkNet utility adaptation | `nvidia_layer_validation/utility_v1/{summary,routerbench,sprout}/` | Rebuilds 299 fold scores from JSON and 52 complete dataset/protocol/penalty aggregates |
| Earlier SharedTrunkNet fixed-L21/common-layer controls | `nvidia/{summary,results,final_comparison}/` | Rebuilds all 104 control aggregate rows; comparison audits retained |
| Earlier SharedTrunkNet CPU inference timing | `nvidia/inference_cpu_v1/audits/` | Saved timing audit and CSVs; applies only to the earlier pipeline |
| GTE layer 21 versus final GTE | `diagnostics/gte_l21/` | Saved per-fold results and report |
| Family utility transfer and family-ranking analysis | `geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer/`; `geometry/_diagnostics/family_transfer_ranking/` | Saved per-source results, all-penalty CSVs, numerical checks and LaTex tables |
| Predicted-label lookup deployability | `diagnostics/deployable_lookup/` | Saved per-fold results, scores and report |
| IID kNN-versus-lookup disagreement | `diagnostics/iid_lookup_disagreement/` | Saved split, penalty and summary CSVs with paired uncertainty |
| Answer-format/template factorial controls | `geometry/_diagnostics/format_factorial_v1/`; encoder-specific `format_factorial_v1/report.json` files | Saved per-encoder metrics, comparison and audit |
| Proof twins | `geometry/_diagnostics/proof_twins_v1/`; `geometry/gte-qwen2-7b/_diagnostics/proof_twins_v1/` | Saved descriptive cosine comparison and audit; no generated-answer experiment implied |
| MMLU native topic preference | `geometry/_diagnostics/mmlu_topic_preference/` | Saved report and audit |
| Main routing, geometry, clustering and family-rank confidence intervals | `diagnostics/appendix_confidence_intervals/{routing,geometry,clustering,family_rank}/` | Saved bootstrap reports; intervals are not recomputed |
| Final appendix table outputs | `diagnostics/appendix_confidence_intervals/overleaf/{aistats_ci_tables.tex,aistats_tables.tex}` | Saved full table outputs, point map, macros and cell audit |
| SharedTrunkNet appendix and source-selected-family table | `diagnostics/appendix_confidence_intervals/sharedtrunknet/{report.json,aistats_sharedtrunknet.tex,layer_selected_families.tex}` | Saved all-control and per-penalty results with intervals |
| kNN reward/cost errors and answer-format controls | `diagnostics/knn_cost_reward_format/{routerbench,sprout}/report.json`; `paper/aistats_cost_reward.tex` within that directory | Saved all errors, paired comparisons, per-family/per-model outcomes and table |
| Latest per-family kNN uncertainty | `diagnostics/per_family_knn_ci/v1/` | Saved 2,000-replicate paired bootstrap report, all family rows, CSV and completion audit |

## Generated compact outputs

`results/reproduced/` contains:

- `baselines_original.csv`, `baselines_reduced.csv`, `ridge.csv`, `layer_sweep.csv` and `nvidia_earlier_controls.csv`: rebuilt aggregates, with completeness flags.
- `geometry_pooled.csv`: rebuilt query-weighted geometry summaries.
- `nvidia_final_per_split.csv` and `nvidia_final_per_penalty.csv`: final per-target utility scores rebuilt from exported fold JSONs.
- `main_comparison_all_penalties.csv` and `main_comparison_mean_utility.csv`: completed reduced-MLP baseline methods, Ridge and final per-target SharedTrunkNet. No penalty is dropped and means weight the thirteen fixed penalties equally.
- `verification.json`: integrity, coverage, numerical comparison results and scope limitations.
- With `--plots`, `figures/*_ood_utility.svg`: utility-over-penalty overview charts. All plotted methods retain every penalty and negative utility values. The CSVs retain methods not shown in the compact overview.

Both `main_comparison_*.csv` files include `method_role` and `required_information`. Every existing method, score and penalty remains present. These columns distinguish the information available to each method:

| `method_role` | Methods | Interpretation |
| --- | --- | --- |
| `source_trained_router` | kNN-100, validated kNN, Family Adaptation, best single model, Avengers-Pro, KM-Prox variants, MLP, Ridge, final SharedTrunkNet | Fitting and selection use source data; test outcomes are used only to evaluate decisions. Prompt-dependent routers consume test embeddings. Best single model needs no test features; the Family Adaptation additionally uses source family labels. |
| `known_metadata_diagnostic` | Family lookup, Family-topic lookup | Requires the test query's true family or native cell/family metadata. These are information-assisted diagnostic references. Predicted-label deployment is evaluated separately. |
| `realized_oracle` | Realized oracle | Uses every candidate's realized reward and cost on the test query to form an upper-bound reference; it is not a deployable routing policy. |
| `diagnostic_clustering_control` | Utility-cluster control | A source-trained clustering control using test embeddings and source outcome tables; it does not use test outcomes for selection. |

`required_information` gives the specific input requirements for each row. Methods carrying different information regimes must not be compared as if all received only the test prompt and source training data.

## Provenance and omissions

`results/release_evidence_manifest.json` records each original repository-relative source path, original SHA-256, exported path, exported SHA-256, byte size and redacted fields. Original hashes describe private source bytes; exported hashes verify the sanitized delivered bytes. Hashes embedded inside saved audit records continue to refer to original-run files, including large artifacts that are not included here.

The evidence export excludes raw corpora, embeddings, PCA arrays, model weights, individual training-fit artifacts, query-level prediction/bootstrap arrays, scheduler logs, old manuscript files and superseded protocol proposals. The original workspace retains those artifacts. This package does not claim that integrity verification alone proves the original model computation. Reproducing prediction audits, bootstrap intervals or historical manuscript figures requires regenerating the corresponding full-workflow artifacts.

The historical manuscript-only figure builder has been removed from this repository.
Use the experimental evidence and compact result commands above; the numerical
producers and geometry plotting tools remain available.
