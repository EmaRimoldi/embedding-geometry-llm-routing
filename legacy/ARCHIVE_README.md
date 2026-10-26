# Code for "The geometry of prompt embeddings and LLM routing" (anonymous ICLR 2027 submission)

This archive contains the code that produced the paper's results, the frozen routing problems (rewards, costs, family and topic labels, and every in-distribution and held-out-family split), and the job scripts. Directory names are kept exactly as used in the experiments, because studies read one another's outputs by relative path. The archive root plays the role of `E4_ROOT`.

## 1. Environment

- Python 3.11 with the packages in `requirements.txt` (numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, torch 2.4.1, transformers 4.44.2, sentence-transformers 3.1.1).
- Embedding extraction and the MLP baselines use one GPU. All routers and analyses run on CPU.
- Job scripts are SLURM (`*.sbatch`). Cluster-specific values were replaced by placeholders: `__PARTITION__`, `__ACCOUNT__`, `__QOS__` and `__NODES__`.

```bash
pip install -r requirements.txt
./configure.sh /path/to/external_data /path/to/model_weights   # fills path placeholders
export E4_ROOT="$(pwd)"
```

`configure.sh` replaces these path placeholders:

| Placeholder | Meaning |
|---|---|
| `__E4_ROOT__` | this directory |
| `__PROJECT_SCRATCH__`, `__PROJECT_HOME__`, `__SCRATCH__`, `__CLUSTER_PATH__` | location of the public benchmark releases and upstream artifacts |
| `__POOL__` | location of the model weights |
| `__HOME__` | the user's home directory |
| `__REMOTE_HOST__` | SSH host used only by `gpt56_lookup_reasoning_20260923/run_full_pipeline.py` to sync runs; set it or run locally |

## 2. Data

**Frozen routing problems (included).** These are the exact inputs of every routing result in the paper.
- `astra_definitive_paper_20260923_v1/inputs/routerbench-0shot/`: RouterBench zero-shot, 34,661 queries × 11 models, 7 families.
- `astra_definitive_paper_20260923_v1/inputs/sprout-sixfam/`: SPROUT, 43,887 queries × 13 models, 6 families.

Each panel folder contains:
- `panel.npz`: recorded rewards `r`, USD costs `c`, family ids `fam`, `topic` labels, query identifiers and row maps;
- `splits.json`: five family-stratified 70/30 in-distribution splits (`iid_0`–`iid_4`, with inner fit/validation folds) and one leave-one-family-out split per family (`ood_*`, with inner source-family holdouts);
- `source_manifest.json`: provenance and SHA-256 hashes of the upstream snapshots and of the original representation arrays.

The same `inputs/` folder also holds:
- `sprout_cells_native_openhermes_v1/cells.npz`: SPROUT lookup cells;
- `graph_descriptions/`: public task and model descriptions used by GraphRouter.

**Sources.** These files are derived from the public releases of RouterBench (Hu et al., 2024) and SPROUT (Somerstep et al., 2025). The panels were built with `src/rb/` (RouterBench), `sprout/scripts/` (SPROUT: token-count costs, OpenHermes subject labels), `astra_reward_transfer_20260922_v1/prepare.py` and `astra_sprout_l21_20260922_v1/prepare.py`. `astra_definitive_paper_20260923_v1/code/freeze.py` then froze them. Please cite the original benchmarks.

## 3. Embeddings (not included: about 0.5–0.6 GB per array)

The routers read `embeddings/<panel>/{gte,reader}.npy`: float32 arrays with one row per query, in the row order of `inputs/<panel>/panel.npz`.

| Representation | What it is | Extraction code |
|---|---|---|
| `gte` | emitted embedding of `Alibaba-NLP/gte-Qwen2-7B-instruct` | `scripts/embed_gpu.py` |
| `reader` | `hidden_states[21]` of `Qwen/Qwen2.5-7B-Instruct` at the last non-padding input token, no chat template, inputs up to 8,192 tokens | `scripts/extract_llm_hidden.py`; all layers: `layer_sweep_20260924_v1/code/extract.py` |

Raw arrays are produced by these scripts:

| Benchmark | GTE | Qwen hidden state |
|---|---|---|
| RouterBench | `scripts/embed_gpu.py` | `scripts/extract_llm_hidden.py` |
| SPROUT | `sprout/scripts/embed_gte.py` (`sprout/slurm/embed_gte.sbatch`) | `ood_cost_geometry_20260921/scripts/extract_hidden_sprout.py` |

Then run `tools/align_embeddings.py --panel <panel> --rep <gte|reader> --raw <array.npy>` to write `embeddings/<panel>/<rep>.npy`. For SPROUT it applies the same row maps (`gte_rows`, `panel_rows`) as the loader used for the paper; RouterBench arrays are already in panel order.

The SHA-256 of the original arrays is recorded in `source_manifest.json`. GPU numerics can differ in the last bits; the routers normalise rows to unit length. The loader is `astra_definitive_paper_20260923_v1/code/common.py:load`. We checked that it returns arrays identical to the loader used for the paper on both benchmarks.

## 4. Reproducing the results

All routers use the same splits, penalty grids (`astra_definitive_paper_20260923_v1/config/protocol.json`) and source-only hyper-parameter selection.

| Paper content | Code |
|---|---|
| kNN-100, validated kNN, soft family prior, best single model (in-distribution and held-out-family tables, frontiers) | `astra_definitive_paper_20260923_v1/code/run_fixed.py --task {0..45} --kind {knn,prior}` (`run_fixed.sbatch`) |
| Avengers-Pro / cluster routers | `code/run_clusters.py` (`run_clusters.sbatch`) |
| KM-Prox (ProxRouter) | `code/run_prox.py` (`run_prox.sbatch`) |
| MLP (published settings) | `code/run_mlp_gpu*.py` |
| Tuned MLP | `mlp_tuned_20260924_v1/` (`run_all.sh`) |
| GraphRouter (in-distribution) | `code/run_graph*.py`, `code/graph_descriptions.py` |
| Decision-to-score audit, paired bootstrap intervals | `code/audit_predictions.py`, `code/check_retrieval*.py` |
| Result assembly, matched held-out-family comparisons | `astra_baseline_completion_20260924_v1/`, `_v2/` |
| Family and family–topic lookups, predicted-cell lookup | `astra_definitive_paper_20260923_v1/code/` (`freeze_approved_cells.py`, `sprout_cell_audit.py`, `audit_cell_thresholds.py`), `astra_iid_geometry_bridge_20260923_v1/` |
| Proximity within cells; kNN versus lookup decisions | `astra_iid_geometry_bridge_20260923_v1/` |
| Embedding geometry (family/topic clustering, neighbour composition, family removal) | `astra_definitive_paper_20260923_v1/code/geometry_candidates.py`, `code/native_neighbors.py`, `scripts/exp0*.py`, `src/rb/geometry_tests/` |
| MiniLM checks | `minilm/` |
| Question format, template removal, mathematical twins, hard examples, technique twins | `results/format_invariance/`, `results/deformatting/`, `results/hard_twins/`, `results/hard_examples/`, `results/technique_twins/`: code and hand-built inputs (`collection.json`, `texts.json`) |
| Difficulty within families and topics | `astra_iid_geometry_bridge_20260923_v1/` (RouterBench), `astra_sprout_difficulty_cells_20260923_v1/` (SPROUT) |
| Neighbour controls, reward/cost swaps, family-mixture decomposition | `astra_definitive_paper_20260923_v1/neighbor_explanation_v1/` |
| Linear prediction on unseen families | `astra_definitive_paper_20260923_v1/linear_probe_v1/` |
| Presentation, pooling and surface-feature controls | `astra_encoder_geometry_20260922_v1/`, `astra_geometry_routing_20260922_v1/` |
| Targeted cost predictor | `ood_cost_geometry_20260921/` (RouterBench), `astra_cost_router_20260923_v1/` (SPROUT) |
| Difficulty probes; layer choice | `scripts/` (difficulty probes), `layer_sweep_20260924_v1/` (layer sweep and nested source-only selection) |
| Oracle-gap recoverability | `scripts/oracle_gap/` |
| Routing-by-reasoning control | `gpt56_lookup_reasoning_20260923/` (needs API access to the evaluated model; keys are read from environment variables) |
| Figures | `reports/figures/` |

Each study writes `results/` and `logs/` inside its own folder; they were removed from this archive. Result files record input and code hashes. Randomness is seeded: split seeds 0–4, validation seeds 10000–10004, and fixed seeds inside each study.

## 5. Notes

- **Checked reproducibility.** Running `run_fixed.py` from this archive for all 92 kNN and soft-prior tasks (both benchmarks; every in-distribution and held-out-family split; both representations) reproduces the paper's results. 56 tasks are bit-identical. In the others, floating-point near-ties change the predictions of at most 0.25% of queries, for example neighbours with equal cosine similarity on a different CPU/BLAS. 38 of 2,990 reported utilities change at the fourth decimal.
- **Input-integrity checks.** `audit_render.py`, `astra_iid_geometry_bridge_20260923_v1/code/run.py` and `astra_sprout_difficulty_cells_20260923_v1/code/run.py` check the SHA-256 of the SPROUT inputs listed in `config/SPROUT_CANONICAL_PROTOCOL.json`. In this archive they skip the manifests, which were anonymized, and the upstream arrays that are not shipped; every other listed file is still verified.
- **Figures.** Figures were drawn from the stored results (`results/*/result.json`, the assembled records of `astra_baseline_completion_*`, and each study's summaries); tables were typed from the same files. The layer-sweep figure has its own script (`layer_sweep_20260924_v1/code/plot.py`) and older figure scripts are in `reports/figures/`. The plotting scripts for the in-distribution frontier, geometry hierarchy, soft-prior frontier and matched held-out-family comparison are not included; every number they show is produced by the code above.
- **Earlier analyses.** Analyses in `scripts/` and `minilm/` (difficulty probes, family and topic clustering, early held-out-family studies) read feature caches under `features/<panel>/…` and the RouterBench panel built by `src/rb/`. Rebuild them with `scripts/embed_gpu.py`, `scripts/extract_llm_hidden.py` and `scripts/extract_hidden_all_layers.py`.
- **SPROUT cost router.** `astra_cost_router_20260923_v1/run.py` reads the SPROUT cohort (`astra_sprout_l21_20260922_v1/inputs/`, included) and loads features from the paths in its manifest, checking their size and modification time. With regenerated features, point these paths to the new arrays and relax that check.
- **Run order.** Run the routers in `astra_definitive_paper_20260923_v1` (`run_fixed.py` and the others) before the studies that read their outputs, such as `astra_iid_geometry_bridge_20260923_v1`, `astra_baseline_completion_*` and `layer_sweep_20260924_v1`. For SPROUT, also run `astra_sprout_difficulty_cells_20260923_v1/code/run.py` (seeds 0–4) before the SPROUT tasks of the bridge.
- **GraphRouter descriptions** are frozen in `astra_definitive_paper_20260923_v1/inputs/graph_descriptions/`, so `graph_descriptions.py` does not need to be re-run.

- Router utility is `r − λ·c` with the recorded reward and USD cost of the selected model. Router inference cost is not included.
- Upstream artifacts that are not needed for the routing results (raw benchmark files, cached features, logs) are not included; they can be regenerated with the included code.
- `astra_final_audits_20260922_v1/sources/` contains third-party code under its original license.
- `CHECKSUMS.sha256` lists SHA-256 hashes of every file in this archive.
