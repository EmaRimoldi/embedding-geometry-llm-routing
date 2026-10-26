# Routing Theory — experiment reproduction

Code, configurations and launch tools for rerunning the LLM routing experiments in *Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization*.

**Current status:** the September source archive is available here. The final October experiment implementation has been located on the cluster but is not readable with the current account. The conference submission package has not yet been imported. This repository does not claim exact reproduction of the final experiments.

The recovered historical code uses **34,661 RouterBench queries**, whereas the final protocol uses **34,771**. SPROUT has 43,887 queries in both, but matching counts alone do not establish identical preprocessing or settings. SharedTrunkNet's final implementation and other October changes still require the conference package. See [reproduction status](docs/reproduction-status.md).

## Start

Use Python 3.11. The small CPU environment supports input validation, reference baselines, kNN and Family Adaptation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
```

The tests execute the **original recovered kNN and soft-family-prior runners** on synthetic inputs, including their validation and prediction exports. Synthetic tests check execution and selected numerical properties; they do not reproduce benchmark results.

## Historical benchmark inputs

Raw benchmark records, model weights and embedding arrays are not tracked in Git. If you have the audited September `routing-geometry-code.zip`, stage its two frozen numeric panels locally:

```bash
python -m repro prepare-legacy-data \
  --archive /path/to/routing-geometry-code.zip \
  --destination data/legacy

python -m repro validate \
  --data-root data/legacy --panel routerbench-0shot
python -m repro validate \
  --data-root data/legacy --panel sprout-sixfam
```

The preparation command verifies the audited archive hash and extracts only the two panels, split registries and source manifests. It checks finite outcomes, unique IDs, outer/inner separation and complete held-out-family coverage. It does not download anything or invoke a model API. The legacy archive is currently required for this staging route; upstream-only preparation for the final cohort remains to be integrated from the conference package.

## Run actual reference controls

These commands recompute the training-selected single-model baseline and the realized-oracle upper bound, for all five ID splits and all held-out-family folds at all thirteen penalties:

```bash
python -m repro reference-baselines \
  --data-root data/legacy --panel routerbench-0shot \
  --split all --output runs/routerbench-reference.json

python -m repro reference-baselines \
  --data-root data/legacy --panel sprout-sixfam \
  --split all --output runs/sprout-reference.json
```

The realized oracle uses observed test outcomes as an upper bound; it is not a deployable router. Outputs contain utility, reward, cost and family summaries, input hashes and environment versions. Existing output files are never overwritten. The checked-in `validation/` results record actual executions on these **historical** inputs.

## Run original kNN or Family Adaptation

Provide aligned embeddings, with exactly one row per frozen-panel query:

```text
data/legacy/
  astra_definitive_paper_20260923_v1/inputs/<panel>/
    panel.npz
    splits.json
    source_manifest.json
  embeddings/<panel>/
    gte.npy
    reader.npy
```

`panel` is `routerbench-0shot` or `sprout-sixfam`. The archive's extraction and alignment scripts are retained under `legacy/`; see [the historical instructions](legacy/ARCHIVE_README.md). GPU extraction and the other historical methods require their additional dependencies. The original dependency list is preserved at `legacy/requirements.txt`; it differs from the CPU environment above and from the final protocol.

```bash
python -m repro run-legacy \
  --data-root data/legacy --panel routerbench-0shot \
  --split ood_0 --representation gte --kind knn \
  --workdir runs/routerbench-ood0-gte-knn

python -m repro run-legacy \
  --data-root data/legacy --panel routerbench-0shot \
  --split ood_0 --representation gte --kind prior \
  --workdir runs/routerbench-ood0-gte-prior
```

`knn` produces fixed k=100 and validation-selected kNN. `prior` is the historical soft-family-prior implementation in the Family Adaptation lineage. Each launch copies the unmodified recovered sources into a fresh workspace, points them to external inputs and records feature/code/input hashes. It does not change the repository sources or shared input files. A legacy run cannot establish compliance with the final October protocol.

## Recover the final submission

Place the conference archive in `incoming/` (ignored by Git) and inspect it:

```bash
python -m repro inspect-submission --archive incoming/submission.zip
```

This inventories the package and its dependency/launch files without importing or executing code. Integration must then verify the final cohort, feature conventions, configurations and experimental entry points before replacing the pending status. See [conference integration](conference/README.md).

## Contents

- `legacy/`: recovered historical Python sources, job scripts and configurations. Files retain their original checksums and license text.
- `repro/`: portable input checks, reference controls, original-runner launchers and archive inspection.
- `configs/target-protocol.json`: a partial final-protocol reference, explicitly not a verified executable implementation.
- `provenance/source-manifest.json`: hashes and provenance of every exported original source/configuration file.
- `provenance/experiment-code-inventory.csv`: **scheduler-derived locations** of the final experiment launchers; their contents are not included or verified.
- `validation/`: numerical checks actually executed during repository preparation.

The manuscript and Overleaf project identify the experiments and their target settings. LaTeX compilation, PDF assembly and publication layout are outside this repository's execution requirements.

