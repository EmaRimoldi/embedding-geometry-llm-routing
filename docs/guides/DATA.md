# Data and model access

Commands for the current checkout run from the [scientific workspace](../../experiments/README.md). From the repository root, enter it with `cd experiments`.

Raw corpora, model weights and large generated arrays are obtained or generated
locally. The repository includes fixed splits, four illustrative prompts and compact
reference scores. It does not redistribute the complete third-party datasets.

## Pinned sources

The executable inventory is [manifests/assets.json](../../experiments/manifests/assets.json).
Every revision below is a full upstream commit, not a moving branch.

| Asset | Upstream repository | Revision | Local destination |
| --- | --- | --- | --- |
| routerbench | [withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench/tree/784021482c3f320c6619ed4b3bb3b41a21424fcb) | `784021482c3f320c6619ed4b3bb3b41a21424fcb` | `datasets/routerbench/raw` |
| sprout | [CARROT-LLM-Routing/SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT/tree/d54e3cda2189ca1ed13a877a44ed34736c4b6920) | `d54e3cda2189ca1ed13a877a44ed34736c4b6920` | `datasets/sprout/raw` |
| minilm-l12 | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/tree/a50ef00143b4d5391434df20ae11632588ac25be) | `a50ef00143b4d5391434df20ae11632588ac25be` | `embedders/minilm-l12` |
| gte-qwen2-7b | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/tree/a8d08b36ada9cacfe34c4d6f80957772a025daf2) | `a8d08b36ada9cacfe34c4d6f80957772a025daf2` | `embedders/gte-qwen2-7b` |
| qwen2.5-7b-instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28) | `a09a35458c702b33eeacc393d103063234e8bc28` | `embedders/qwen2.5-7b-instruct` |
| mmlupro-subject-metadata | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be) | `b189ec765aa7ed75c8acfea42df31fdae71f97be` | `datasets/sprout/source_metadata/mmlupro` |
| math-subject-metadata | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3) | `21a5633873b6a120296cce3e2df9d5550074f4a3` | `datasets/sprout/source_metadata/math` |

Download selections, including file patterns, are defined in the manifest.
The production downloader resolves those revisions, downloads selected files and
runs the pinned HF CLI remote-checksum verification. Status and detailed checks are
written under `.setup/`; compute jobs use the completed local cache offline.

## Obtain and prepare inputs

Use Linux x86-64 and Python 3.11 for the full study. From a new release copy:

```bash
bash scripts/bootstrap.sh
source jobs/env.sh
.venv-download/bin/python -B src/download_assets.py
.venv/bin/python -B src/prepare_data.py routerbench
.venv/bin/python -B src/prepare_data.py sprout
.venv/bin/python -B scripts/reproduce_outcome_controls.py
```

The downloader fetches models as well as data. The two benchmark raw payloads
total about 680 MB; auxiliary metadata, tokenizers, caches and prepared files add
space. The verified model-weight inventory totals 45,815,257,928 bytes. Allow more
than 100 GB for complete experiment outputs. Network availability and any upstream
authentication or license acceptance are prerequisites for downloading; tokens
remain in the provider credential store, outside the repository.

For a data/tokenizer-only preparation check on Linux, without model weights or
CUDA packages, create the CPU environment in a new release copy:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r environment/cpu-requirements.lock
.venv/bin/python -m pip check
.venv/bin/python -B scripts/initialize_release.py --stage layout
.venv/bin/python -B scripts/check_public_inputs.py --download --prepare
```

This uses the CPU environment and prepares both datasets. Use a new copy:
`--prepare` rejects existing prepared manifests. This narrower check does not
establish that the full model downloads or encoder extraction have completed.

## Canonical prepared data

| Dataset | Queries | Candidate models | Families | Matched MiniLM subset |
| --- | ---: | ---: | ---: | ---: |
| routerbench | 34,771 | 11 | 7 | 34,659 |
| sprout | 43,887 | 13 | 6 | 41,324 |

Preparation preserves released candidate-model rewards and costs. SPROUT costs
are computed from released token counts and the frozen price table. Re-querying
current model APIs or changing current prices would define a different input.
Preparation records exact exclusions, ordering and model names in each manifest.
The original 34,661-query export is superseded by the 34,771-query RouterBench
cohort; use the pinned preparation route and fixed row splits.

Each `datasets/<dataset>/prepared/` directory contains:

| File | Meaning |
| --- | --- |
| `panel.npz` | Aligned reward/cost matrices and family, topic, subgroup, format, query-key and MiniLM eligibility arrays |
| `queries.jsonl` | Canonical prompt rows in exactly the order used by the panel and embeddings |
| `excluded.jsonl` | Excluded rows and reasons |
| `manifest.json` | Model order, cohort counts, exclusions and source fingerprints |

The panel uses `allow_pickle=False` when loaded. Do not reorder its rows or replace
the canonical files with an independently sampled subset. Fixed IID/OOD outer and
inner row indices are in `configs/baseline_splits/`. Their family-stratified 70/30
IID and complete-family OOD definitions are described in [METHODS.md](../reference/METHODS.md).

## Expected SHA-256 values

The canonical reference is
[public_data_preparation.json](../../experiments/release_provenance/reviewer_smoke/public_data_preparation.json).
Both fresh CPU verification commands check all eight values before computing.

| Dataset | File | SHA-256 |
| --- | --- | --- |
| routerbench | `panel.npz` | `b3fc98d24f92279cf5881b95996645ab5bc30bb797b49c6ef8fa55da70c13a0f` |
| routerbench | `queries.jsonl` | `9aafd0134ece05f17ea2c15c2db1c2948d1583dc0b3fab6ba5fc968e57d9d4bd` |
| routerbench | `excluded.jsonl` | `205d589b57185bba271f4602d711ebb7012fb9bf50b16d356bfc3f5a67f9ff6d` |
| routerbench | `manifest.json` | `557bc48609f0ff15040bdf83120770ef7da8775c52a640709a85d1d3168b4f88` |
| sprout | `panel.npz` | `e125721301ce056fdaaebe2c51ac0fdfca4ab460825ce8bd1f8a4be1d302a972` |
| sprout | `queries.jsonl` | `977295d4650e20ecbb1b1b14d9250d2bec9b42efb0633a21d98098157c78c0f9` |
| sprout | `excluded.jsonl` | `c6d255d6c369056092b608a544f39a2a89d0b3575cf6db025fb3c46573f5485b` |
| sprout | `manifest.json` | `a1f6bb0cad2e88a49489d2b043a1ec5204d54bf32e1139a46967bf3e09cbfec7` |

Raw data/tokenizer fingerprints are recorded in
[public_inputs.json](../../experiments/release_provenance/current_checks/public_inputs.json);
full weight fingerprints are in
[model_weight_checksums.json](../../experiments/manifests/model_weight_checksums.json).

## Terms and availability

The project CC BY 4.0 license covers original project material. Dataset and model
terms remain those of their respective creators; see [THIRD_PARTY.md](../THIRD_PARTY.md).
The inspected pinned RouterBench and SPROUT README files do not declare a license.
This repository supplies retrieval instructions without asserting permission to
relicense or redistribute their complete payloads. Consult upstream terms before
redistributing them.

If an upstream file becomes unavailable, a checksum detects a substitute but
does not recover the original bytes. Report the asset name and revision in a
GitHub issue. A later mirror must preserve provenance, hashes and applicable
rights. No independent mirror of the full datasets or generated arrays is
published by this release.

## Fresh CPU checks

```bash
.venv/bin/python -B scripts/reproduce_outcome_controls.py
.venv/bin/python -B scripts/reproduce_cpu_diagnostics.py
```

The first recomputes the source-selected single-model and realized-oracle controls.
The second executes all 20 IID known-metadata lookup reports and the complete MMLU
topic diagnostic, including its 1,000-draw paired bootstrap at each penalty. It
runs unchanged production sources in a new `.cache/` workspace, compares all
scientific report fields with frozen evidence, and writes a compact report under
`results/reproduced/`. The source fingerprints of the new run are recorded separately
from historical provenance. These analyses need no embeddings or GPU; their import
dependencies are included in the compute/CPU environments.

Embedding extraction, fitted routers and other bootstrap jobs follow
[RUN_COMMANDS.md](RUN_COMMANDS.md). The [experiment map](../reference/EXPERIMENTS.md) identifies
their prerequisites and expected outputs.
