# Embedding Geometry Shapes LLM Routing

**Leonardo Torti<sup>1,&#42;</sup> · Emanuele Rimoldi<sup>1,&#42;</sup> · Tomer Galanti<sup>2</sup> · Tomaso Poggio<sup>1</sup> · Pierfrancesco Beneventano<sup>1</sup>**

<sup>1</sup> Massachusetts Institute of Technology · <sup>2</sup> Texas A&M University<br>
<sup>*</sup> Main contribution.

[Paper](https://arxiv.org/abs/2610.19198) · [Quickstart](#quickstart) · [Experiments](docs/reference/EXPERIMENTS.md) · [Documentation](docs/README.md) · [Citation](#citation)

**The right model starts with the right representation.**

Research code and reproducibility materials for *Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization*. The pipeline embeds each prompt, then selects an LLM to balance quality and cost. It compares semantic embeddings with frozen Qwen hidden states, lightweight routers and SharedTrunkNet adaptations.

[![A prompt arrives, a frozen encoder generates its embedding, a lightweight router classifies it, and a model is selected.](docs/demo/routing-demo.gif)](docs/demo/README.md)

![Blue GTE and red L21 curves compare utility for the same kNN-100 router on RouterBench and SPROUT across all 13 cost penalties.](docs/demo/embedding-comparison.svg)

With the same kNN-100 router, L21 raises mean held-out utility from 0.6615 to 0.7007 on RouterBench and from 0.7529 to 0.7731 on SPROUT. The curves include every evaluated penalty. In a separate RouterBench comparison, L21 Family Adaptation raises mean reward from 71.04% to 74.63% versus a source-selected single model, while mean candidate-model inference cost falls from $2.07 to $1.55 per 1,000 queries. See the [methods](docs/reference/METHODS.md) and [result index](docs/reference/RESULT_INDEX.md) for definitions and scope.

## Quickstart

Use Python 3.11 to verify the bundled evidence. These commands require no GPU, downloads or third-party Python dependencies:

```bash
git clone https://github.com/EmaRimoldi/embedding-geometry-llm-routing.git
cd embedding-geometry-llm-routing
python3 -B scripts/check_docs.py
python3 -B scripts/verify_release.py
python3 -B scripts/reproduce_results.py
python3 -B scripts/check_paper_results.py
```

With Make installed, `make verify` runs the same checks. It validates repository paths and exported bytes, reconstructs saved aggregates and checks the pinned manuscript values. It does not fit new models or repeat bootstrap sampling.

## Reproduce experiments

The full workflow uses Linux, Python 3.11, pinned public assets and CUDA for neural components. Follow the [execution guide](docs/guides/RUN_COMMANDS.md) in stage order; each stage lists its prerequisites and outputs.

| Task | Guide |
| --- | --- |
| Choose an experiment or verification level | [Experiment map](docs/reference/EXPERIMENTS.md) |
| Download and prepare canonical inputs | [Data and model access](docs/guides/DATA.md) |
| Generate embeddings, fit routers and run diagnostics | [Execution commands](docs/guides/RUN_COMMANDS.md) |
| Plan CPU, GPU and storage requirements | [Resources](docs/guides/RESOURCES.md) |
| Inspect result files and numerical tables | [Result index](docs/reference/RESULT_INDEX.md) and [saved tables](docs/reference/SAVED_TABLES.md) |

## Repository

The root separates implementation, protocols, automation, tests, documentation and retained research evidence:

```text
embedding-geometry-llm-routing/
├── src/                Scientific implementations and report producers
├── configs/            Fixed protocols, hyperparameters and row splits
├── scripts/            Setup, verification and packaging tools
├── jobs/               Slurm launchers
├── tests/              Regression and component tests
├── docs/               Guides, methods, results and audits
├── reproducibility/    Pinned environments, manifests, provenance and evidence
├── README.md           Overview, quickstart and paper citation
├── CONTRIBUTING.md     Development and validation workflow
├── requirements.lock   Pinned compute dependencies
├── Makefile            Convenient commands from the repository root
├── CITATION.cff        Machine-readable citation
└── LICENSE             CC BY 4.0
```

Run all commands from the repository root. Start with the [documentation index](docs/README.md) for execution, the [reproducibility index](reproducibility/README.md) for retained inputs and checks, or the [directory and retention guide](docs/reference/STRUCTURE.md) for the full map. Downloaded datasets, weights, generated arrays and new results are local runtime material excluded from Git.

## Validation and limitations

The saved-evidence checks cover 5,493 manuscript quantities and settings across all 44 pinned table labels. CPU/CUDA component executions and fresh full-input runs are documented separately in the [reproduction audit](docs/validation/REPRODUCTION_AUDIT.md). Historical numerical agreement is not established for every experiment; the audit preserves the observed differences and incomplete coverage.

For development, install the pinned compute or CPU environment described in the [setup guide](docs/guides/RUN_COMMANDS.md#platform-and-setup), then run:

```bash
make check-docs
make test
make smoke
```

The smoke suite uses small synthetic inputs. The [documentation index](docs/README.md) distinguishes current validation from historical reviewer checks.

## Contributing

See [contributing guide](CONTRIBUTING.md) for change validation and issue reports. After edits, `make manifest` refreshes the current file manifest without rebuilding an existing release ZIP; run `make verify` before committing. [Release instructions](docs/releases/RELEASE.md) cover intentional new-version packaging.

## License

Original project material uses **[CC BY 4.0](LICENSE)**. Third-party sources, assets, datasets and models retain their own terms; see [THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Citation

If you use this code, data processing or results, please cite the paper below. Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{torti2026embedding,
  title={Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization},
  author={Leonardo Torti and Emanuele Rimoldi and Tomer Galanti and Tomaso Poggio and Pierfrancesco Beneventano},
  year={2026},
  eprint={2610.19198},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.19198}
}
```
