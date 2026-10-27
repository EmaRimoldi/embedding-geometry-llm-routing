# Embedding Geometry Shapes LLM Routing

**Similar prompts do not always need the same model.**

Code and reproducibility materials for **[Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization](https://arxiv.org/abs/2610.19198)**.

**Leonardo Torti<sup>1,*</sup> · Emanuele Rimoldi<sup>1,*</sup> · Tomer Galanti<sup>2</sup> · Tomaso Poggio<sup>1</sup> · Pierfrancesco Beneventano<sup>1</sup>**

<sup>1</sup> Massachusetts Institute of Technology · <sup>2</sup> Texas A&M University<br>
<sup>*</sup> Main contributors, as noted in the paper.

[Paper](https://arxiv.org/abs/2610.19198) · [Run the experiments](docs/RUN_COMMANDS.md) · [Methods](docs/METHODS.md) · [Results & audit](docs/REPRODUCTION_AUDIT.md) · [Citation](CITATION.cff)

[![Animated Figure 1: a prompt arrives, its embedding is generated and classified, and model 2 is selected. Measured RouterBench comparison: quality 71.04% to 74.63%; inference cost $2.07 to $1.55 per 1,000 queries.](docs/demo/routing-demo.gif)](docs/demo/README.md)

*The paper's original Figure 1, animated. Prompt and model IDs are illustrative; counters show held-out RouterBench means across all 13 cost penalties, comparing a source-selected single model with L21 Family Adaptation. Quality denotes benchmark reward. [Interactive version, static figure & sources](docs/demo/README.md).*

## Why geometry matters

Semantic neighbors can be unreliable guides to model performance on unseen task families. We study this failure and improve generalization with **Qwen2.5-7B-Instruct layer-21 representations** and **Family Adaptation**, a simple router that combines source-family performance profiles using predicted family probabilities.

The experiments cover **RouterBench: 34,771 queries, 11 models** and **SPROUT: 43,887 queries, 13 models**, with frozen splits, seeds and cost penalties.

## Reproduce

Check the bundled numerical evidence with Python 3.11; no GPU or external dependencies required:

```bash
git clone https://github.com/EmaRimoldi/routing-theory-reproducibility.git
cd routing-theory-reproducibility
python3 scripts/verify_release.py
python3 scripts/reproduce_results.py
python3 scripts/check_paper_results.py
```

For fresh data, embeddings and router training, follow the **[full execution guide](docs/RUN_COMMANDS.md)**. Public assets and environments are pinned; neural components require Linux and CUDA.

The audit passes **5,493 checks against the pinned manuscript reference** and includes CPU/CUDA component runs. Full-scale fits and bootstraps have not all been independently rerun; see [coverage and limitations](docs/REPRODUCTION_AUDIT.md) and the [result index](docs/RESULT_INDEX.md).

## Citation

If you use this code or build on the study, please cite the paper. GitHub citation metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@misc{torti2026embeddinggeometry,
  title         = {Embedding Geometry Shapes {LLM} Routing: Tracing Failure Modes and Improving Generalization},
  author        = {Leonardo Torti and Emanuele Rimoldi and Tomer Galanti and Tomaso Poggio and Pierfrancesco Beneventano},
  year          = {2026},
  eprint        = {2610.19198},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.19198}
}
```

Third-party code and asset notices: [THIRD_PARTY.md](THIRD_PARTY.md). After edits, refresh the export with `python scripts/build_submission.py` and commit its updated manifest.
