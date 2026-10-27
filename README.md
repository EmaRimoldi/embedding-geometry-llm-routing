# Embedding Geometry Shapes LLM Routing

**Similar prompts do not always need the same model.**

Code and reproducibility materials for **[Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization](https://arxiv.org/abs/2610.19198)**.

**Leonardo Torti<sup>1,*</sup> · Emanuele Rimoldi<sup>1,*</sup> · Tomer Galanti<sup>2</sup> · Tomaso Poggio<sup>1</sup> · Pierfrancesco Beneventano<sup>1</sup>**

<sup>1</sup> Massachusetts Institute of Technology · <sup>2</sup> Texas A&M University<br>
<sup>*</sup> Main contributors, as noted in the paper.

[Paper](https://arxiv.org/abs/2610.19198) · [Run the experiments](docs/RUN_COMMANDS.md) · [Methods](docs/METHODS.md) · [Results & audit](docs/REPRODUCTION_AUDIT.md) · [Citation](CITATION.cff)

The framework embeds each prompt and selects an LLM to balance answer quality and inference cost. Our main finding: **better representations help simple routers generalize to unseen task families**. Frozen Qwen hidden states improve average held-out routing utility over semantic GTE embeddings.

[![A prompt arrives, a frozen encoder generates its embedding, a lightweight router classifies it, and a model is selected. The sequence repeats automatically.](docs/demo/routing-demo.gif)](docs/demo/README.md)

*An illustrative sequence using the paper's original Figure 1. [Automatic browser demo, static version and sources](docs/demo/README.md).*

![Blue GTE and red L21 curves compare held-out utility for the same kNN-100 router on RouterBench and SPROUT across all 13 cost penalties.](docs/demo/embedding-comparison.svg)

With the same kNN-100 router, L21 raises mean held-out utility from 0.6615 to 0.7007 on RouterBench. On SPROUT, it rises from 0.7529 to 0.7731. The curves show every evaluated penalty, including the low-penalty region where GTE is slightly better on RouterBench.

In a separate comparison on RouterBench, L21 Family Adaptation raises mean reward from 71.04% to 74.63% versus a source-selected single model. Mean candidate-model inference cost falls from $2.07 to $1.55 per 1,000 queries.

These means cover all 13 frozen cost penalties under held-out-family evaluation. Reward includes graded tasks, and cost excludes encoder/router overhead. [Data, metric definitions and plot sources](docs/demo/README.md).

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

The audit passes 5,493 checks against the pinned manuscript reference and includes CPU/CUDA component runs. Full-scale fits and bootstraps have not all been independently rerun; see [coverage and limitations](docs/REPRODUCTION_AUDIT.md) and the [result index](docs/RESULT_INDEX.md).

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
