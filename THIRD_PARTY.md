# Third-party assets and attribution

## Inter typeface

The presentation assets use Inter, Copyright 2020 The Inter Project Authors,
under the SIL Open Font License 1.1. The original variable font, license and pinned
upstream revision/checksums are retained in `docs/demo/fonts/`. Static instances
used for rendering are generated locally and remain under the same font license.

## NVIDIA reference implementation

The included reference sources come from NVIDIA-AI-Blueprints/llm-router, commit `8a9d3509fbde879d9795258081bab9553b458e04`. The upstream provenance records name the exact paths and SHA-256 checksums. The Apache-2.0 license and NVIDIA notices are retained in the vendor directories.

The routing study adapts BCE to MSE and uses unrestricted utility predictions. Layer selection and preprocessing choices are described in `docs/METHODS.md`; do not describe the utility adaptations as a literal reproduction of every published setting.

## Downloaded public assets

The full source revisions and download selections are in `manifests/assets.json`. Large datasets and model weights are not redistributed in this ZIP.

| Asset | Pinned upstream repository | License information present in inspected asset files |
|---|---|---|
| RouterBench | withmartian/routerbench | The inspected pinned README does not declare a license. Obtain the asset from upstream and consult its terms. |
| SPROUT | CARROT-LLM-Routing/SPROUT | The inspected pinned README does not declare a license. Obtain the asset from upstream and consult its terms. |
| MMLU-Pro metadata | TIGER-Lab/MMLU-Pro | Card declares MIT. |
| MATH metadata | EleutherAI/hendrycks_math | Card declares MIT. |
| MiniLM-L12 | sentence-transformers/all-MiniLM-L12-v2 | Card declares Apache-2.0. |
| GTE-Qwen2-7B-Instruct | Alibaba-NLP/gte-Qwen2-7B-instruct | Card declares Apache-2.0; downloaded custom model source contains upstream notices. |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | Apache-2.0 license file is present. |

Dataset/model creators should be cited in the paper. Third-party dependency licenses remain those of their respective distributions. Upstream authors' names and citations are retained; anonymization concerns the submitting authors, not attribution to unrelated upstream work.

The four short proof-comparison prompts are supplied explicitly as experiment inputs; they do not include model-generated answers. The remaining raw query corpora are fetched from the pinned public sources.

## Original project code

No new open-source license grant is asserted by this packaging operation. This anonymous review artifact is supplied for inspection and reproducibility; the authors must decide the project's public-release license. This statement does not replace any upstream license.
