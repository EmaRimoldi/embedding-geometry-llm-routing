# Frozen study definitions

## Data and evaluation

RouterBench uses the seven included families and SPROUT uses six native families. Preparation retains released rewards and costs (SPROUT costs are computed from released token counts and the documented prices). Invalid, duplicate, empty or over-context inputs are handled by the frozen preparation code. There is no silent truncation or generated OpenHermes topic labeling.

Five family-stratified 70/30 IID splits per dataset and seven/six complete-family OOD splits produce 23 outer evaluations. Exact rows are in `configs/baseline_splits/`. Validation and fitting use source/training rows. The two datasets use their own frozen 13-penalty grids; utility is reward minus penalty times cost.

Primary OOD summaries pool test queries, while equal-family summaries are explicitly marked. IID summaries average the five split means. Means across penalties average the complete prescribed grid and are grid-dependent. Do not treat penalties as independent observations.

## Baselines and representations

GTE final embeddings and raw Qwen2.5-7B-Instruct hidden_states[21] use the frozen encoder implementation and aligned query cohorts. L21 is a fixed representation; its original exploratory selection and later source-only layer-selection controls are distinct. GTE hidden layer 21 is a separate within-encoder diagnostic.

MiniLM is retained as an auxiliary geometry and clustering comparator. Its frozen encoder uses FP32, attention-mask mean pooling and complete prompts of at most 512 tokens, including special tokens, without truncation. Its tokenizer defines the `shared_minilm` eligibility mask used for matched geometry comparisons with GTE and Qwen. The routing baselines and NVIDIA adaptations use GTE or Qwen representations on the primary cohorts.

The baseline configuration defines fixed-k kNN-100, validated kNN, Family Adaptation, single-model source selection, clustering baselines and native-label lookup controls. Reduced MLP uses the two/three 100-unit hidden-layer grid and its fixed reduced validation folds; display it as MLP. Original incomplete/full-validation attempts are historical and are not substituted for the audited reduced run. Ridge and the complete layer sweep have separate configurations. GraphRouter is excluded.

## NVIDIA adaptations: three distinct variants

The final utility adaptation lives in `src/nvidia_utility/` and `configs/nvidia_utility_v1.json`. For each outer split, candidate model and cost penalty, it uses Ridge(alpha=1) probes with five shuffled source-only KFold folds, selecting among cached raw Qwen layers 15–28 using the reference ternary-search comparisons on negative MSE. StandardScaler/PCA100 is fitted once on the outer source pool and held fixed during probe CV. This is not inner-fold-fitted preprocessing.

Selected 100-dimensional blocks are concatenated in model order. The pinned official shared network has hidden dimensions 256/128 and dropout 0.3/0.2. Its loss is adapted from BCE to MSE and outputs are unrestricted utility. Ten seeds 42–51 are fit, retaining five by internal validation MSE. Optimizer and stopping settings are retained in the frozen config. All 23 outer evaluations and 13 penalties are included: 299 ensemble scores and 2,990 final networks.

The earlier `src/nvidia/` pipeline selects one common layer by mean source utility and includes a separate fixed-L21 control with its original repeated-block parameterization. Those comparisons remain historical controls. Neither utility adaptation implements published Fisher selection.

CPU inference results under the earlier NVIDIA path use precomputed raw embeddings, four CPU threads, matched queries and batches 1/128. They exclude encoder runtime and do not measure the final per-target utility pipeline. Setup measurements include integrity verification. Fixed-k retrieval and validated-kNN search must not be combined under a fixed-k training-time label.

## Diagnostics and uncertainty

The GTE family-transfer diagnostic centers predicted and observed utility across candidate models and ranks each retrospective lowest-error source family by its unrestricted k100 retrieval share. It is not a deployable test-selected router.

The crossed MMLU format experiment changes prompt layout and requested response encoding while preserving question/options. Topic is observed, not causally manipulated; the experiment is not a held-out-family routing intervention. The proof twins are illustrative embedding-only examples, with no measured answer cost and no calibrated cross-encoder absolute-cosine comparison.

The later answer-format control restricts semantic GTE retrieval to the existing coarse format label. The reward/cost hybrids retain all four predictor combinations and every penalty. The control assumes a known format label and does not isolate answer format as the sole mechanism.

Routing confidence intervals use the frozen paired query-bootstrap protocol, usually 2,000 draws. OOD draws preserve held-out families and query weights; IID draws keep repeated appearances of a query together. These intervals are conditional on fixed fits and observed families, not training variability or a population of unseen families. Existing 1,000-draw lookup-disagreement intervals are explicitly retained. Geometry/layer-selection intervals computed outside the publication scope are not promoted into the paper tables.
