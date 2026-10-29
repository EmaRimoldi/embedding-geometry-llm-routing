# Final utility pipeline: reviewer audit and small smoke

Run from the extracted package after installing compute dependencies:

```bash
.venv/bin/python -B scripts/smoke_nvidia.py
# Optional, inside an existing GPU allocation:
.venv/bin/python -B scripts/smoke_nvidia.py --device cuda
```

The command uses the actual final utility modules and vendored SharedTrunkNet. It creates an isolated synthetic workspace under `results/reproduced/reviewer_nvidia/run-*/`, leaving shipped source, frozen configurations, saved evidence, and previous runs intact. It never submits a scheduler job. The GPU option requires an available allocated CUDA device.

## Scope of the smoke

The toy experiment has 60 source rows, 12 held-out rows, three candidate models, six candidate layers and two penalties. Feature blocks have the implementation's 100 coordinates. Twenty networks use four epochs and batches of 16; these reduced counts are smoke-only and are not experiment settings.

The command checks:

- Actual per-target utility-column ordering and production batched Ridge predictions against independent scikit-learn Ridge fits for every synthetic target and fold.
- Production ternary selection and visited-layer paths against the exact vendored reference function; repeating selection must use cached scores without any Ridge refit.
- Disjoint source/held-out rows and source-only validation folds; actual feature concatenation retains repeated selected layers and model order.
- The actual adapted SharedTrunkNet fitting core, production prediction helper, best-state validation loss, unrestricted negative predictions, and a save/reload round trip for every checkpoint.
- Production checkpoint verification and ensemble assembly retain the five lowest-validation-MSE seeds out of ten, average raw outputs, and choose their argmax.
- Altering held-out realized costs changes measured utility but does not change routing choices. Realized cost enters the evaluation once and is not subtracted again from predicted utility.

A passing JSON report identifies the exact checks and synthetic settings. This is more than report rendering, but it does not reproduce the paper experiments. Synthetic adapters replace data, cached feature arrays, path roots, and the full-data integrity gate within this process only. `scripts/verify_release.py` checks package file integrity separately. Encoder extraction, PCA fitting, full-data preflight, Slurm orchestration and the production GPU-only worker wrapper are outside the smoke. CPU execution uses the actual `trunk().train_mlp` fitting core rather than `train.fit`, whose wrapper hardcodes CUDA synchronization and CUDA placement.

## Conceptual code review

The final path uses a separate Ridge target for every candidate model and cost penalty, five shuffled source-row folds, and the reference ternary-search comparisons on negative mean fold MSE. Its StandardScaler/PCA transforms are fitted on the outer source pool. The final target is reward minus penalty times cost; the network predicts that utility directly and uses no sigmoid or second cost deduction. Selected feature blocks remain in model order, including duplicates. Parameters are reset from each seed by the vendored trainer; five of ten seeds are retained by source validation MSE.

These are protocol limitations, not newly discovered implementation errors:

- Outer-source PCA includes features from the internal validation folds. It excludes the outer test family, but internal CV is not a fully nested preprocessing estimate. This scope is already stated in `docs/reference/METHODS.md`.
- Ternary search is a reference heuristic and need not find the global best layer on a non-unimodal score curve.
- Seeds use different internal 85/15 validation splits, following the reference ensemble procedure. The reported conditional intervals do not measure uncertainty from refitting those networks.
- The earlier common-layer and fixed-L21 variants use separate selection procedures. Their reported CPU latency is not a measurement of the final per-target pipeline.

No scientific settings or recorded outcomes were changed for this audit. Full fresh-run numerical equivalence remains separate from these synthetic checks.
