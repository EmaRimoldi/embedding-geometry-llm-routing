# Scientific consistency audit

The review checked the released source, configurations, compact evidence and split files. The original experiment workspace was read only. It did not retrain the published models or recompute confidence intervals. The metadata update to the compact comparison preserves all existing rows and numerical values.

## Confirmed consistency

- **Cost and utility:** SPROUT costs are computed as input/output token counts times the fixed USD-per-million-token prices, divided by one million (`src/prepare_data.py`). Routing utility is reward minus the prescribed penalty times USD/query cost. The inspected kNN, Ridge, MLP and SharedTrunkNet paths do not subtract cost twice. Cost/reward diagnostics keep their different error units explicit.
- **Splits and source-only selection:** All 23 outer train/test partitions are disjoint and exhaustive. Each query appears in exactly one OOD test family. Every inner fit/validation partition lies inside outer training. kNN candidate selection, clustering, KM-Prox and Ridge use source validation outcomes. Reduced MLP's chosen inner indices match the deterministic SHA256 rule; its two OOD validation sets are whole source families.
- **Aggregation:** IID test sizes are constant across the five splits: 10,432 for RouterBench and 13,167 for SPROUT. Therefore the compact query-weighted calculation equals the stated average of five split means. OOD routing summaries pool queries; explicit equal-family summaries and family-transfer rank summaries have their separately declared weighting. Means across penalties retain all thirteen fixed values.
- **Fixed versus validated kNN:** kNN-100 always uses k=100. Validated kNN chooses k on source validation separately for each penalty. They have separate saved choices and result names. The fixed method's runtime should not be described using the combined validation-search runtime.
- **Reward versus relative utility:** The original geometry transfer diagnostic measures centered reward prediction error. The later family-utility diagnostic centers predicted and observed reward-minus-penalty-times-cost across candidate models and evaluates every penalty. Its quadratic MSE expression is audited against direct centered-utility MSE. The best source is selected retrospectively for the diagnostic, not deployed using test-selected information.
- **Confidence-interval definitions:** Paired bootstrap draws keep methods and penalties together. OOD resampling preserves held-out families and their sizes; IID resampling keeps repeated appearances of each query together. These intervals condition on fixed fits and observed benchmark families and do not estimate training variability or variation over a population of new families.
- **Final NVIDIA protocol:** Final per-target utility selection uses the documented Ridge/MSE and ternary-search procedure. Scaling/PCA are fitted once on the outer source pool and held fixed during inner probing; this is an explicit protocol choice, not inner-fold-fitted preprocessing. Earlier fixed-L21/common-layer controls and their inference timings remain distinct from the final pipeline.
- **Comparison information regimes:** Both main-comparison CSVs explicitly label ordinary source-trained routers, known-metadata diagnostics, the realized oracle and the diagnostic clustering control. The oracle uses test outcomes; known-family/cell lookups require true test metadata. All original rows are retained.

## Corrected bootstrap rank tie inconsistency; frozen statistics unchanged

`src/diagnostics/family_utility_transfer.py` computes Top-1 as average retrieval rank at most one. `src/diagnostics/appendix_ci/family_rank.py` counts any best-error source tied for maximum retrieval share. For two tied maximum shares, the first convention assigns average rank 1.5 and no Top-1 credit, whereas the second allows Top-1 credit. The table caption's phrase "highest retrieval share" does not resolve this difference explicitly.

The repository now makes the bootstrap use the point estimator's existing rule:
Top-1 means average retrieval rank at most one. Thus a tied maximum with rank 1.5
receives no Top-1 credit. This keeps resampled statistics consistent with the
original point-estimation function rather than silently using a different estimator.

Regression tests cover `[0.5, 0.5]` with a unique lowest-error source and independently
recompute **all 169 observed family/penalty point estimates** from the saved source
errors and shares. Every original point is unchanged. Frozen evidence was not
edited. Family-rank bootstrap intervals are outside the publication scope and have
not been resampled here; a new run can differ on tied bootstrap draws. The original
submitted bytes remain identifiable through the preserved original manifest.

## Embedding-cache integrity

The baseline loader previously checked feature row count and query-file hash but
did not check the vector-file hash already present in its manifest. Finite changed
vectors could therefore silently change routing results. The loader now checks
that hash before normalization. A regression test changes one feature while keeping
its dimensions and all values finite, and verifies rejection. Intact arrays and the
scientific algorithms are unchanged.

## Small-input geometry edge case

The review identified a self-exclusion boundary in `geometry.Neighbours.get`: clipping k to the total source-pool size before masking a query's own entry can leave too few finite neighbors when the query belongs to a small source pool. The published cohorts do not reach this boundary. The release copy now caps k by the minimum available nonself source count and raises clear errors for an empty eligible source pool or duplicate source rows. Two regression tests were added in `tests/test_geometry.py`: a four-vector pool with self included and requested k=100, and rejection of impossible or duplicate source pools. Both tests passed as part of the ten-test CPU reviewer smoke suite. This boundary correction does not change the frozen ranking conventions above.

## Scope of the compact checks

The compact script verifies exported hashes, re-aggregates saved fold scores, checks reward-minus-penalty-times-cost identities and compares the results against saved summaries. Final per-target utility fold scores are additionally reconstructed from exported result JSONs. These checks are not independent reruns of per-query decisions, fitted models or bootstrap arrays. See `RESULT_INDEX.md` for the included evidence and the full-workflow requirements.
