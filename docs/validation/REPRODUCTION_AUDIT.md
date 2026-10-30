# Reproduction audit

The active code is now the final conference supplement, identified by archive
SHA-256 `043676877fed7b9ecf4f93da0ba428a5f8dbd0025212873fcebc94470623494e`.
The original archive passed its 1,981-file release-manifest check before any changes.
`release_provenance/conference_original_manifest.json` preserves those original
fingerprints; `release_provenance/release_manifest.json` describes this repository.

The owner stopped the full fresh-experiment campaign on 2026-10-29 and requested
a pipeline review instead. All remaining campaign jobs were cancelled, the
periodic monitor was removed, and existing outputs were preserved. At that point
32 of 46 reduced-MLP result records existed; the published 16-record checkpoint
below remains a separately scoped comparison. Unfinished experiments and full
numerical reproduction remain incomplete, rather than being counted as passes.
Pipeline validation uses regression tests and small synthetic production runs;
it does not resume the full scientific experiment grid.
[`v1_pipeline_review.json`](../../experiments/release_provenance/current_checks/v1_pipeline_review.json)
records the revised scope, source fingerprints and executed pipeline checks.

## Executed checks

| Level | Actual execution | Outcome |
| --- | --- | --- |
| Public inputs | Download and remote-checksum verification of seven pinned assets, 62 selected files, excluding model weights | Passed |
| Production downloader | Unchanged production downloader with pinned HF CLI, two real public tokenizer files; both dependency environments checked | Passed; this small CLI check does not redownload all weights |
| Complete preparation | Production preparation commands for both benchmarks | All eight output hashes match the frozen canonical files; cohorts 34,771/43,887 and matched MiniLM subsets 34,659/41,324 |
| Fresh numeric controls | Single-model selection and realized-oracle controls from newly prepared public outcomes | 598 method scores; 3,588 saved metric comparisons, maximum error zero |
| Complete fresh CPU diagnostics | Unchanged production IID known-metadata lookups and MMLU topic analysis on complete canonical panels, including the prescribed MMLU bootstrap | 20 lookup reports, 520 lookup policy scores, 21 total reports and 24,209 numeric values match; maximum error zero |
| Saved-result reconstruction | Seven result groups, including final per-model SharedTrunkNet JSONs | 23,740 fold-level records, 4,031 reconstructed aggregate rows, saved-aggregate tolerance 1e-10 relative / 1e-12 absolute |
| Printed results | Pinned manuscript values and settings, all 44 table labels | 5,493 checks passed at printed precision; supplied intervals compared without resampling |
| CPU regression tests | Pipeline/geometry, integrity, paper-gate, rank-tie, distribution and fresh-diagnostic/baseline comparison cases | Passed, including 14 baseline-audit cases |
| CPU baseline execution | Actual kNN, Family Adaptation, Ridge, Avengers-Pro and KM-Prox CLI paths on synthetic ID/OOD inputs | 260 score checks; 16 MLP-equivalence and 36 direct-utility Ridge cases |
| CPU/CUDA SharedTrunkNet | Actual probe, ternary search, fitting core, checkpoint reload and seed retention on synthetic inputs | 20 networks and 20 reloads on each device; cost-once and held-out-cost isolation checks passed |
| PCA preprocessing | Actual production PCA100 entry point, both full-SVD and randomized branches | 10 checks per branch, including held-out poisoning and stale-cache rejection |
| CUDA encoders | Actual MiniLM, GTE and Qwen L21/L23/L25 extraction on four illustrative prompts | All vectors finite, completed and hash-consistent |
| Complete fresh encoder extraction | GTE, MiniLM and Qwen layers 15–28 on both complete canonical cohorts | 32 arrays generated; all four GTE/L21 baseline input fingerprints exactly match the original |
| Fresh main baselines and controls, partial | Production kNN, Family Adaptation, Ridge, clustering, KM-Prox and IID lookup paths on their prescribed outer populations | All 250 non-MLP records checked; 240 match scientifically, including all 46 Ridge records; 139 differing fields across ten clustering/KM-Prox records remain under investigation; reduced MLP is reported separately below |
| Initial fresh reduced-MLP checkpoint | New production validation and refits on canonical RouterBench inputs; saved scores, choices and input fingerprints compared | 16/46 records compared; 113 validation-utility fields differ and five alpha choices change; full prediction reconstruction and aggregate comparison remain pending |
| Complete fresh Ridge prediction audit | Production audit reconstructing scores from new saved choices and canonical outcomes, verifying query alignment and recorded alpha selection | All 598 policy scores audited; 598 per-fold rows and 104 aggregate rows reproduce both numerical CSVs byte-for-byte |
| Fresh final-utility preprocessing preflight | Production input checks, source-only PCA reconstruction, reference search and independent Ridge equivalence | Passed across 23 outer evaluations and 322 new source transforms; this checks the new caches internally |
| Complete fresh final-utility probe comparison | Newly computed source-only layer choices, search traces and validation MSE compared with pinned selections | All 23 folds compared: 185 of 3,575 layer choices differ, with 2,833 differing fields overall |
| Complete fresh final-utility training and internal audit | All prescribed final-model fits, saved networks, predictions, seed retention and score reconstruction | 2,990 fits and checkpoint checks, 299 utility scores, 23 successful audit tasks; historical raw results differ, and the aggregate comparator report awaits the reduced-MLP summary |
| Complete fresh predicted-label lookup | Ten IID production folds and their aggregate report | All 1,665 numeric comparisons match exactly; completion fingerprints retained |
| Model identity | Full cached weights compared with pinned public LFS checksums | 12 files, 45,815,257,928 bytes, passed |
| Execution plumbing | Inert-scheduler orchestration and isolated saved-report renderers | 17 orchestration categories; seven regenerated tables byte-identical to their supplied counterparts |

The current machine-readable records are under `release_provenance/current_checks/`.
They are distinct from `release_provenance/reviewer_smoke/`, which describes the
conference package's earlier checks.
The complete fresh CPU diagnostic run is recorded separately in
`release_provenance/current_checks/v1_cpu_diagnostics.json`; it verifies canonical
input bytes and records production-source fingerprints and execution times.

The fresh full-input campaign completed all encoder tasks on 2026-10-27 UTC,
using newly prepared public inputs and verified model weights. It generated GTE
and all 14 Qwen candidate layers for 34,771 RouterBench and 43,887 SPROUT queries,
plus MiniLM on the prescribed 34,659/41,324 compatible subsets. Every eligible
vector passed the production extractor's finite/nonzero and completion checks.
The four complete GTE/L21 vector-file SHA-256 fingerprints match those preserved
in the original baseline input manifest, along with all 13 compared scientific
metadata fields. The other 28 arrays have fresh complete manifests; no historical
byte-equality claim is made for arrays without a matching original fingerprint.

[`v1_full_encoders.json`](../../experiments/release_provenance/current_checks/v1_full_encoders.json)
records all 32 manifests and the four comparisons. Extraction computes each hash
after flushing the complete array. Production baseline preparation also independently
rehashed all four input arrays and verified the canonical panels and frozen splits.
This is an encoder-stage checkpoint. New PCA fitting, final-utility preprocessing,
probe selection and final-model training have since completed. Reduced MLP,
historical trained-router, geometry and uncertainty comparisons remain incomplete.

The non-MLP fitted-baseline checkpoint, collected on 2026-10-29 UTC, is recorded in
[`v1_fresh_baselines.json`](../../experiments/release_provenance/current_checks/v1_fresh_baselines.json).
It compares 250 of the prescribed 296 records, including all 46 Ridge evaluations
across both benchmarks, representations and IID/OOD populations. All Ridge scores
and selections match exactly. The separate production audit also reconstructed
all 598 policy scores from the new saved choices and outcomes, verified test-row
and query identities and checked recorded alpha selection. The resulting 598-row
per-fold and 104-row aggregate CSVs match the frozen tables byte-for-byte.
[`v1_ridge_predictions.json`](../../experiments/release_provenance/current_checks/v1_ridge_predictions.json)
records the hashes and scope. This does not independently refit/reload the
regressors or compare omitted original query-level predictions.

Of the 250 compared records, 240 have exact scientific-field agreement. Across all
records, 230,189 of 230,328 numeric fields match exactly; 139 fields differ beyond
the declared tolerances in ten clustering/KM-Prox records. The full difference
list distinguishes validation/selection changes from test-score changes.
At penalty 10, GTE/SPROUT IID fold 1 selects 64
rather than 128 clusters and reports Avengers-Pro utility 0.8470125159489633
rather than 0.8455388292321713. At penalty 0, L21/RouterBench OOD fold 5 reports
Avengers-Pro reward 0.73125 rather than 0.72. The KM-Prox mean-utility
difference at penalty 150 is about 5.64e-7 and is retained as a difference.
Their cause is not yet established. These are actual differences, not a successful reproduction claim;
no data, selection grid or tolerance was changed to remove them.

The 6,500 policy scores include prompt routers, clustering/metadata diagnostics
and single-model/oracle controls; they are not 6,500 independent fitted models.
The 138 source/config hash differences match transitions already recorded in
`release_provenance/source_changes.json`, and the new source bytes were checked
against that registry. These provenance differences and all scientific differences
remain explicit in the report. The complete main-baseline comparison has not passed.

The reduced-MLP comparison at 2026-10-29 06:27 UTC adds 16 newly fitted
RouterBench records: all five IID folds and OOD folds 0–2 for both canonical
representations. The overall partial comparison therefore covers 266/296
prescribed records. All saved candidate utilities and selected settings in the
ten IID records match exactly. The reader's OOD folds 1 and 2 instead differ in
113 validation-utility fields, with five alpha choices changing in OOD fold 2.
These differences exceed the predeclared tolerances; the candidate grid,
inner-fold indices, seeds and scientific settings were not changed.

At penalty 150 in L21/RouterBench OOD fold 2, the selected alpha changes from 1
to 0.0001 and the recorded test utility changes from 0.6786014568810994 to
0.5715887761402112. The other changed choices include increases as well as
decreases in test utility, so this example is not a benchmark-wide mean.
The complete canonical panel, query and preparation hashes, frozen split hash
and L21 vector-file hash were rechecked for this case. Its saved aggregate
arithmetic is consistent with its reward/cost and per-family summaries.
[`v1_reduced_mlp_checkpoint.json`](../../experiments/release_provenance/current_checks/v1_reduced_mlp_checkpoint.json)
retains the five changed choices, raw values and file fingerprints. This is an
initial result-file comparison: the complete reduced-phase prediction audit,
full baseline aggregate report and manuscript printed-precision comparison
remain pending. The cause is not established, and variation is not confined
to final refitting because validation scores also differ.

The supplied conference archive and active checkout contain byte-identical
`src/baselines/batched_mlp.py` files, but the historical results name a different
trainer hash. This pre-existing discrepancy is recorded alongside the MLP
checkpoint. The historical trainer bytes have not been recovered. A hash
difference alone does not establish changed training semantics or explain the
numerical differences. Primary outputs and historical evidence are preserved.

All 28 source-PCA scheduler packs exited successfully. The subsequent complete
utility preflight exceeded its 30-minute allocation before producing its passed
marker, so the dependent probe and training jobs did not start. A retry with a
four-hour allocation passed with the same scientific checks: 23 outer evaluations
and 322 new source transforms were checked, including independently reconstructed
training/test rows, reference ternary search and independent Ridge objectives.
[`v1_utility_preflight.json`](../../experiments/release_provenance/current_checks/v1_utility_preflight.json)
records that internal validation. It does not certify that the new PCA caches or
downstream choices equal the historical ones. All 23 probe tasks and both final
GPU workers have now completed, with no continuation workers remaining.

All 23 new probe records have been compared with the frozen selections. They
differ in 185 of 3,575 selected layers and in 2,833 fields overall, including
validation MSE and search traces. Scientific protocol settings match; the changed
runtime NVIDIA-config fingerprint is recorded separately. All 1,105 comparable
MSE values at unchanged selected canonical layers 21/23/25 match exactly. The
other 2,470 MSE values differ: 2,285 retain the same noncanonical selected layer
and 185 have a different selected layer. This pattern does not establish
whether raw extraction, PCA fitting, runtime differences or another cause explains
the deviation. Historical cached-input fingerprints are the next comparison.
[`v1_utility_probe_comparison.json`](../../experiments/release_provenance/current_checks/v1_utility_probe_comparison.json)
retains every difference and the complete 23/23 selection coverage.

The final-model workers generated all 2,990 prescribed fits. All 23 independent
production audit tasks completed successfully, checking saved checkpoints against
predictions, query alignment, validation MSE, retained seeds, ensembles and 299
utility scores. This establishes internal consistency of the new execution.
It does not establish agreement with the original execution: 105 of 299 retained
seed sets differ; another 118 contain the same five seeds in a different order.
In total, 7,698 of 13,335 compared numeric result fields differ.
[`v1_final_utility.json`](../../experiments/release_provenance/current_checks/v1_final_utility.json)
preserves those differences and prediction fingerprints.

Mean utility, weighted over test queries within each penalty and then averaged
over all 13 penalties, is:

| Dataset | Protocol | Original | Fresh | Fresh minus original |
| --- | --- | ---: | ---: | ---: |
| RouterBench | IID | 0.737972293 | 0.738044253 | +0.000071960 |
| RouterBench | OOD | 0.694920043 | 0.690112512 | -0.004807531 |
| SPROUT | IID | 0.818570330 | 0.818403008 | -0.000167322 |
| SPROUT | OOD | 0.782869138 | 0.782003154 | -0.000865983 |

None of these four means matches at six decimal places. This explicit rounding
comparison is separate from manuscript-specific printed precision and scientific
interpretation. No historical score, candidate, seed or tolerance was changed.
The aggregate comparison job stopped when it reached the absent reduced-baseline
`scores.csv`, after checking the 299 scores and 2,990 fit records. Its historical
failure and the successful per-fold audits are retained. The launcher now defers
that comparison until the complete baseline report exists and submits it with
an `afterok` dependency on an unfinished utility audit array, or verifies all 23
successful accounting records before submitting after that array has ended.
Either completion order is
supported, and a file lock prevents duplicate report submissions. Regression
coverage includes the previously failing missing-baseline case, partial summaries,
both completion orders, concurrent calls, retired job IDs, failed audit tasks
and scheduler failure. This orchestration
correction changes no fitting, selection, scientific settings or existing results.
It applies to fresh run copies; existing provenance is not rebound to new sources.

The fresh predicted-label lookup stage completed all ten folds and its aggregate
report. All 1,400 per-fold and 265 aggregate numeric values match exactly, as do
the compared nonnumeric fields. Runtime and scheduler identifiers are excluded;
result and prediction completion fingerprints remain in
[`v1_deployable_lookup.json`](../../experiments/release_provenance/current_checks/v1_deployable_lookup.json).

All six isolated, unchanged-protocol repetitions are complete. For GTE/SPROUT
IID fold 1 clustering, it differs in 157 compared fields from the primary new run
and in 140 from the frozen reference. At penalty 10 it selects 128 clusters,
compared with 64 in the primary run, and reports utility 0.8456225896939318.
The code, input sources, splits, random seed, candidate grid and four-thread
settings were preserved; the repetition records its CPU and numerical libraries.
This establishes variation across the tested executions, without establishing
whether hardware, parallel reductions or numerical/tie sensitivity caused it.
The original primary outputs are retained. L21/RouterBench OOD fold 2 clustering
matches the primary execution and still differs from the frozen reference.
OOD fold 4 clustering and KM-Prox match the frozen reference while differing from
the primary execution. The other repetitions retain their differences from both.
[`v1_cluster_repetitions.json`](../../experiments/release_provenance/current_checks/v1_cluster_repetitions.json)
records every comparison, including the unfavorable outcomes. These six cases
do not cover every discrepancy in the complete 250-record baseline checkpoint.

`scripts/compare_baseline_results.py` makes this comparison reusable and refuses
to count missing experiments as matches. Fourteen adversarial tests cover
changed scores and selections, invalid counts/shapes, nonfinite values, incomplete
coverage, provenance classification and protected inputs. Matching saved scores
and selections does not establish identical omitted per-query predictions or
model parameters. Printed paper precision and scientific interpretation remain
separate checks.

CI fixes Ubuntu 24.04, Python 3.11.17 and the exact Node-24 action commits instead
of floating runner/Python/action versions. Local and cluster executions recorded
here used Python 3.11.15; the original study reports 3.11.16. These patch-version
differences are recorded rather than presented as an identical original environment.

## Corrections and cleanup

1. **Embedding integrity:** the baseline loader now checks the recorded vector-file
   SHA-256 as well as row count and query-file hash. A changed but correctly shaped,
   finite cache is rejected before fitting. The regression failed before this fix.
2. **Rank bootstrap estimator:** Top-1 under bootstrap now uses the existing point
   estimator's average-rank rule. Tied maximum shares therefore do not receive
   Top-1 credit. All 169 observed family/penalty point estimates remain unchanged.
   Frozen evidence is untouched; future family-rank bootstrap draws with ties can
   differ from the earlier, inconsistent auxiliary intervals.
3. **Obsolete code removed:** the 500-file September export and its duplicate launch
   layer were replaced by the actual conference sources. The old export remains in
   the first Git commit. The obsolete manuscript-only figure builder was removed;
   scientific numerical producers, geometry plots, source dependencies and notices
   remain available.
4. **Repeatable verification:** the pinned paper-reference registry, strict numeric
   comparison, public-input/weight checks and CI workflow make these checks rerunnable.
   No saved outcome was edited to make a comparison pass.
5. **Complete Git export:** the old unanchored `paper/` ignore rule also excluded
   a required numerical-evidence file nested inside `results/evidence`. It now
   applies only to the root manuscript directory. Finder metadata is excluded
   from manifests and archives. Tests check both cases, and the tracked file list
   must equal the complete release-manifest file list.
6. **Regenerated-value gate:** saved-table rendering now rejects a nonempty output
   whose bytes differ from the frozen numerical table. A successful process exit
   alone cannot mark such a table reproduced. Both matching and changed producers
   are covered by regression tests.
7. **Storage cleanup:** redundant illustration vectors and the bundled font folder
   were removed; the optional builder fetches the same pinned, checksum-verified
   font, with its license retained. Compact serialization of `configs/nvidia.json`
   preserves every parsed value and saves 13.1 MB; provenance updates retain that
   format. Fixed splits, frozen evidence and final demo assets are unchanged.
   Duplicate local extractions, retired data and synthetic check workspaces were
   removed after verification; the original conference ZIP is preserved outside
   the repository. `release_provenance/storage_cleanup.json` records the accounting.
8. **Versioned distribution:** CC BY 4.0 now covers original project material,
   matching the paper's arXiv license. Third-party exceptions remain explicit.
   Data, experiment and resource guides, Make entrypoints, versioned citation
   metadata and prepared Zenodo metadata make the artifact easier to retrieve
   and rerun. The root license and metadata are included in the export allowlist.
   The new CPU diagnostic gate rejects changed values, shapes and non-finite
   values before reporting success.
9. **Exact settings and candidate grids:** the fresh-baseline comparator now
   requires exact equality for KM-Prox `inverse_temperature` and the explicit
   clustering candidate grid, matching its treatment of other discrete settings.
   Regressions showed that the previous comparator incorrectly accepted small
   changes in these settings under the score tolerance.
   Validation scores retain the declared numerical tolerances. Scientific
   producers, frozen evidence and the earlier exact-value checkpoint are unchanged.

## What this establishes and what it does not

The inputs can be regenerated exactly from pinned public assets. All supplied
experimental numeric tables match the paper, and the tested execution components
work in fresh CPU/CUDA environments. The added integrity and estimator bugs have
regression coverage; the final local check cycle has no outstanding test failure.

This audit **has not yet completed all full-scale fitted routers, all 2,990 final
SharedTrunkNet network fits, every source-PCA job or every paired bootstrap
from newly generated predictions**. Full encoder extraction is complete. The
supplement omits those large arrays and
checkpoints by design. The full commands to regenerate them are in [RUN_COMMANDS.md](../guides/RUN_COMMANDS.md).
Consequently the audit establishes source/input availability, exact saved-result
consistency and component execution; it does not label the whole study independently
rerun or declare that no future scientific issue is possible.

The original manuscript figures were not recreated as document assets. This is an
experiment-code repository; no LaTeX/PDF build is required.
