# Reproduction audit

The active code is now the final conference supplement, identified by archive
SHA-256 `043676877fed7b9ecf4f93da0ba428a5f8dbd0025212873fcebc94470623494e`.
The original archive passed its 1,981-file release-manifest check before any changes.
`release_provenance/conference_original_manifest.json` preserves those original
fingerprints; `release_provenance/release_manifest.json` describes this repository.

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
| CPU regression tests | Pipeline/geometry, integrity, paper-gate, rank-tie, distribution and fresh-diagnostic/baseline comparison cases | Passed, including 12 new baseline-audit cases |
| CPU baseline execution | Actual kNN, Family Adaptation, Ridge, Avengers-Pro and KM-Prox CLI paths on synthetic ID/OOD inputs | 260 score checks; 16 MLP-equivalence and 36 direct-utility Ridge cases |
| CPU/CUDA SharedTrunkNet | Actual probe, ternary search, fitting core, checkpoint reload and seed retention on synthetic inputs | 20 networks and 20 reloads on each device; cost-once and held-out-cost isolation checks passed |
| PCA preprocessing | Actual production PCA100 entry point, both full-SVD and randomized branches | 10 checks per branch, including held-out poisoning and stale-cache rejection |
| CUDA encoders | Actual MiniLM, GTE and Qwen L21/L23/L25 extraction on four illustrative prompts | All vectors finite, completed and hash-consistent |
| Complete fresh encoder extraction | GTE, MiniLM and Qwen layers 15–28 on both complete canonical cohorts | 32 arrays generated; all four GTE/L21 baseline input fingerprints exactly match the original |
| Fresh main baselines and controls, partial | Production kNN, Family Adaptation, Ridge, clustering, KM-Prox and IID lookup paths on their prescribed outer populations | 35 of 296 result records checked; all 39,260 numeric fields exactly match, including selections; remaining coverage is incomplete |
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

[`v1_full_encoders.json`](../release_provenance/current_checks/v1_full_encoders.json)
records all 32 manifests and the four comparisons. Extraction computes each hash
after flushing the complete array. Production baseline preparation also independently
rehashed all four input arrays and verified the canonical panels and frozen splits.
This is an encoder-stage checkpoint. New PCA,
fitted-router, geometry and uncertainty comparisons remain in progress.

The first fitted-baseline checkpoint is recorded in
[`v1_fresh_baselines.json`](../release_provenance/current_checks/v1_fresh_baselines.json).
It covers RouterBench IID evaluations and the first held-out-family Ridge fold
for both representations. All 39,260 compared numeric fields match exactly across
35 completed result records. The 819 policy scores include prompt routers,
clustering/metadata diagnostics and single-model/oracle controls; they are not
819 independent fitted models. Twenty-two source/config hash differences match
transitions already recorded in `release_provenance/source_changes.json`, and the
new source bytes were checked against that registry. Those hash differences remain
explicit in the report. No scientific or unexplained difference was found in this
subset; the complete 296-record main-baseline comparison has not passed yet.

`scripts/compare_baseline_results.py` makes this comparison reusable and refuses
to count missing experiments as matches. Twelve new adversarial tests cover
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

## What this establishes and what it does not

The inputs can be regenerated exactly from pinned public assets. All supplied
experimental numeric tables match the paper, and the tested execution components
work in fresh CPU/CUDA environments. The added integrity and estimator bugs have
regression coverage; the final local check cycle has no outstanding test failure.

This audit **has not yet completed all full-scale fitted routers, all 2,990 final
SharedTrunkNet network fits, every source-PCA job or every paired bootstrap
from newly generated predictions**. Full encoder extraction is complete. The
supplement omits those large arrays and
checkpoints by design. The full commands to regenerate them are in `RUN_COMMANDS.md`.
Consequently the audit establishes source/input availability, exact saved-result
consistency and component execution; it does not label the whole study independently
rerun or declare that no future scientific issue is possible.

The original manuscript figures were not recreated as document assets. This is an
experiment-code repository; no LaTeX/PDF build is required.
