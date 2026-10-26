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
| Saved-result reconstruction | Seven result groups, including final per-model SharedTrunkNet JSONs | 23,740 fold-level records, 4,031 reconstructed aggregate rows, saved-aggregate tolerance 1e-10 relative / 1e-12 absolute |
| Printed results | Pinned manuscript values and settings, all 44 table labels | 5,493 checks passed at printed precision; supplied intervals compared without resampling |
| CPU regression tests | Pipeline/geometry, integrity, paper-gate, rank-tie and distribution regression cases | 18 tests passed |
| CPU baseline execution | Actual kNN, Family Adaptation and Ridge CLI paths on synthetic ID/OOD inputs | 156 score checks; 16 MLP-equivalence and 36 direct-utility Ridge cases |
| CPU/CUDA SharedTrunkNet | Actual probe, ternary search, fitting core, checkpoint reload and seed retention on synthetic inputs | 20 networks and 20 reloads on each device; cost-once and held-out-cost isolation checks passed |
| PCA preprocessing | Actual production PCA100 entry point, both full-SVD and randomized branches | 10 checks per branch, including held-out poisoning and stale-cache rejection |
| CUDA encoders | Actual MiniLM, GTE and Qwen L21/L23/L25 extraction on four illustrative prompts | All vectors finite, completed and hash-consistent |
| Model identity | Full cached weights compared with pinned public LFS checksums | 12 files, 45,815,257,928 bytes, passed |
| Execution plumbing | Inert-scheduler orchestration and isolated saved-report renderers | 17 orchestration categories; seven regenerated tables byte-identical to their supplied counterparts |

The current machine-readable records are under `release_provenance/current_checks/`.
They are distinct from `release_provenance/reviewer_smoke/`, which describes the
conference package's earlier checks.

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

## What this establishes and what it does not

The inputs can be regenerated exactly from pinned public assets. All supplied
experimental numeric tables match the paper, and the tested execution components
work in fresh CPU/CUDA environments. The added integrity and estimator bugs have
regression coverage; the final local check cycle has no outstanding test failure.

This audit **has not repeated all full-scale fitted routers, all 2,990 final
SharedTrunkNet network fits, every large embedding/PCA job or every paired bootstrap
from newly generated predictions**. The supplement omits those large arrays and
checkpoints by design. The full commands to regenerate them are in `RUN_COMMANDS.md`.
Consequently the audit establishes source/input availability, exact saved-result
consistency and component execution; it does not label the whole study independently
rerun or declare that no future scientific issue is possible.

The original manuscript figures were not recreated as document assets. This is an
experiment-code repository; no LaTeX/PDF build is required.
