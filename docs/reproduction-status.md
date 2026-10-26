# Experiment execution status

The executable sources exported so far are from the audited September archive. Their origin and hashes are preserved in `provenance/source-manifest.json`. They have not been replaced by inferred implementations of the missing October code.

| Component | Available code or verification | Remaining work |
| --- | --- | --- |
| Frozen numeric panels and splits | Local staging from the pinned historical archive; structural validation | Recover final cohort/preparation and query identities |
| Training-selected model and realized oracle | Portable runner uses the original scoring function | Final-cohort rerun after its inputs are recovered |
| kNN and Family Adaptation lineage | Original runner; end-to-end synthetic checks | Benchmark runs require aligned embeddings; verify final code changes |
| GTE/Qwen extraction | Historical extraction/alignment sources | Final raw-prompt conventions, checkpoint hashes, masks and environment |
| Avengers-Pro and KM-Prox | Historical source and job scripts | Import final configurations and validate benchmark execution |
| Ridge and MLP | Historical probes/MLP source; final job locations known | Final Ridge baseline and reduced MLP selection implementation |
| SharedTrunkNet and its three variants | Final job paths known from scheduler accounting | **Final implementation is not included in the recovered archive** |
| Geometry, subject clustering and MiniLM | Historical code and final scheduler leads | Final matched cohorts and full-query estimators |
| Lookup, transfer and ordering diagnostics | Historical implementations and final scheduler leads | Final source, settings and numeric outputs |
| Layer sweeps | Historical extraction/routing code and final job paths | Final source-only selection and benchmark reruns |
| Reward/cost, format controls and intervals | Historical diagnostic source and dedicated final job paths | Final producer code, per-query decisions and statistical protocol |

The repository includes source code, not a claim that every historical script is independently portable. Some archived studies consume outputs of other studies or anonymized external paths. The first portable launchers cover reference controls, kNN and the soft family prior. Other launchers should be completed against the actual conference implementation rather than guessed from document descriptions.

`provenance/numeric_comparison.csv` compares historical curves with the final reported values: 18 of 20 inspected means differ at four decimals. This is evidence that the old source/data campaign is not interchangeable with the final one; it is not a validation failure of the final experiments.

The CPU environment pins NumPy 1.26.4 and scikit-learn 1.5.2, matching those two final reported versions. It is sufficient for the implemented checks. It is not a full historical or GPU experiment lock. Actual execution environment versions are saved with validation outputs.

