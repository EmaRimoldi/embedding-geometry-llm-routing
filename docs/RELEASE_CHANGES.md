# Release provenance and permitted changes

This is a curated copy of the completed research workspace. Original results and original sources were not changed.

Permitted release changes are limited to portable paths, scheduler examples, environment/bootstrap instructions, inclusion of previously implicit dependencies, complete first-run orchestration, anonymization of operational metadata, and compact result reproduction/verification. Scientific objectives, model architectures, grids, input cohorts, split indices, penalties, predictions and reported numerical outcomes are preserved.

The geometry batch includes task 0 when started from scratch; the research batch originally assumed a previously run pilot. The format-control runtime links are created relative to the new location. Required upstream NVIDIA source is packaged explicitly. Hash initialization creates release-specific provenance; original-run hashes remain separate.

`release_provenance/original_source_hashes.json` records original source/configuration fingerprints for traceability. `release_provenance/release_manifest.json` records exactly the exported bytes. `results/release_evidence_manifest.json` records copied evidence and any metadata redaction. Checksums establish file identity, not correctness of an experiment.

Only the public paths listed by the packaging script enter the ZIP. Private build records, caches, environments, generated verification outputs, full model/data archives and historical manuscript sources do not enter the ZIP.

Historical cleanup and one-off monitoring tools are omitted from the public export.
The public sources include method implementations and complete-run launchers; these
omissions do not remove recorded scientific outcomes.

## Reviewer smoke corrections

The release initializer now checks partial checkpoints and PCA caches before rebinding provenance and validates prerequisites before changing utility metadata. Scheduler accounting reports missing records explicitly. Launcher roots and shell wrapping are portable, and report instructions wait for dependencies. The neighbour helper caps k after self exclusion for small pools and rejects empty/non-distinct references; published cohorts do not encounter that boundary. Comparison tables now label the information required by diagnostic/oracle controls. These changes preserve all saved outcomes and frozen scientific settings.

## Method naming

The method is named **Family Adaptation**, with the machine-readable identifier family_adaptation. Display labels, configurations, saved evidence keys and reporting code use this convention consistently. The naming update preserves algorithms, numerical settings, splits and scientific results.
