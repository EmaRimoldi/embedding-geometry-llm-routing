# Final conference package integration

The final experiment jobs were submitted from `paper_repo`; a separate `paper_repo_anonymous_submission` project records completed CPU, GPU, public-data, PCA, MiniLM-padding and rename smoke jobs. Their locations were recovered from scheduler accounting. The final sources are not currently readable through the available cluster account, and no final package is included here.

When the submitted archive is recovered:

1. Keep the untouched archive under the ignored `incoming/` directory and record its SHA-256 using `python -m repro inspect-submission --archive incoming/submission.zip`.
2. Read its README, license, dependency files, source modules and input-preparation scripts. Identify the exact executable entry point for each experiment group in the inventory.
3. Verify the final 34,771-row RouterBench cohort, 43,887-row SPROUT cohort, MiniLM matched subsets, splits, seeds, row alignment, model revisions and feature conventions.
4. Import the experiment source with its notices and provenance. Keep benchmark text and weights external. Replace private paths with configurable input/output locations and preserve the experimental algorithms.
5. Check installation and a small end-to-end run, then run the experiment producers or clearly document the remaining compute requirements. Saved-decision replay is a complementary check, not a replacement for executable extraction/training/analysis code.

Only mark an experiment reproduced after its input identities, settings, commands and numerical outputs have actually been checked. Manuscript compilation is not required.

