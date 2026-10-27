# Contributing and reporting reproduction issues

Use GitHub Issues for questions, failures and proposed corrections. Include the
release tag or commit, command, platform/Python/package versions, relevant input
hashes, expected result and observed result. Include a small log excerpt with
credentials and machine-specific details removed.

For code changes, run `make test` and the relevant component check in the pinned
CPU environment. `COMPUTE_PYTHON` can select another installed interpreter.
Explain any effect on row selection, source/test separation, seeds, penalties,
aggregation or uncertainty. Add a regression case for a demonstrated bug.

Keep saved evidence and original-source fingerprints historical. New runs belong
in ignored runtime directories; record their provenance and compare against the
reference outputs. A changed report must not silently become a new reference.

Run `make package` after edits, commit the regenerated release manifest, and run
`make verify`. Large data, embeddings, checkpoints, environments and temporary
outputs remain outside Git. The directory and retention policy is in
[docs/STRUCTURE.md](docs/STRUCTURE.md).

Original contributions use CC BY 4.0 as described in [LICENSE](LICENSE), with
third-party exceptions listed in [THIRD_PARTY.md](THIRD_PARTY.md). Retain upstream
attribution when adapting existing components.
