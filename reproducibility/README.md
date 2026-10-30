# Reproducibility materials

This directory retains the inputs, environments and evidence needed to inspect and reproduce the study. Algorithms are in [src/](../src/), protocols in [configs/](../configs/), automation in [scripts/](../scripts/) and [jobs/](../jobs/), and regression cases in [tests/](../tests/).

## Contents

| Location | Purpose |
| --- | --- |
| [environment/](environment/) | Pinned CPU, download and optional illustration environments; runtime provenance templates |
| [manifests/](manifests/) | Public asset revisions, weight checksums and third-party license records |
| [evidence/](evidence/) | Frozen original scores, numerical summaries and audit reports |
| [evidence_manifest.json](evidence_manifest.json) | Original evidence identities and exported SHA-256 values |
| [provenance/](provenance/) | Original source fingerprints, recorded transformations, current export manifest and scoped checks |
| [examples/proof_twins.jsonl](examples/proof_twins.jsonl) | Four illustrative prompts used by the encoder and proof-twin checks |

## Verify or run

Run commands from the repository root:

```bash
make verify
make test
make smoke
```

`make verify` uses the Python standard library. Tests and synthetic execution checks require the pinned compute environment in [requirements.lock](../requirements.lock) or the CPU environment described in the [execution guide](../docs/guides/RUN_COMMANDS.md#platform-and-setup).

The historical evidence manifest retains its original `results/evidence/` names and hashes. Current readers resolve those names under `reproducibility/evidence/`; the archived records are unchanged. The current [export manifest](provenance/release_manifest.json) lists repository-relative paths in this checkout.

The layout initializer copies the four illustrative prompts to the ignored runtime location `datasets/proof_twins_v1/prepared/queries.jsonl`. It rejects different existing bytes. It does not prepare either full benchmark or alter existing run provenance.

New downloads, embeddings, fits and reports go to ignored runtime directories described in the [retention guide](../docs/reference/STRUCTURE.md). See the [reproduction audit](../docs/validation/REPRODUCTION_AUDIT.md) for completed work, numerical differences and remaining coverage. Saved evidence checks and synthetic tests do not establish full independent reproduction.
