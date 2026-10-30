# Versioned release and archival

Version **1.0.0**, tag **v1.0.0**, packages the experiment code, fixed protocols,
reference evidence and documented checks associated with the paper. The pinned
numeric reference is Overleaf snapshot `6e3bea88ec5d65de3587c0719025702e110ac3e0`.
Full-scale fitting and all paired bootstraps have not all been independently rerun;
the release version identifies an artifact, not an independent reproduction badge.

The current checkout separates source, configurations, tools, jobs, tests, documentation and reproducibility materials. All current commands run from the repository root. The original v1.0.0 ZIP retains its published flat layout; the extraction commands below apply to that unchanged archive.

## Retrieve and verify

The [GitHub release](https://github.com/EmaRimoldi/embedding-geometry-llm-routing/releases/tag/v1.0.0)
contains `routing_reproduction.zip` and `routing_reproduction.zip.sha256`. Repository
access is required while the project remains private. GitHub's automatically
generated source archives also identify the tag; the attached ZIP uses the explicit
export allowlist and neutral ZIP metadata.

With GitHub CLI installed and authenticated when required:

```bash
gh release download v1.0.0 --repo EmaRimoldi/embedding-geometry-llm-routing \
  --pattern 'routing_reproduction.zip*' --dir release-v1.0.0
cd release-v1.0.0
python3 - <<'PY'
import hashlib
from pathlib import Path
archive = Path('routing_reproduction.zip')
expected = Path('routing_reproduction.zip.sha256').read_text().split()[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
print('Archive SHA-256 verified')
PY
unzip routing_reproduction.zip
cd routing-reproduction
python3 -B scripts/verify_release.py
python3 -B scripts/reproduce_results.py
python3 -B scripts/check_paper_results.py
```

The JSON release manifest fingerprints every exported file except itself. The
evidence manifest separately preserves original/exported evidence fingerprints.
Checksums establish identity; they do not establish scientific correctness.

## Build and identify a new version

For ordinary edits, run `make manifest` followed by `make verify`. This updates
current file fingerprints without modifying any ZIP. The steps below are for an
intentional new software release; retain copies of previously published archives
before invoking the packaging target.

1. Complete the relevant tests and numeric checks; update the documented audit
   scope with actual executions and limitations.
2. Record the new software version and release date in the release notes. Keep
   the README's paper citation separate from any future software DOI.
3. Run `make package`, commit its updated manifest, then run `make verify`.
4. Check that every tracked file is exported and that no exported file is absent
   from the Git index. Require the exact commit's CPU CI to pass.
5. Create an annotated tag on that commit and upload the generated ZIP and checksum
   as release assets. Download them and verify the published bytes.

Fixes after publication receive a new version; preserve the earlier tag, evidence
and release assets. Large regenerated arrays can be attached to a separately
versioned data deposit with their own checksums and input/code provenance.

## Optional Zenodo archival

The README contains the preferred paper citation; the release notes identify
software versions. A future Zenodo deposit can describe the software with
creators, affiliations, version, CC BY 4.0 license, third-party exceptions and the
paper DOI as a related publication. That metadata can be supplied during deposit.

**No software DOI has been issued for this repository.** For a future deposit,
upload the verified release archive with its metadata and intended access settings,
or configure Zenodo's GitHub integration. Record the returned version DOI and
concept DOI after deposit; keep these distinct from the paper DOI.

Official guidance: [GitHub archival with Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
and [Zenodo JSON metadata](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
