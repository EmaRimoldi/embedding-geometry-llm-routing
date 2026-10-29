# Versioned release and archival

Version **1.0.0**, tag **v1.0.0**, packages the experiment code, fixed protocols,
reference evidence and documented checks associated with the paper. The pinned
numeric reference is Overleaf snapshot `6e3bea88ec5d65de3587c0719025702e110ac3e0`.
Full-scale fitting and all paired bootstraps have not all been independently rerun;
the release version identifies an artifact, not an independent reproduction badge.

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
2. Update software `version` and `date-released` in `CITATION.cff`, the matching
   metadata in `.zenodo.json`, and the release notes. Keep the paper DOI separate.
3. Run `make package`, commit its updated manifest, then run `make verify`.
4. Check that every tracked file is exported and that no exported file is absent
   from the Git index. Require the exact commit's CPU CI to pass.
5. Create an annotated tag on that commit and upload the generated ZIP and checksum
   as release assets. Download them and verify the published bytes.

Fixes after publication receive a new version; preserve the earlier tag, evidence
and release assets. Large regenerated arrays can be attached to a separately
versioned data deposit with their own checksums and input/code provenance.

## Zenodo metadata

The root `.zenodo.json` contains creators, affiliations, version, CC BY 4.0 license
and the paper DOI with the `isSupplementTo` relationship. `CITATION.cff` preserves
the preferred paper citation and identifies this software version.

**No software DOI has been issued by this packaging step.** The repository is
private and no authenticated Zenodo connection is configured in this workspace.
To archive via the GitHub integration, an owner must connect the intended Zenodo
account and enable this repository when it is public. An owner can also make a
manual deposit of the verified release ZIP under the intended access settings.
Record the returned version DOI and concept DOI after deposit; do not substitute
the paper DOI or invent a placeholder identifier.

Official guidance: [GitHub archival with Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
and [Zenodo JSON metadata](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
