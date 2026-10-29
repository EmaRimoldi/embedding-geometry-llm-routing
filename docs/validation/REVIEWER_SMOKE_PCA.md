# Reviewer smoke: actual PCA100 preprocessing

## Reproduction command

Run inside a CPU allocation using the compute environment:

```bash
source jobs/env.sh
.venv/bin/python -B scripts/smoke_pca.py
```

Default dimensions are 200 synthetic rows, 512 input features and 160 prescribed
source rows. The frozen production choice remains **100 PCA components**.
For the smaller full-SVD branch:

```bash
.venv/bin/python -B scripts/smoke_pca.py --rows 160 --source-rows 120 --features 128
```

The smoke copies the actual `src/nvidia/pca.py` entrypoint and its real source
dependencies into a small temporary repository. It invokes the unchanged CLI in
a subprocess, with synthetic panel/vector files and verified fixture manifests.
It does not monkeypatch the scaler, PCA, source-row indexing, serialization or
cache logic. No model download, label generation, GPU allocation or production
configuration change is needed. All generated files stay under `.cache/`.

## Measured checks

Both branches completed successfully in the freshly installed compute environment
(NumPy 1.26.4, scikit-learn 1.5.2), using CPU-only allocations with four threads and
8 GB requested host memory.

| Fixture | Input shape | Source rows | PCA components | Auto solver | Check categories |
|---|---:|---:|---:|---|---:|
| Small | 160 × 128 | 120 | 100 | Full SVD | 10 passed |
| Wide | 200 × 512 | 160 | 100 | Randomized | 10 passed |

For each fixture, all 40 held-out vectors were changed to
`1000 * vector + 10000` while the prescribed source rows stayed identical.

- Every fitted scaler/PCA array remained **exactly identical**.
- Source-row projections remained exactly identical: maximum difference **0**.
- Held-out projections changed, as expected.
- Reloaded transforms reconstructed the saved projections; both recorded
  maximum reconstruction errors were **0**.
- Output shape, finite float32 values, stored source-row identities, config and
  raw-vector hashes, and projected-vector checksum were verified.
- Re-running the completed cache preserved its saved bytes.
- A changed config identity and an inconsistent raw-vector checksum each caused
  the expected rejection without silently overwriting a cached transform.

Reports and captured subprocess logs are preserved in
`.cache/reviewer_pca_smoke/run-*/`; the latest report is
`.cache/reviewer_pca_smoke/latest_report.json`. The two expected rejection
subprocesses are successful negative checks, not failed experiment fits.

## Limits

This checks actual preprocessing, saved-transform reloads and source isolation
on small synthetic inputs, including both auto-solver branches. It is not a
rerun of the paper's full embeddings, PCA cache, routing fits or numerical results.
Cache-byte integrity is checked independently by the smoke; the production
completed-cache fast path itself checks its identity, with additional integrity
checks performed by downstream preflight/audit code.
