"""Variance spectrum of a representation: effective rank, anisotropy, participation ratio (outcome-free)."""
from __future__ import annotations

import numpy as np

from ._sampling import _sample


def spectrum_stats(x: np.ndarray, max_rows: int = 20000, seed: int = 0) -> dict:
    idx = _sample(len(x), max_rows, seed)
    xs = x[idx].astype(np.float64)
    norms = np.linalg.norm(xs, axis=1)
    unit = xs / np.maximum(norms[:, None], 1e-12)
    # Anisotropy (Ethayarajh 2019): expected cosine between random pairs.
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(unit), 5000)
    b = rng.integers(0, len(unit), 5000)
    keep = a != b
    aniso = float(np.mean(np.sum(unit[a[keep]] * unit[b[keep]], axis=1)))
    centered = unit - unit.mean(0, keepdims=True)
    # Eigenvalues of covariance via SVD.
    s = np.linalg.svd(centered, compute_uv=False)
    ev = s ** 2 / max(len(centered) - 1, 1)
    p = ev / ev.sum()
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return {
        "dim": int(x.shape[1]),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "norm_cv": float(norms.std() / max(norms.mean(), 1e-12)),
        "anisotropy_mean_cosine": aniso,
        "effective_rank_exp_entropy": float(np.exp(entropy)),
        "participation_ratio": float(ev.sum() ** 2 / np.sum(ev ** 2)),
        "var_top1": float(p[0]), "var_top10": float(p[:10].sum()), "var_top50": float(p[:50].sum()),
        "n_components_95pct": int(np.searchsorted(np.cumsum(p), 0.95) + 1),
    }
