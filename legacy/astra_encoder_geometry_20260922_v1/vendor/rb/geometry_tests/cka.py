"""Linear CKA between two representations (are two embeddings the same representation?)."""
from __future__ import annotations

import numpy as np

from ._sampling import _sample


def linear_cka(x: np.ndarray, y: np.ndarray, max_rows: int = 6000, seed: int = 0) -> float:
    idx = _sample(len(x), max_rows, seed)
    a = x[idx].astype(np.float64)
    b = y[idx].astype(np.float64)
    a = a - a.mean(0)
    b = b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b, "fro") ** 2
    return float(hsic / (np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro") + 1e-12))
