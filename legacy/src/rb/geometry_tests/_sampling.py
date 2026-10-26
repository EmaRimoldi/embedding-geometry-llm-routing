"""Row subsampling shared by the geometry tests."""
from __future__ import annotations

import numpy as np


def _sample(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n, k), replace=False))
