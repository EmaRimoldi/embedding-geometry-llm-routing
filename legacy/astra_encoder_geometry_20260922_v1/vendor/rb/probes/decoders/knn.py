"""kNN router (PLAN.md, approved 2026-09-13).

Nothing is fit. For a test query, the k nearest training queries under
``metric`` (cosine on unit vectors in the approved configuration) are found and
the prediction for each model is the mean of their target values.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def predict(k: int, x_tr: np.ndarray, x_te: np.ndarray, reward_tr: np.ndarray, metric: str) -> np.ndarray:
    k = min(k, len(x_tr))
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute", n_jobs=8).fit(x_tr)
    _, idx = nn.kneighbors(x_te)
    neigh = reward_tr[idx]                                   # n_te x k x M
    with np.errstate(invalid="ignore"):
        out = np.nanmean(neigh, axis=1)
    col = np.nanmean(reward_tr, axis=0)
    return np.where(np.isnan(out), col, out)
