"""K-means cluster router (PLAN.md, approved 2026-09-13).

K-means on the (unit-normalised) training embeddings, targets not involved;
per cluster the mean target of its training queries; a test query gets the
means of its nearest centroid. Empty clusters fall back to the global mean.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def predict(k: int, x_tr: np.ndarray, x_te: np.ndarray, reward_tr: np.ndarray, metric: str, seed: int) -> np.ndarray:
    k = min(k, max(2, len(x_tr) // 5))
    u = x_tr if metric == "euclidean" else x_tr / np.maximum(np.linalg.norm(x_tr, axis=1, keepdims=True), 1e-12)
    ute = x_te if metric == "euclidean" else x_te / np.maximum(np.linalg.norm(x_te, axis=1, keepdims=True), 1e-12)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(u)
    lab_tr = km.labels_
    lab_te = km.predict(ute)
    grand = np.nanmean(reward_tr, axis=0)
    means = np.vstack([np.nanmean(reward_tr[lab_tr == c], axis=0) if (lab_tr == c).any() else grand
                       for c in range(k)])
    means = np.where(np.isfinite(means), means, grand)
    return means[lab_te]
