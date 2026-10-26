"""How much of a representation is dataset identity: eta^2, linear identifiability, K-means/dataset alignment (outcome-free)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ._sampling import _sample


def dataset_variance_fraction(x: np.ndarray, groups: np.ndarray) -> dict:
    """eta^2: between-dataset sum of squares over total, on unit vectors and on PCs."""
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    unit = unit.astype(np.float64)
    mu = unit.mean(0)
    total = np.sum((unit - mu) ** 2)
    between = 0.0
    for g in np.unique(groups):
        m = groups == g
        between += m.sum() * np.sum((unit[m].mean(0) - mu) ** 2)
    # Per principal component eta^2 for the first 20 PCs.
    centered = unit - mu
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ vt[:20].T
    per_pc = []
    for j in range(pcs.shape[1]):
        col = pcs[:, j]
        tot = np.sum((col - col.mean()) ** 2)
        bet = sum((groups == g).sum() * (col[groups == g].mean() - col.mean()) ** 2 for g in np.unique(groups))
        per_pc.append(float(bet / max(tot, 1e-12)))
    return {"eta2_total": float(between / max(total, 1e-12)), "eta2_per_pc_top20": per_pc,
            "var_frac_top20": (s[:20] ** 2 / np.sum(s ** 2)).tolist()}


def dataset_identifiability(x: np.ndarray, groups: np.ndarray, seed: int = 0, max_rows: int = 15000) -> dict:
    idx = _sample(len(x), max_rows, seed)
    xs = StandardScaler().fit_transform(x[idx])
    clf = LogisticRegression(max_iter=300, C=0.5, n_jobs=8)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, xs, groups[idx], cv=cv, scoring="accuracy")
    majority = np.bincount(groups[idx]).max() / len(idx)
    return {"linear_probe_accuracy": float(acc.mean()), "linear_probe_accuracy_std": float(acc.std()),
            "majority_baseline": float(majority), "n_rows": int(len(idx))}


def cluster_alignment(x: np.ndarray, groups: np.ndarray, ks=(7, 16, 32, 64, 128, 256), seed: int = 0,
                      max_rows: int = 20000) -> dict:
    idx = _sample(len(x), max_rows, seed)
    unit = x[idx] / np.maximum(np.linalg.norm(x[idx], axis=1, keepdims=True), 1e-12)
    out = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=3, random_state=seed)
        labels = km.fit_predict(unit)
        # Purity: fraction of points whose cluster's majority dataset equals their dataset.
        purity = 0
        for c in range(k):
            m = labels == c
            if m.any():
                purity += np.bincount(groups[idx][m]).max()
        out[str(k)] = {"nmi_with_dataset": float(normalized_mutual_info_score(groups[idx], labels)),
                       "purity_wrt_dataset": float(purity / len(idx))}
    sil_idx = _sample(len(idx), 6000, seed + 1)
    out["silhouette_by_dataset"] = float(silhouette_score(unit[sil_idx], groups[idx][sil_idx], metric="cosine"))
    return out
