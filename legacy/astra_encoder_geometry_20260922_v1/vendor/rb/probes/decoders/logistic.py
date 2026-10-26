"""Logistic router: one success/failure classifier per model (target: success >= 0.5). λ = 0 only.

``cv=False``: fixed C = 1. ``cv=True``: C chosen by an inner 2-fold CV on a
<= 3000-row subsample of the train fold, then refit on the full train fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def predict(cv: bool, x_tr: np.ndarray, x_te: np.ndarray, success_tr: np.ndarray, seed: int) -> np.ndarray:
    m = success_tr.shape[1]
    n_te = len(x_te)
    out = np.zeros((n_te, m), dtype=np.float64)
    for j in range(m):
        rows = ~np.isnan(success_tr[:, j])
        y = success_tr[rows, j]
        if rows.sum() < 10 or y.min() == y.max():
            out[:, j] = y.mean() if rows.any() else 0.5
            continue
        if not cv:
            clf = LogisticRegression(C=1.0, max_iter=2000)
        else:
            rng = np.random.default_rng(seed + j)
            sub = np.flatnonzero(rows)
            if len(sub) > 3000:
                sub = rng.choice(sub, 3000, replace=False)
            ys = success_tr[sub, j]
            best_c, best_ll = 1e-2, np.inf
            if ys.min() != ys.max():
                half = rng.permutation(len(sub))
                parts = (half[: len(sub) // 2], half[len(sub) // 2:])
                for c in (1e-3, 1e-2, 1e-1, 1.0):
                    ll = 0.0
                    for a, b in (parts, parts[::-1]):
                        if ys[a].min() == ys[a].max():
                            continue
                        f = LogisticRegression(C=c, max_iter=500).fit(x_tr[sub[a]], ys[a])
                        p = np.clip(f.predict_proba(x_tr[sub[b]])[:, 1], 1e-6, 1 - 1e-6)
                        ll -= np.mean(ys[b] * np.log(p) + (1 - ys[b]) * np.log(1 - p))
                    if ll < best_ll:
                        best_ll, best_c = ll, c
            clf = LogisticRegression(C=best_c, max_iter=2000)
        clf.fit(x_tr[rows], y)
        out[:, j] = clf.predict_proba(x_te)[:, 1]
    return out
