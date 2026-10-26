"""Shared pieces of the λ-sweep experiments (exp08 kNN / family lookup, exp09 MLP): splits, per-λ
summaries, RouterBench AIQ. Copied verbatim from scripts/exp08_basic_model_tests.py so that later
experiments use exactly the same definitions."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1e3, 1e4]
SEEDS = (0, 1, 2, 3, 4)
TEST_FRAC = 0.3


def split(n: int, fam: np.ndarray, seed: int):
    return train_test_split(np.arange(n), test_size=TEST_FRAC, random_state=seed, stratify=fam)


def aiq(points: np.ndarray, max_cost: float) -> float:
    """RouterBench AIQ: area under the non-decreasing convex hull of (cost, reward) points, / max_cost."""
    pts = points[np.argsort(points[:, 0])]
    hull = []
    for c, r in pts:  # upper envelope, non-decreasing in reward
        if hull and r < hull[-1][1]:
            continue
        hull.append([c, r])
    hull = np.array(hull)
    if hull[-1, 0] < max_cost:
        hull = np.vstack([hull, [max_cost, hull[-1, 1]]])
    return float(np.trapezoid(hull[:, 1], hull[:, 0]) / max_cost)


def summarise(choice: np.ndarray, r: np.ndarray, c: np.ndarray, lam: float, fam: np.ndarray, dis: np.ndarray, n_fam: int) -> dict:
    rr = r[np.arange(len(choice)), choice]; cc = c[np.arange(len(choice)), choice]; uu = rr - lam * cc
    def agg(mask):
        per_q = {"utility": float(uu[mask].mean()), "reward": float(rr[mask].mean()), "cost": float(cc[mask].mean())}
        per_f = {k: float(np.mean([v[mask & (fam == g)].mean() for g in range(n_fam) if (mask & (fam == g)).sum() > 0]))
                 for k, v in (("utility", uu), ("reward", rr), ("cost", cc))}
        return {"per_query": per_q, "per_family_equal_weight": per_f, "n": int(mask.sum())}
    return {"all": agg(np.ones_like(dis)), "disagreement": agg(dis),
            "per_family": {int(g): {"utility": float(uu[fam == g].mean()), "reward": float(rr[fam == g].mean()), "cost": float(cc[fam == g].mean())} for g in range(n_fam)},
            "_per_query_utility": uu}
