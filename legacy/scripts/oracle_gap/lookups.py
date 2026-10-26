"""Lookup routers on the released outcome table, fitted on training rows only.
Kept separate from common.py so the running exp80/81/83 jobs are not affected."""
from __future__ import annotations
import numpy as np

MIN_CELL = 50


def utility(panel, lam: float) -> np.ndarray:
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    c = np.nan_to_num(panel.costs.astype(np.float64), nan=0.0)
    return r - lam * c


def fit_lookup(U_tr, key_tr, fam_tr, F, n_keys, min_cell=MIN_CELL):
    """Best model by mean training utility per key, with fallback to the family's best model
    when the key has fewer than min_cell training rows. Returns (fam_best[F], key_best[n_keys] or -1)."""
    fam_best = np.array([int(U_tr[fam_tr == g].mean(0).argmax()) if (fam_tr == g).any() else 0 for g in range(F)])
    key_best = np.full(n_keys, -1, dtype=np.int64)
    for k in np.unique(key_tr):
        m = key_tr == k
        if m.sum() >= min_cell:
            key_best[k] = int(U_tr[m].mean(0).argmax())
    return fam_best, key_best


def apply_lookup(fam, key, fam_best, key_best) -> np.ndarray:
    choice = key_best[key]
    fallback = choice < 0
    choice = choice.copy(); choice[fallback] = fam_best[fam[fallback]]
    return choice


def combine_keys(*cols):
    """Integer key for the cartesian product of integer label columns."""
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        key = key * (int(c.max()) + 1) + c.astype(np.int64)
    _, key = np.unique(key, return_inverse=True)
    return key
