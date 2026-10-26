"""Train/test split schemes. Test sets partition the panel; no row is in train and test of the same fold."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from ..data import Targets


def make_folds(split: str, targets: Targets, seed: int = 0, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs whose test sets partition the panel."""
    n = len(targets.difficulty)
    groups = targets.dataset_index
    if split == "random":
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in cv.split(np.zeros(n), groups)]
    if split == "lodo":
        return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]
    if split == "within":
        folds = []
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            if len(rows) < 2 * n_folds:
                continue
            cv = KFold(n_folds, shuffle=True, random_state=seed)
            folds.extend((rows[tr], rows[te]) for tr, te in cv.split(rows))
        return folds
    raise ValueError(split)
