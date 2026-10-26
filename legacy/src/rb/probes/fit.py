"""Out-of-fold fitting: one (representation, transform, decoder, split) unit -> predictions for every row."""
from __future__ import annotations

import time

import numpy as np

from ..data import Targets
from .decoders import decode
from .splits import make_folds
from .transforms import Transform


def fit_predict_oof(x: np.ndarray, targets: Targets, transform: str, decoder: str, split: str,
                    controls: np.ndarray | None = None, log_norm: np.ndarray | None = None,
                    seed: int = 0) -> dict:
    folds = make_folds(split, targets, seed)
    n, m = targets.reward.shape
    pred = np.full((n, m), np.nan, dtype=np.float64)
    best_single = np.full(n, -1, dtype=np.int64)     # per-row best-single-model chosen on its train fold
    domain_best = np.full(n, -1, dtype=np.int64)     # per-row best model of its dataset on train fold
    tf = Transform(transform, targets.dataset_index, int(targets.dataset_index.max()) + 1, controls, log_norm)
    t0 = time.time()
    for tr, te in folds:
        a, b = tf.fit_transform(x, tr, te)
        pred[te] = decode(decoder, a, b, targets.reward[tr], targets.success[tr], tf.metric, seed)
        train_mean = np.nanmean(targets.reward[tr], axis=0)
        best_single[te] = int(np.nanargmax(train_mean))
        for g in np.unique(targets.dataset_index[te]):
            rows_tr = tr[targets.dataset_index[tr] == g]
            rows_te = te[targets.dataset_index[te] == g]
            if len(rows_tr):
                domain_best[rows_te] = int(np.nanargmax(np.nanmean(targets.reward[rows_tr], axis=0)))
            else:
                domain_best[rows_te] = best_single[rows_te[0]]
    covered = ~np.isnan(pred).all(1)
    return {"pred": pred, "best_single": best_single, "domain_best": domain_best,
            "covered": covered, "n_folds": len(folds), "elapsed_s": time.time() - t0}
