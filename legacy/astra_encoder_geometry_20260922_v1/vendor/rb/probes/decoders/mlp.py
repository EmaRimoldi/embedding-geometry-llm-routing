"""MLP router (PLAN.md, approved 2026-09-13).

``spec == ""``: our fixed (256,) head with early stopping.
``spec == "100"`` / ``"100x100"``: RouterBench's published configurations
(hidden (100,) / (100,100), relu, Adam lr 1e-3, 200 iterations, seed 1234).
One multi-output regressor on the target matrix (see PLAN.md for the
per-model variant used in the fidelity rows).
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from ._common import _fill


def predict(spec: str, x_tr: np.ndarray, x_te: np.ndarray, reward_tr: np.ndarray, seed: int) -> np.ndarray:
    if spec == "":
        hidden, kwargs = (256,), {"early_stopping": True, "alpha": 1e-4}
    else:
        hidden = tuple(int(h) for h in spec.split("x"))
        kwargs = {"learning_rate_init": 1e-3, "max_iter": 200, "random_state": 1234}
    mlp = MLPRegressor(hidden_layer_sizes=hidden, max_iter=kwargs.pop("max_iter", 200),
                       random_state=kwargs.pop("random_state", seed), **kwargs)
    mlp.fit(x_tr, _fill(reward_tr))
    return np.clip(mlp.predict(x_te), 0, 1)
