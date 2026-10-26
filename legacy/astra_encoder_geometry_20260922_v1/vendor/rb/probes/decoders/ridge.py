"""Ridge router: one linear regression per model, shared penalty chosen by leave-one-out GCV on the train fold."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

from ..constants import RIDGE_ALPHAS
from ._common import _fill


def predict(x_tr: np.ndarray, x_te: np.ndarray, reward_tr: np.ndarray) -> np.ndarray:
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(x_tr, _fill(reward_tr))
    return np.clip(model.predict(x_te), 0, 1)
