"""Helpers shared by the decoders."""
from __future__ import annotations

import numpy as np


def _fill(reward_tr: np.ndarray) -> np.ndarray:
    col = np.nanmean(reward_tr, axis=0)
    col = np.where(np.isfinite(col), col, 0.5)
    return np.where(np.isnan(reward_tr), col, reward_tr)
