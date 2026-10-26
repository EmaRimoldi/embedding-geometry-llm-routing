"""Decoder dispatcher: a decoder name -> the classifier module that implements it.

Names: ``ridge``, ``logistic``, ``logisticcv``, ``knn<k>``, ``kmeans<K>``,
``mlp`` (ours), ``mlp100`` / ``mlp100x100`` (RouterBench's).
"""
from __future__ import annotations

import numpy as np

from . import kmeans, knn, logistic, mlp, ridge


def decode(decoder: str, x_tr: np.ndarray, x_te: np.ndarray, reward_tr: np.ndarray,
           success_tr: np.ndarray, metric: str, seed: int = 0) -> np.ndarray:
    if decoder == "ridge":
        return ridge.predict(x_tr, x_te, reward_tr)
    if decoder in ("logistic", "logisticcv"):
        return logistic.predict(decoder == "logisticcv", x_tr, x_te, success_tr, seed)
    if decoder.startswith("knn"):
        return knn.predict(int(decoder[3:]), x_tr, x_te, reward_tr, metric)
    if decoder.startswith("kmeans"):
        return kmeans.predict(int(decoder[6:]), x_tr, x_te, reward_tr, metric, seed)
    if decoder.startswith("mlp"):
        return mlp.predict(decoder[3:], x_tr, x_te, reward_tr, seed)
    raise ValueError(decoder)
