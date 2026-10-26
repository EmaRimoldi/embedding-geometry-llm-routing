"""Reward-cost frontier metrics used by evaluate.py (nondominated points, convex hull, AIQ)."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def nondominated(points: pd.DataFrame, cost: str, quality: str) -> pd.DataFrame:
    ordered = points.sort_values([cost, quality], ascending=[True, False]).copy()
    kept: list[int] = []
    best = -np.inf
    for index, row in ordered.iterrows():
        value = float(row[quality])
        if value > best + 1e-12:
            kept.append(index)
            best = value
    return ordered.loc[kept].sort_values(cost)


def ndch(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": np.asarray(list(x), float), "y": np.asarray(list(y), float)})
    frame = frame.groupby("x", as_index=False).y.max().sort_values("x")
    frame = nondominated(frame, "x", "y")
    hull: list[tuple[float, float]] = []
    for row in frame.itertuples(index=False):
        point = (float(row.x), float(row.y))
        while len(hull) >= 2:
            s1 = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
            s2 = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
            if s2 <= s1 + 1e-12:
                break
            hull.pop()
        hull.append(point)
    return np.asarray([p[0] for p in hull]), np.asarray([p[1] for p in hull])


def aiq(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    lower = max(float(x.min()) for x, _ in curves.values())
    upper = min(float(x.max()) for x, _ in curves.values())
    if upper <= lower:
        return {name: float("nan") for name in curves}
    grid = np.linspace(lower, upper, 2001)
    return {
        name: float(np.trapezoid(np.interp(grid, x, y), grid) / (upper - lower))
        for name, (x, y) in curves.items()
    }
