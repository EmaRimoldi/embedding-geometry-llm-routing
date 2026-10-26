"""Pathological-pair diagnostics (exp74/exp75).

A pathological pair is two queries that are near-identical in a base embedding but have
different outcome profiles (different utility-argmax model, or solver sets with low
Jaccard).  These functions extract such pairs, measure how much an augmented
representation separates them relative to a scale normaliser, and score how well the
argmax label clusters in each space.  Pure numpy/scikit-learn, no I/O, no panel access.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def argmax_utility(rewards: np.ndarray, costs: np.ndarray, lam: float,
                   available: np.ndarray | None = None) -> np.ndarray:
    """Model maximising r - lam*c per row; ties resolved by lower cost, then lower index.

    Args:
        rewards: N x M reward matrix (NaNs already replaced by the caller).
        costs: N x M cost matrix (infs already replaced by the caller for the arithmetic).
        lam: cost penalty.
        available: optional N x M bool mask.  Models that were not run on a row get
            utility -inf and can never be chosen; without it every model is eligible,
            which silently lets a row pick a model the panel never evaluated.
    """
    u = rewards - lam * costs
    if available is not None:
        u = np.where(np.asarray(available, dtype=bool), u, -np.inf)
    best = u.max(axis=1, keepdims=True)
    tie = np.isclose(u, best, rtol=0.0, atol=1e-12)
    c = np.where(tie, costs, np.inf)
    return c.argmin(axis=1)


def solver_sets(rewards: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(rewards) >= threshold


def jaccard_pairs(S: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    a = S[pairs[:, 0]]; b = S[pairs[:, 1]]
    inter = (a & b).sum(axis=1); union = (a | b).sum(axis=1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0).astype(np.float32)


def candidate_pairs(x_unit: np.ndarray, fam: np.ndarray, k: int = 10, cos_min: float = 0.90,
                    batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Unordered pairs (i<j) where one is among the other's k nearest same-family rows with
    cosine >= cos_min.  Returns (pairs P×2 int64, cosine P float32), unique, sorted."""
    rows, cols, sims = [], [], []
    for f in np.unique(fam):
        idx = np.flatnonzero(fam == f)
        if len(idx) < 2:
            continue
        xf = x_unit[idx]
        kk = min(k + 1, len(idx))
        for st in range(0, len(idx), batch):
            s = xf[st:st + batch] @ xf.T
            part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
            sv = np.take_along_axis(s, part, axis=1)
            r = np.repeat(np.arange(st, st + len(part)), kk)
            c = part.ravel(); v = sv.ravel()
            keep = (c != r) & (v >= cos_min)
            rows.append(idx[r[keep]]); cols.append(idx[c[keep]]); sims.append(v[keep])
    if not rows:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    i = np.concatenate(rows); j = np.concatenate(cols); v = np.concatenate(sims).astype(np.float32)
    lo = np.minimum(i, j); hi = np.maximum(i, j)
    pairs = np.stack([lo, hi], axis=1).astype(np.int64)
    pairs, first = np.unique(pairs, axis=0, return_index=True)
    return pairs, v[first]


def classify_pairs(pairs: np.ndarray, argmax_by_lambda: dict[float, np.ndarray], jac: np.ndarray,
                   jac_path: float = 0.5, jac_benign: float = 0.8, lam_head: float = 100.0) -> dict[str, np.ndarray]:
    am = argmax_by_lambda[lam_head]
    diff = am[pairs[:, 0]] != am[pairs[:, 1]]
    p_arg = diff
    p_jac = jac <= jac_path
    benign = (~diff) & (jac >= jac_benign)
    return {"pathological_argmax": p_arg, "pathological_jaccard": p_jac,
            "pathological": p_arg | p_jac, "benign": benign}


def random_pairs(fam: np.ndarray, n_pairs: int, rng: np.random.Generator, same_family: bool = True) -> np.ndarray:
    n = len(fam)
    if not same_family:
        a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    else:
        # Only draw a from rows whose family has at least 2 members
        eligible = np.flatnonzero(np.isin(fam, [f for f in np.unique(fam) if (fam == f).sum() >= 2]))
        if len(eligible) == 0:
            raise ValueError("random_pairs: no family with two or more rows")
        a = eligible[rng.integers(0, len(eligible), n_pairs)]
        b = np.empty_like(a)
        for f in np.unique(fam):
            idx = np.flatnonzero(fam == f); sel = fam[a] == f
            b[sel] = idx[rng.integers(0, len(idx), sel.sum())]
    same = a == b
    while same.any():                      # resample collisions
        b[same] = rng.integers(0, n, same.sum()) if not same_family else b[same]
        if same_family:
            for f in np.unique(fam):
                idx = np.flatnonzero(fam == f); sel = same & (fam[a] == f)
                b[sel] = idx[rng.integers(0, len(idx), sel.sum())]
        same = a == b
    return np.stack([a, b], axis=1)


def pair_cosine_distance(x_unit: np.ndarray, pairs: np.ndarray, batch: int | None = None) -> np.ndarray:
    """Cosine distance of each (i, j) pair, computed in dimension-aware batches.

    ``x_unit[p[:, 0]]`` and ``x_unit[p[:, 1]]`` are two ``batch x D`` fancy-index copies, so a
    fixed ``batch`` costs memory proportional to D: at D = 40,264 a batch of 100,000 rows is
    two 15 GiB temporaries.  With ``batch=None`` the batch is chosen so each temporary holds
    about 64M float32 elements (256 MiB): 1,589 pairs at 40k dims, 100,000+ at small D.
    """
    if batch is None:
        batch = max(1, 64_000_000 // int(x_unit.shape[1]))
    out = np.empty(len(pairs), dtype=np.float32)
    for st in range(0, len(pairs), batch):
        p = pairs[st:st + batch]
        out[st:st + len(p)] = 1.0 - np.einsum("ij,ij->i", x_unit[p[:, 0]], x_unit[p[:, 1]])
    return np.clip(out, 0.0, 2.0)


def multiplier(d_base: np.ndarray, d_aug: np.ndarray, med_base: float, med_aug: float, floor: float = 1e-4) -> np.ndarray:
    nb = np.maximum(d_base / med_base, floor); na = np.maximum(d_aug / med_aug, floor)
    return na / nb


def knn_label_purity(x_unit: np.ndarray, labels: np.ndarray, fam: np.ndarray, k: int = 10, batch: int = 1024) -> float:
    tot = 0.0; cnt = 0
    for f in np.unique(fam):
        idx = np.flatnonzero(fam == f)
        if len(idx) <= k:
            continue
        xf = x_unit[idx]; lf = labels[idx]
        for st in range(0, len(idx), batch):
            s = xf[st:st + batch] @ xf.T
            s[np.arange(len(s)), np.arange(st, st + len(s))] = -np.inf
            nb = np.argpartition(-s, k - 1, axis=1)[:, :k]
            tot += (lf[nb] == lf[st:st + len(s), None]).mean(axis=1).sum(); cnt += len(s)
    return float(tot / max(cnt, 1))


def clustering_scores(x_unit: np.ndarray, labels: np.ndarray, rng_seed: int = 0, max_rows: int = 5000) -> dict:
    labels = np.asarray(labels); ks = int(len(np.unique(labels)))
    if ks < 2 or len(labels) < 2 * ks:
        return {"ari": float("nan"), "nmi": float("nan"), "silhouette": float("nan"), "k": ks}
    rng = np.random.default_rng(rng_seed)
    sel = rng.choice(len(labels), size=min(max_rows, len(labels)), replace=False)
    xs, ls = x_unit[sel], labels[sel]
    km = KMeans(n_clusters=ks, n_init=4, random_state=rng_seed).fit(xs)
    sil = float(silhouette_score(xs, ls, metric="cosine")) if len(np.unique(ls)) > 1 else float("nan")
    return {"ari": float(adjusted_rand_score(ls, km.labels_)),
            "nmi": float(normalized_mutual_info_score(ls, km.labels_)), "silhouette": sil, "k": ks}


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a - b); returns (estimate, 2.5%, 97.5%)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed); n = len(d)
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of ``values`` under non-negative ``weights`` (lower-value tie break).

    Returns the smallest value whose cumulative weight reaches half the total weight, or
    NaN when the total weight is zero (a bootstrap draw in which no pair survives).
    """
    v = np.asarray(values, dtype=np.float64); w = np.asarray(weights, dtype=np.float64)
    tot = float(w.sum())
    if len(v) == 0 or not np.isfinite(tot) or tot <= 0.0:
        return float("nan")
    o = np.argsort(v, kind="stable"); v = v[o]; w = w[o]
    return float(v[int(np.searchsorted(np.cumsum(w), 0.5 * tot, side="left"))])


def bootstrap_ratio_of_medians_by_query(m_path: np.ndarray, pairs_path: np.ndarray,
                                        m_benign: np.ndarray, pairs_benign: np.ndarray,
                                        n_rows: int, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap of median(m_path) / median(m_benign) over QUERIES, not pairs.

    Pairs share endpoints, so resampling pairs independently understates the interval.  Each
    draw instead resamples ``n_rows`` query rows with replacement, giving row multiplicities
    ``w = bincount(rng.integers(0, n_rows, n_rows))``; a pair (i, j) then enters the weighted
    median with weight ``w[i] * w[j]`` (its multiplicity in the resampled pair set).  The
    point estimate is the plain unweighted ratio of medians, as in the pair-level version.

    Args:
        m_path/m_benign: multiplier values of the pathological / benign pair groups.
        pairs_path/pairs_benign: the matching P x 2 row-index arrays (indices < n_rows).
        n_rows: number of distinct query rows the pair indices refer to.

    Returns:
        (estimate, 2.5th percentile, 97.5th percentile); NaNs if either group is empty.
    """
    mp = np.asarray(m_path, dtype=np.float64); mb = np.asarray(m_benign, dtype=np.float64)
    if len(mp) == 0 or len(mb) == 0:
        return float("nan"), float("nan"), float("nan")
    est = float(np.median(mp) / np.median(mb))
    pp = np.asarray(pairs_path, dtype=np.int64); pb = np.asarray(pairs_benign, dtype=np.int64)
    rng = np.random.default_rng(seed); draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows).astype(np.float64)
        num = weighted_median(mp, w[pp[:, 0]] * w[pp[:, 1]])
        den = weighted_median(mb, w[pb[:, 0]] * w[pb[:, 1]])
        draws[b] = (num / den) if (np.isfinite(den) and den != 0.0) else np.nan
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_ratio_of_medians(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Unpaired bootstrap of median(x)/median(y); returns (estimate, 2.5%, 97.5%)."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.array([np.median(x[rng.integers(0, len(x), len(x))]) / np.median(y[rng.integers(0, len(y), len(y))])
                      for _ in range(n_boot)])
    return float(np.median(x) / np.median(y)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------------------
# Dimension-free pair measures (exp74b).  Cosine, 1 - cos, the angle and the Euclidean
# distance between unit vectors are monotone in one another, so any rank or percentile
# computed from one of them is identical for all of them; only the multiplier depends on
# the choice.  ``unit_euclidean`` is the true metric used for the multiplier.
# ---------------------------------------------------------------------------------------

def unit_euclidean(cos: np.ndarray) -> np.ndarray:
    """Euclidean distance between unit vectors with cosine ``cos``: sqrt(2 - 2 cos)."""
    return np.sqrt(np.clip(2.0 - 2.0 * np.asarray(cos, dtype=np.float64), 0.0, 4.0))


def pair_cosine(x_unit: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine of each pair (i, j) of unit rows; memory-bounded like ``pair_cosine_distance``."""
    return 1.0 - pair_cosine_distance(x_unit, pairs).astype(np.float64)


def pair_unit_euclidean(x_unit: np.ndarray, pairs: np.ndarray, batch: int | None = None) -> np.ndarray:
    """||x_i - x_j|| for unit rows, computed from the difference (not from sqrt(2 - 2 cos),
    which loses all precision in float32 when cos is close to 1)."""
    if batch is None:
        batch = max(1, 64_000_000 // int(x_unit.shape[1]))
    out = np.empty(len(pairs), dtype=np.float64)
    for st in range(0, len(pairs), batch):
        p = pairs[st:st + batch]
        out[st:st + len(p)] = np.linalg.norm(x_unit[p[:, 0]] - x_unit[p[:, 1]], axis=1)
    return out


def family_percentile(x_unit: np.ndarray, fam: np.ndarray, pairs: np.ndarray, pair_cos: np.ndarray,
                      rng: np.random.Generator, n_ref: int = 20_000) -> np.ndarray:
    """Percentile of each pair's distance among ``n_ref`` random same-family pairs of the same
    family, in the same space: 0 = closer than every reference pair, 1 = farther than all.
    Computed on cosines (monotone in every distance above), so it is dimension-free."""
    out = np.full(len(pairs), np.nan)
    pf = fam[pairs[:, 0]]
    for f in np.unique(pf):
        idx = np.flatnonzero(fam == f)
        if len(idx) < 2:
            continue
        a = idx[rng.integers(0, len(idx), n_ref)]; b = idx[rng.integers(0, len(idx), n_ref)]
        keep = a != b
        ref = np.sort(1.0 - pair_cosine_distance(x_unit, np.stack([a[keep], b[keep]], 1)).astype(np.float64))
        sel = pf == f
        # fraction of reference pairs MORE similar than the pair = distance percentile
        out[sel] = 1.0 - np.searchsorted(ref, pair_cos[sel], side="right") / len(ref)
    return out


def neighbor_rank(x_unit: np.ndarray, fam: np.ndarray, pairs: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pair (i, j): rank of j among i's same-family rows (1 = nearest), rank of i
    among j's, and the family size.  Ties count against the pair (strictly-more-similar rows
    are counted).  Cost: one similarity row per distinct endpoint."""
    r_ij = np.zeros(len(pairs), dtype=np.int64); r_ji = np.zeros(len(pairs), dtype=np.int64)
    n_f = np.zeros(len(pairs), dtype=np.int64)
    s_pair = 1.0 - pair_cosine_distance(x_unit, pairs).astype(np.float32)
    for f in np.unique(fam[pairs[:, 0]]):
        idx = np.flatnonzero(fam == f); xf = x_unit[idx]
        local = np.full(len(fam), -1, dtype=np.int64); local[idx] = np.arange(len(idx))
        sel = np.flatnonzero(fam[pairs[:, 0]] == f); n_f[sel] = len(idx)
        for col, other, dest in ((0, 1, r_ij), (1, 0, r_ji)):
            q = pairs[sel, col]; uq, inv = np.unique(q, return_inverse=True)
            for st in range(0, len(uq), batch):
                ub = uq[st:st + batch]
                S = x_unit[ub] @ xf.T
                S[np.arange(len(ub)), local[ub]] = -np.inf          # exclude the query itself
                S.sort(axis=1)
                m = (inv >= st) & (inv < st + len(ub))
                rows = inv[m] - st
                thr = s_pair[sel[m]]
                cnt = np.array([S.shape[1] - np.searchsorted(S[r], t, side="right") for r, t in zip(rows, thr)])
                dest[sel[m]] = cnt + 1
    return r_ij, r_ji, n_f


def bootstrap_median_diff_by_query(v_path: np.ndarray, pairs_path: np.ndarray, v_ben: np.ndarray,
                                   pairs_ben: np.ndarray, n_rows: int, n_boot: int = 1000,
                                   seed: int = 0) -> tuple[float, float, float]:
    """Query-level cluster bootstrap of median(v_path) - median(v_ben) (same weighting as
    ``bootstrap_ratio_of_medians_by_query``); NaNs if a group is empty."""
    vp = np.asarray(v_path, dtype=np.float64); vb = np.asarray(v_ben, dtype=np.float64)
    if len(vp) == 0 or len(vb) == 0:
        return float("nan"), float("nan"), float("nan")
    est = float(np.median(vp) - np.median(vb))
    pp = np.asarray(pairs_path, dtype=np.int64); pb = np.asarray(pairs_ben, dtype=np.int64)
    rng = np.random.default_rng(seed); draws = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows).astype(np.float64)
        draws[b] = weighted_median(vp, w[pp[:, 0]] * w[pp[:, 1]]) - weighted_median(vb, w[pb[:, 0]] * w[pb[:, 1]])
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
