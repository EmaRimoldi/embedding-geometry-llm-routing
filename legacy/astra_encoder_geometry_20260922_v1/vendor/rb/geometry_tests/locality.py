"""Does outcome similarity decay with embedding distance? Pair curves and k-NN neighbour agreement vs a same-dataset random baseline."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from ..data import Targets


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def outcome_agreement(reward: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - mean |r_a - r_b| over co-available models, per pair."""
    ra, rb = reward[a], reward[b]
    both = ~np.isnan(ra) & ~np.isnan(rb)
    diff = np.where(both, np.abs(ra - rb), 0.0)
    return 1.0 - diff.sum(1) / np.maximum(both.sum(1), 1)


def success_jaccard(success: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sa = np.nan_to_num(success[a]) > 0.5
    sb = np.nan_to_num(success[b]) > 0.5
    inter = (sa & sb).sum(1)
    union = (sa | sb).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def pair_analysis(x: np.ndarray, targets: Targets, n_pairs: int = 200000, seed: int = 0, n_bins: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    u = _unit(x.astype(np.float32))
    n = len(u)
    groups = targets.dataset_index
    out = {}
    for scope in ("within", "across"):
        if scope == "within":
            a = rng.integers(0, n, n_pairs * 3)
            # Pair each a with a random row of the same dataset.
            order = np.argsort(groups)
            sorted_groups = groups[order]
            starts = np.searchsorted(sorted_groups, np.arange(groups.max() + 1), "left")
            ends = np.searchsorted(sorted_groups, np.arange(groups.max() + 1), "right")
            b = np.array([order[rng.integers(starts[g], ends[g])] for g in groups[a]])
            keep = a != b
            a, b = a[keep][:n_pairs], b[keep][:n_pairs]
        else:
            a = rng.integers(0, n, n_pairs * 2)
            b = rng.integers(0, n, n_pairs * 2)
            keep = groups[a] != groups[b]
            a, b = a[keep][:n_pairs], b[keep][:n_pairs]
        dist = 1.0 - np.sum(u[a] * u[b], axis=1)
        agree = outcome_agreement(targets.reward, a, b)
        jac = success_jaccard(targets.success, a, b)
        ddiff = np.abs(targets.difficulty[a] - targets.difficulty[b])
        bins = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
        which = np.clip(np.searchsorted(bins, dist, "right") - 1, 0, n_bins - 1)
        curve = [{"dist_lo": float(bins[i]), "dist_hi": float(bins[i + 1]),
                  "agreement": float(agree[which == i].mean()), "success_jaccard": float(jac[which == i].mean()),
                  "abs_difficulty_diff": float(ddiff[which == i].mean()), "n": int((which == i).sum())}
                 for i in range(n_bins)]
        out[scope] = {
            "n_pairs": int(len(a)),
            "spearman_dist_agreement": float(spearmanr(dist, agree).correlation),
            "spearman_dist_jaccard": float(spearmanr(dist, jac).correlation),
            "spearman_dist_absdiffdiff": float(spearmanr(dist, ddiff).correlation),
            "mean_agreement": float(agree.mean()), "mean_jaccard": float(jac.mean()),
            "curve": curve,
        }
    return out


def neighbour_analysis(x: np.ndarray, targets: Targets, ks=(1, 5, 20, 100), seed: int = 0,
                       max_rows: int = 15000) -> dict:
    rng = np.random.default_rng(seed)
    u = _unit(x.astype(np.float32))
    n = len(u)
    q = np.sort(rng.choice(n, size=min(n, max_rows), replace=False))
    kmax = max(ks)
    nn = NearestNeighbors(n_neighbors=kmax + 1, metric="cosine", algorithm="brute", n_jobs=-1).fit(u)
    dist, idx = nn.kneighbors(u[q])
    idx, dist = idx[:, 1:], dist[:, 1:]            # drop self
    groups = targets.dataset_index
    same_ds = groups[idx] == groups[q][:, None]
    out = {"n_queries": int(len(q)), "same_dataset_rate_by_k": {}}
    # Random same-dataset baseline neighbours for each query.
    order = np.argsort(groups)
    sg = groups[order]
    starts = np.searchsorted(sg, np.arange(groups.max() + 1), "left")
    ends = np.searchsorted(sg, np.arange(groups.max() + 1), "right")
    for k in ks:
        nb = idx[:, :k]
        a = np.repeat(q, k)
        b = nb.reshape(-1)
        agree = outcome_agreement(targets.reward, a, b).reshape(len(q), k).mean(1)
        jac = success_jaccard(targets.success, a, b).reshape(len(q), k).mean(1)
        ddiff = np.abs(targets.difficulty[a] - targets.difficulty[b]).reshape(len(q), k).mean(1)
        rb = np.array([order[rng.integers(starts[g], ends[g])] for g in np.repeat(groups[q], k)])
        agree_r = outcome_agreement(targets.reward, a, rb).reshape(len(q), k).mean(1)
        jac_r = success_jaccard(targets.success, a, rb).reshape(len(q), k).mean(1)
        ddiff_r = np.abs(targets.difficulty[a] - targets.difficulty[rb]).reshape(len(q), k).mean(1)
        # Neighbour-mean difficulty as a predictor of own difficulty.
        pred_d = targets.difficulty[b].reshape(len(q), k).mean(1)
        pred_d_r = targets.difficulty[rb].reshape(len(q), k).mean(1)
        out["same_dataset_rate_by_k"][str(k)] = float(same_ds[:, :k].mean())
        out[str(k)] = {
            "agreement_knn": float(agree.mean()), "agreement_random_same_dataset": float(agree_r.mean()),
            "jaccard_knn": float(jac.mean()), "jaccard_random_same_dataset": float(jac_r.mean()),
            "abs_difficulty_diff_knn": float(ddiff.mean()), "abs_difficulty_diff_random_same_dataset": float(ddiff_r.mean()),
            "difficulty_spearman_knn": float(spearmanr(targets.difficulty[q], pred_d).correlation),
            "difficulty_spearman_random_same_dataset": float(spearmanr(targets.difficulty[q], pred_d_r).correlation),
            "agreement_knn_disagreement_subset": float(agree[targets.disagreement[q]].mean()),
            "agreement_random_disagreement_subset": float(agree_r[targets.disagreement[q]].mean()),
        }
    # Per-dataset neighbour agreement gain (kNN minus random) at k=20.
    k = 20 if 20 in ks else ks[-1]
    nb = idx[:, :k]
    a = np.repeat(q, k)
    agree = outcome_agreement(targets.reward, a, nb.reshape(-1)).reshape(len(q), k).mean(1)
    rb = np.array([order[rng.integers(starts[g], ends[g])] for g in np.repeat(groups[q], k)])
    agree_r = outcome_agreement(targets.reward, a, rb).reshape(len(q), k).mean(1)
    out["per_dataset_gain_k20"] = {int(g): float(agree[groups[q] == g].mean() - agree_r[groups[q] == g].mean())
                                   for g in np.unique(groups[q])}
    return out
