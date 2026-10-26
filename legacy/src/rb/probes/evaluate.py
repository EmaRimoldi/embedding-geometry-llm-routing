"""Metrics from out-of-fold predictions: routing reward vs references, AUROC, difficulty, cost frontier."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ..data import Targets
from .constants import LAMBDA_GRID


def _choice(score: np.ndarray, avail: np.ndarray) -> np.ndarray:
    s = np.where(avail, score, -np.inf)
    return np.argmax(s, axis=1)


def _realized(choice: np.ndarray, reward: np.ndarray) -> np.ndarray:
    r = reward[np.arange(len(choice)), choice]
    return np.where(np.isfinite(r), r, 0.0)


def frontier(pred: np.ndarray, reward: np.ndarray, cost: np.ndarray, avail: np.ndarray,
             cost_scale: np.ndarray, lambdas=LAMBDA_GRID) -> list[dict]:
    cnorm = np.where(avail, cost / np.nanmax(cost_scale), 0.0)
    cnorm = np.where(np.isfinite(cnorm), cnorm, 0.0)   # unavailable cells are masked by _choice anyway
    pts = []
    for lam in lambdas:
        ch = _choice(pred - lam * cnorm, avail)
        rr = _realized(ch, reward)
        cc = cost[np.arange(len(ch)), ch]
        cc = np.where(np.isfinite(cc), cc, 0.0)
        pts.append({"lambda": lam, "reward": float(rr.mean()), "cost": float(cc.mean())})
    return pts


def frontier_area(points: list[dict], ref_points: list[dict]) -> float:
    """Mean reward of the monotone upper envelope over the reference cost support."""
    def envelope(pts):
        arr = sorted({(p["cost"], p["reward"]) for p in pts})
        xs, ys = [], []
        best = -1
        for c, r in arr:
            best = max(best, r)
            xs.append(c); ys.append(best)
        return np.asarray(xs), np.asarray(ys)
    xr, yr = envelope(ref_points)
    if len(xr) < 2 or xr.max() <= xr.min():
        return float("nan")
    grid = np.linspace(xr.min(), xr.max(), 200)
    xs, ys = envelope(points)
    interp = np.interp(grid, xs, ys, left=ys[0] if xs.min() <= grid[0] else 0.0, right=ys[-1])
    return float(interp.mean())


def evaluate(result: dict, targets: Targets, cost: np.ndarray) -> dict:
    pred = result["pred"]
    avail = ~np.isnan(targets.reward)
    reward = targets.reward
    cov = result["covered"]
    choice = _choice(np.nan_to_num(pred, nan=-1.0), avail)
    router = _realized(choice, reward)
    oracle = targets.oracle_reward
    bs = _realized(result["best_single"], reward)
    db = _realized(result["domain_best"], reward)
    random_pick = 1.0 - targets.difficulty
    pred_diff = 1.0 - np.nanmean(np.where(avail, pred, np.nan), axis=1)
    sufficient = np.isclose(router, oracle)
    exact_cheapest = choice == targets.cheapest_sufficient
    # "cheapest predicted-sufficient" decision rule at threshold 0.5.
    pmask = avail & (pred >= 0.5)
    c_all = np.where(pmask, cost, np.inf)
    cheap_choice = np.where(pmask.any(1), np.argmin(c_all, axis=1), choice)
    cheap_reward = _realized(cheap_choice, reward)
    cheap_cost = cost[np.arange(len(cheap_choice)), cheap_choice]
    cheap_cost = np.where(np.isfinite(cheap_cost), cheap_cost, 0.0)
    argmax_cost = cost[np.arange(len(choice)), choice]
    argmax_cost = np.where(np.isfinite(argmax_cost), argmax_cost, 0.0)

    out = {"n_folds": result["n_folds"], "elapsed_s": result["elapsed_s"], "subsets": {}, "per_dataset": {}}
    masks = targets.subset_masks()
    for name, mask in masks.items():
        mk = mask & cov
        if mk.sum() < 20:
            continue
        aucs, briers = [], []
        for j in range(pred.shape[1]):
            rows = mk & avail[:, j] & ~np.isnan(pred[:, j])
            y = targets.success[rows, j]
            if rows.sum() >= 20 and y.min() != y.max():
                aucs.append(roc_auc_score(y, pred[rows, j]))
                briers.append(float(np.mean((pred[rows, j] - y) ** 2)))
        d_true, d_pred = targets.difficulty[mk], pred_diff[mk]
        ss_res = np.sum((d_true - d_pred) ** 2)
        ss_tot = np.sum((d_true - d_true.mean()) ** 2)
        gap = oracle[mk].mean() - bs[mk].mean()
        out["subsets"][name] = {
            "n": int(mk.sum()),
            "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "n_models_scored": len(aucs),
            "brier": float(np.mean(briers)) if briers else float("nan"),
            "difficulty_r2": float(1 - ss_res / max(ss_tot, 1e-12)),
            "difficulty_spearman": float(spearmanr(d_true, d_pred).correlation) if mk.sum() > 2 else float("nan"),
            "router_reward": float(router[mk].mean()),
            "oracle_reward": float(oracle[mk].mean()),
            "best_single_reward": float(bs[mk].mean()),
            "domain_best_reward": float(db[mk].mean()),
            "random_reward": float(random_pick[mk].mean()),
            "gap_closed_vs_best_single": float((router[mk].mean() - bs[mk].mean()) / gap) if gap > 1e-9 else float("nan"),
            "gap_closed_vs_domain_best": float((router[mk].mean() - db[mk].mean()) / max(oracle[mk].mean() - db[mk].mean(), 1e-9)),
            "sufficient_rate": float(sufficient[mk].mean()),
            "exact_cheapest_rate": float(exact_cheapest[mk].mean()),
            "argmax_cost": float(argmax_cost[mk].mean()),
            "cheapest_sufficient_rule_reward": float(cheap_reward[mk].mean()),
            "cheapest_sufficient_rule_cost": float(cheap_cost[mk].mean()),
        }
        if name == "all":
            pts = frontier(np.nan_to_num(pred[mk], nan=0.0), reward[mk], cost[mk], avail[mk], targets.model_cost_scale)
            ref = frontier(np.nan_to_num(reward[mk], nan=0.0), reward[mk], cost[mk], avail[mk], targets.model_cost_scale)
            out["frontier"] = {"router": pts, "oracle": ref,
                               "router_area": frontier_area(pts, ref), "oracle_area": frontier_area(ref, ref)}
    for g in np.unique(targets.dataset_index):
        mk = (targets.dataset_index == g) & cov
        if mk.sum() < 20:
            continue
        out["per_dataset"][int(g)] = {
            "n": int(mk.sum()), "router_reward": float(router[mk].mean()), "oracle_reward": float(oracle[mk].mean()),
            "best_single_reward": float(bs[mk].mean()), "domain_best_reward": float(db[mk].mean()),
            "difficulty_spearman": float(spearmanr(targets.difficulty[mk], pred_diff[mk]).correlation),
        }
    out["per_query"] = {"router_reward": router.astype(np.float32), "pred_difficulty": pred_diff.astype(np.float32),
                        "choice": choice.astype(np.int16), "covered": cov}
    return out
