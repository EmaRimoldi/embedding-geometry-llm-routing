"""exp18: can a regressor on the GTE embedding recover "how many models solve this query"
beyond the family (and family x topic) mean?  See experiments/exp18_recovering_difficulty.yaml.

    python scripts/exp18_recovering_difficulty.py [--boost hgb|xgb|none] [--seeds 0 1 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel, text_controls  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp18_recovering_difficulty"
OUT = RESULTS_ROOT / "recovering_difficulty" / EXP
KNN_KS = (5, 10, 40, 100); MIN_CELL = 50; N_BOOT = 1000; NULL_PERMS_CHEAP = 3


# ----------------------------------------------------------------------------- regressors
def fit_ridge(xtr, ytr, xte):
    return RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, ytr).predict(xte)


def knn_index(utr, ute, kmax):
    nn = NearestNeighbors(n_neighbors=kmax, metric="cosine", algorithm="brute", n_jobs=8).fit(utr)
    return nn.kneighbors(ute)[1]                                    # sorted by distance


def fit_boost(backend, xtr, ytr, xte, seed=0):
    if backend == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
        m.fit(xtr, ytr); return m.predict(xte), int(m.n_iter_)
    if backend == "xgb":
        import xgboost as xgb
        rng = np.random.default_rng(seed); perm = rng.permutation(len(xtr)); nv = len(xtr) // 10
        va, fit = perm[:nv], perm[nv:]
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", early_stopping_rounds=20, n_jobs=8, random_state=seed)
        m.fit(xtr[fit], ytr[fit], eval_set=[(xtr[va], ytr[va])], verbose=False)
        return m.predict(xte), int(m.best_iteration + 1)
    raise ValueError(backend)


# ----------------------------------------------------------------------------- metrics
def metrics(y, yhat, floor, fam, F):
    sse = float(((y - yhat) ** 2).sum()); sse_f = float(((y - floor) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum())
    out = {"r2": 1 - sse / sst, "r2_beyond_family": 1 - sse / sse_f, "mae": float(np.abs(y - yhat).mean()),
           "spearman": float(spearmanr(y, yhat).correlation) if np.std(yhat) > 0 else 0.0, "per_family": {}}
    for g in range(F):
        m = fam == g
        if m.sum() < 10: continue
        s_ = float(((y[m] - yhat[m]) ** 2).sum()); sf = float(((y[m] - floor[m]) ** 2).sum())
        out["per_family"][g] = {"r2_beyond_family": 1 - s_ / sf if sf > 0 else float("nan"), "mae": float(np.abs(y[m] - yhat[m]).mean()),
                                "spearman": float(spearmanr(y[m], yhat[m]).correlation) if np.std(yhat[m]) > 0 and np.std(y[m]) > 0 else 0.0}
    return out


def boot_r2b(y, yhat, floor, rng, n=N_BOOT):
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)); vals.append(1 - ((y[idx] - yhat[idx]) ** 2).sum() / ((y[idx] - floor[idx]) ** 2).sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--boost", default="hgb", choices=["hgb", "xgb", "none"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS)); args = ap.parse_args()
    t0 = time.time()
    panel = load_panel(PANEL); spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep).astype(np.float32)
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    bow = load_rep(find_rep(PANEL, "computed/hash-bow@prompt"), panel.n, panel.keep).astype(np.float32)
    fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); n = panel.n
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    targets = {"n_solved": (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64), "reward_sum": r.sum(1)}
    # surface features: 10 text controls + log token counts (lengths.npz was computed on the full release -> mask)
    extra, extra_names = lengths_features(panel)
    surf, surf_names = text_controls(panel.prompts); surf = np.column_stack([surf, extra]).astype(np.float64)
    surf_names = surf_names + extra_names
    # family x topic cells (MMLU subject; HellaSwag source), as exp15/16
    cell_str = topic_str(panel)
    _, cell = np.unique(cell_str, return_inverse=True)
    onehot = np.eye(F, dtype=np.float32)[fam]

    def lookup(tr, te, y, groups, fallback=None):
        ytr = y[tr]; gtr, gte_ = groups[tr], groups[te]
        pred = fallback.copy() if fallback is not None else np.full(len(te), ytr.mean())
        for g in np.unique(gte_):
            rows = gtr == g
            if rows.sum() >= (MIN_CELL if fallback is not None else 1): pred[gte_ == g] = ytr[rows].mean()
        return pred

    def feats(kind, tr, te):
        if kind == "gte": sc = StandardScaler().fit(x[tr]); return sc.transform(x[tr]), sc.transform(x[te])
        if kind == "surface": sc = StandardScaler().fit(surf[tr]); return sc.transform(surf[tr]), sc.transform(surf[te])
        if kind == "bow": sc = StandardScaler().fit(bow[tr]); return sc.transform(bow[tr]), sc.transform(bow[te])
        if kind.startswith("family+"):
            a, b = feats(kind.split("+", 1)[1], tr, te); w = np.sqrt(a.shape[1])
            return np.hstack([onehot[tr] * w, a]), np.hstack([onehot[te] * w, b])
        raise ValueError(kind)

    results = {t: {} for t in targets}; null_results = {t: {} for t in targets}; boot = {t: {} for t in targets}; iters = {}
    target_desc = {t: {"panel": {"mean": float(y.mean()), "sd": float(y.std())},
                       "per_family": {names[g]: {"n": int((fam == g).sum()), "mean": float(y[fam == g].mean()), "sd": float(y[fam == g].std())} for g in range(F)},
                       "within_family_variance_share": float(sum(((y[fam == g] - y[fam == g].mean()) ** 2).sum() for g in range(F)) / ((y - y.mean()) ** 2).sum())}
                   for t, y in targets.items()}

    def record(store, t, name, seed, m):
        store[t].setdefault(name, {})[seed] = m

    for seed in args.seeds:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); rng = np.random.default_rng(seed)
        nb = knn_index(u[tr], u[te], max(KNN_KS))
        cache = {k: feats(k, tr, te) for k in ("gte", "surface", "bow", "family+surface", "family+gte")}
        for t, y in targets.items():
            ytr, yte, fte = y[tr], y[te], fam[te]
            floor = lookup(tr, te, y, fam); preds = {"floor:family_mean": floor, "floor:cell_mean": lookup(tr, te, y, cell, fallback=floor)}
            for k in ("surface", "bow", "gte", "family+surface", "family+gte"):
                a, b = cache[k]; preds[f"ridge:{k}"] = fit_ridge(a, ytr, b)
            for k in KNN_KS: preds[f"knn{k}:gte"] = ytr[nb[:, :k]].mean(1)
            if args.boost != "none":
                for k in ("surface", "gte", "family+surface", "family+gte"):
                    a, b = cache[k]; p, it = fit_boost(args.boost, a, ytr, b, seed); preds[f"boost_{args.boost}:{k}"] = p; iters[f"{t}|{k}|{seed}"] = it
            for name, p in preds.items():
                record(results, t, name, seed, metrics(yte, p, floor, fte, F))
                if seed == args.seeds[0]: boot[t][name] = boot_r2b(yte, p, floor, np.random.default_rng(0))
            # within-family permutation null on the train fold
            nulls = {"ridge:gte": [], "knn40:gte": []}
            if args.boost != "none": nulls[f"boost_{args.boost}:gte"] = []
            for pi in range(NULL_PERMS_CHEAP):
                yp = ytr.copy()
                for g in range(F):
                    rows = np.flatnonzero(fam[tr] == g); yp[rows] = ytr[rows][rng.permutation(len(rows))]
                a, b = cache["gte"]
                nulls["ridge:gte"].append(metrics(yte, fit_ridge(a, yp, b), floor, fte, F)["r2_beyond_family"])
                nulls["knn40:gte"].append(metrics(yte, yp[nb[:, :40]].mean(1), floor, fte, F)["r2_beyond_family"])
                if args.boost != "none" and pi == 0:
                    nulls[f"boost_{args.boost}:gte"].append(metrics(yte, fit_boost(args.boost, a, yp, b, seed)[0], floor, fte, F)["r2_beyond_family"])
            for name, v in nulls.items(): record(null_results, t, name, seed, v)
            print(f"seed {seed} target {t} done ({time.time() - t0:.0f}s): " + ", ".join(f"{k}={v[seed]['r2_beyond_family']:+.3f}" for k, v in results[t].items()), flush=True)

    # ------------------------------------------------------------------ aggregate + write
    OUT.mkdir(parents=True, exist_ok=True)
    agg = {}
    for t in targets:
        agg[t] = {}
        for name, per_seed in results[t].items():
            ms = list(per_seed.values())
            row = {k: {"mean": float(np.mean([m[k] for m in ms])), "sd": float(np.std([m[k] for m in ms]))} for k in ("r2", "r2_beyond_family", "mae", "spearman")}
            row["per_family"] = {names[g]: {k: float(np.nanmean([m["per_family"][g][k] for m in ms if g in m["per_family"]])) for k in ("r2_beyond_family", "mae", "spearman")}
                                 for g in range(F) if any(g in m["per_family"] for m in ms)}
            row["r2_beyond_family_ci95_split0"] = boot[t].get(name); row["n_splits"] = len(ms); agg[t][name] = row
        for name, per_seed in null_results[t].items():
            vals = [v for vs in per_seed.values() for v in vs]
            agg[t][f"null(within-family shuffle):{name}"] = {"r2_beyond_family": {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "max": float(np.max(vals))}, "n": len(vals)}
    out = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "boost_backend": args.boost, "seeds": args.seeds, "knn_ks": list(KNN_KS), "min_cell": MIN_CELL,
           "surface_features": surf_names, "targets": target_desc, "boost_iterations": iters, "results": agg}
    (OUT / "result.json").write_text(json.dumps(out, indent=1, default=float))

    md = [f"# {EXP}", "", f"Boosting backend: `{args.boost}`. Splits: seeds {args.seeds} (70/30, stratified by family). Metrics on the test rows, mean ± sd over splits.",
          "R²_beyond_family = 1 − SSE(method) / SSE(family-mean floor): the share of the variance *left after the family mean* that the method explains (0 = the family floor; < 0 = worse than the floor). CI: paired bootstrap over test queries of split 0.", ""]
    for t in targets:
        d = target_desc[t]
        md += [f"## Target `{t}` — panel mean {d['panel']['mean']:.2f}, sd {d['panel']['sd']:.2f}; share of variance within families: {d['within_family_variance_share']:.3f}", "",
               "family | n | mean | sd", "---|---|---|---"]
        md += [f"{g} | {v['n']} | {v['mean']:.2f} | {v['sd']:.2f}" for g, v in d["per_family"].items()]
        md += ["", "method | R² (panel) | R²_beyond_family | 95% CI (split 0) | MAE | Spearman (panel)", "---|---|---|---|---|---"]
        for name, row in agg[t].items():
            if name.startswith("null"): continue
            ci = row["r2_beyond_family_ci95_split0"]
            md.append(f"{name} | {row['r2']['mean']:.3f} ± {row['r2']['sd']:.3f} | {row['r2_beyond_family']['mean']:+.3f} ± {row['r2_beyond_family']['sd']:.3f} | "
                      f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['mae']['mean']:.3f} | {row['spearman']['mean']:.3f}")
        for name, row in agg[t].items():
            if name.startswith("null"):
                md.append(f"{name} | – | {row['r2_beyond_family']['mean']:+.3f} ± {row['r2_beyond_family']['sd']:.3f} (max {row['r2_beyond_family']['max']:+.3f}, n={row['n']}) | – | – | –")
        md += ["", "### Per family: R²_beyond_family (mean over splits)", "", "method | " + " | ".join(names), "---|" + "---|" * F]
        for name, row in agg[t].items():
            if name.startswith("null"): continue
            md.append(f"{name} | " + " | ".join(f"{row['per_family'][g]['r2_beyond_family']:+.3f}" if g in row["per_family"] else "–" for g in names))
        md += ["", "### Per family: Spearman (mean over splits)", "", "method | " + " | ".join(names), "---|" + "---|" * F]
        for name, row in agg[t].items():
            if name.startswith("null"): continue
            md.append(f"{name} | " + " | ".join(f"{row['per_family'][g]['spearman']:.3f}" if g in row["per_family"] else "–" for g in names))
        md.append("")
    if iters: md += ["Boosting iterations used (early stopping): " + ", ".join(f"{k}: {v}" for k, v in iters.items()), ""]
    (OUT / "result.md").write_text("\n".join(md)); print("\n".join(md)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
