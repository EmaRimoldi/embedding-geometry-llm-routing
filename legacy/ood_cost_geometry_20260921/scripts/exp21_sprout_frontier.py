"""exp21 - SPROUT cost-quality frontiers (mean cost, mean reward per lambda over the 11-lambda grid, 5 splits) for the
cell lookup, kNN-100, K-means-64 utility router, best single model and oracle, per query and equal weight over
families; AIQ (RouterBench normalisation) and reward at fixed budgets of 10 / 20 / 30 % of GPT-4o's mean cost per query
(linear interpolation on the non-decreasing hull). Env SPROUT_TOPICS=v2 uses the refined cells. Writes
results/exp21_sprout_frontier[_v2]/."""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import sprout_common as S
from exp19_sprout_routing import lookup

EXP = "exp21_sprout_frontier" + ("_v2" if os.environ.get("SPROUT_TOPICS") == "v2" else ""); LAMS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0); SEEDS = (0, 1, 2, 3, 4)
METHODS = ("cell_lookup", "knn100", "kmeans64_top1", "family_lookup", "best_single", "oracle")


def hull(pts):
    pts = pts[np.argsort(pts[:, 0])]; h = []
    for cc, rr in pts:
        if h and rr < h[-1][1]: continue
        h.append([cc, rr])
    return np.array(h)


def at_budget(pts, b):
    h = hull(pts)
    if b < h[0, 0]: return None
    return float(np.interp(b, h[:, 0], h[:, 1]))


def main():
    D = S.load(); r, c, fam, fn, names = D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, n = len(fn), D["n"]; xu = S.unit(D["x"]); _, cell = np.unique(D["cell_str"], return_inverse=True)
    strong = int(r.mean(0).argmax()); cref = float(c[:, strong].mean()); cref_eq = float(np.mean([c[fam == g, strong].mean() for g in range(F)]))
    acc = {m: {"pq": np.zeros((len(LAMS), 2)), "eq": np.zeros((len(LAMS), 2))} for m in METHODS}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=fam); ii = np.arange(len(te)); nb = tr[S.knn_idx(xu[te], xu[tr], 100)]
        km = KMeans(64, n_init=1, random_state=seed).fit(xu[tr]); ktr, kte = km.labels_, km.predict(xu[te])
        for li, lam in enumerate(LAMS):
            U = r - lam * c; ch = {"best_single": np.full(len(te), int(U[tr].mean(0).argmax())), "oracle": U[te].argmax(1), "family_lookup": np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)])[fam[te]], "cell_lookup": lookup(U, tr, te, fam, cell, F), "knn100": U[nb].mean(1).argmax(1)}
            cm = np.array([U[tr][ktr == k].mean(0).argmax() if (ktr == k).any() else int(U[tr].mean(0).argmax()) for k in range(64)]); ch["kmeans64_top1"] = cm[kte]
            for m in METHODS:
                cc, rr = c[te, ch[m]], r[te, ch[m]]; acc[m]["pq"][li] += [cc.mean() / len(SEEDS), rr.mean() / len(SEEDS)]; acc[m]["eq"][li] += [np.mean([cc[fam[te] == g].mean() for g in range(F)]) / len(SEEDS), np.mean([rr[fam[te] == g].mean() for g in range(F)]) / len(SEEDS)]
        print("seed", seed, flush=True)
    res = S.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMS), "strongest": names[strong], "c_ref_per_query": cref, "c_ref_equal_weight": cref_eq, "curves": {m: acc[m]["pq"].tolist() for m in METHODS}, "curves_macro": {m: acc[m]["eq"].tolist() for m in METHODS}})
    cmin, cmax = float(c.mean(0).min()), float(c.mean(0).max()); res["aiq_range"] = {}; res["budget"] = {}
    for m in METHODS:
        h = hull(acc[m]["pq"]); x = np.r_[cmin, np.clip(h[:, 0], cmin, cmax), cmax]; y = np.r_[h[0, 1], h[:, 1], h[-1, 1]]; o = np.argsort(x); res["aiq_range"][m] = float(np.trapezoid(y[o], x[o]) / (cmax - cmin))
        res["budget"][m] = {"pq": {f"{p}%": at_budget(acc[m]["pq"], p / 100 * cref) for p in (10, 20, 30)}, "eq": {f"{p}%": at_budget(acc[m]["eq"], p / 100 * cref_eq) for p in (10, 20, 30)}}
    md = [f"# {EXP} (SPROUT)", "", f"Strongest single model {names[strong]}: mean cost {cref*1e3:.2f}e-3 per query ({cref_eq*1e3:.2f}e-3 equal weight). Curves over the 11-lambda grid, 5 splits.", "", "method | AIQ (RouterBench norm.) | reward at 10% / 20% / 30% of GPT-4o cost, per query | equal weight", "---|---|---|---"]
    for m in METHODS:
        b = res["budget"][m]; f = lambda d: " / ".join("n/a" if d[k] is None else f"{d[k]:.3f}" for k in ("10%", "20%", "30%")); md.append(f"{m} | {res['aiq_range'][m]:.4f} | {f(b['pq'])} | {f(b['eq'])}")
    md += ["", "## curves (cost e-3, reward) per lambda, per query", ""] + [f"- {m}: " + " ".join(f"({p[0]*1e3:.2f},{p[1]:.3f})" for p in acc[m]["pq"]) for m in METHODS]
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
