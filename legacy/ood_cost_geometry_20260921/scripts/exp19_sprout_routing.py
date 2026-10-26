"""exp19 - Section 4 routing measurements on SPROUT (GTE-Qwen2-7B; six families; cells = authoritative + accepted Qwen
topics, pending -> family). Five family-stratified 70/30 splits. Utility = reward - lambda * cost. Methods: best single
(train), oracle, family lookup, (family, topic) cell lookup (MIN_CELL 50, family fallback), predicted-cell lookup (linear
classifier on standardised GTE), kNN-5/10/40/100 (cosine, mean utility over neighbours), 100 random training rows of the
cell, K-means-64 utility router (top-1 cluster), per query and equal weight over families. Agreement rates, paired
bootstraps (cell - kNN-100, family - kNN-100, cell - family) per split, per-family utilities, model shares.
Writes results/exp19_sprout_routing/."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = "exp19_sprout_routing"; import os; EXP = EXP + ("_v2" if os.environ.get("SPROUT_TOPICS") == "v2" else ""); SEEDS = (0, 1, 2, 3, 4); MIN_CELL = 50
LAMS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0); REPORT = (0.0, 0.5, 5.0, 25.0, 50.0, 250.0, 500.0, 1000.0)   # reported grid = half the RouterBench grid (c_strongest ratio 1.5)
METHODS = ["best_single", "family_lookup", "cell_lookup", "predicted_cell_lookup", "knn5", "knn10", "knn40", "knn100", "random_cell_100", "kmeans64_top1", "oracle"]


def lookup(U, tr, te, fam, cell, F):
    fb = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)]); ch = fb[fam[te]].copy()
    for k in np.unique(cell[te]):
        m = cell[tr] == k
        if m.sum() >= MIN_CELL: ch[cell[te] == k] = U[tr][m].mean(0).argmax()
    return ch


def boot(d, B, rng):
    return [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(B)]


def main():
    t0 = time.time(); D = S.load(); r, c, fam, fn, names = D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, n, M = len(fn), D["n"], len(names); xu = S.unit(D["x"])
    _, cell = np.unique(D["cell_str"], return_inverse=True); ncell = int(cell.max()) + 1
    cbar, rbar = c.mean(0), r.mean(0); strong = int(rbar.argmax()); res = S.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMS), "strongest_single_model": names[strong], "c_ref_strongest_mean_cost": float(cbar[strong]), "lambda_scale_note": f"lambda_RB-equivalent = lambda * {cbar[strong]:.4g} / 3.25e-3; reported grid 0, 0.5, 5, 25, 50, 250 = half the RouterBench grid", "mean_cost_per_model": dict(zip(names, cbar.tolist())), "mean_reward_per_model": dict(zip(names, rbar.tolist()))})
    per = {m: {f"{l:g}": {"qw": [], "eq": [], "fam": []} for l in LAMS} for m in METHODS}; agree = {f"{l:g}": {k: [] for k in ("knn100_vs_cell", "pred_vs_cell", "cell_vs_family", "knn100_vs_family", "random_vs_cell")} for l in LAMS}
    paired = {f"{l:g}": {k: {"qw_mean": [], "qw_lo": [], "qw_hi": [], "eq_mean": [], "eq_lo": [], "eq_hi": []} for k in ("cell-knn100", "family-knn100", "cell-family", "pred-cell")} for l in LAMS}
    shares = {f"{l:g}": {"cell_lookup": np.zeros(M), "knn100": np.zeros(M)} for l in LAMS}; pred_acc = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=fam); ii = np.arange(len(te))
        nb = {k: tr[S.knn_idx(xu[te], xu[tr], k)] for k in (100,)}; nb100 = nb[100]; nb[40] = nb100[:, :40]; nb[10] = nb100[:, :10]; nb[5] = nb100[:, :5]
        cnt = np.bincount(cell[tr], minlength=ncell); rc = np.empty((len(te), 100), dtype=np.int64)
        for k in np.unique(cell[te]):
            pool = tr[cell[tr] == k] if cnt[k] >= MIN_CELL else tr[fam[tr] == fam[te][cell[te] == k][0]]; m = cell[te] == k; rc[m] = rng.choice(pool, (m.sum(), 100), replace=True)
        sc = StandardScaler().fit(xu[tr]); clf = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(xu[tr]), cell[tr]); pcell = clf.predict(sc.transform(xu[te])); pred_acc.append(float((pcell == cell[te]).mean()))
        km = KMeans(64, n_init=1, random_state=seed).fit(xu[tr]); ktr, kte = km.labels_, km.predict(xu[te]); print("seed", seed, "prep done", time.time() - t0, flush=True)
        for lam in LAMS:
            U = r - lam * c; Ute = U[te]; L = f"{lam:g}"; ch = {}
            ch["best_single"] = np.full(len(te), int(U[tr].mean(0).argmax())); ch["oracle"] = Ute.argmax(1)
            fb = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)]); ch["family_lookup"] = fb[fam[te]]
            ch["cell_lookup"] = lookup(U, tr, te, fam, cell, F); cell_pred = cell.copy(); cell_pred[te] = pcell; ch["predicted_cell_lookup"] = lookup(U, tr, te, fam, cell_pred, F)
            for k in (5, 10, 40, 100): ch[f"knn{k}"] = U[nb[k]].mean(1).argmax(1)
            ch["random_cell_100"] = U[rc].mean(1).argmax(1)
            cm = np.array([U[tr][ktr == k].mean(0).argmax() if (ktr == k).any() else int(U[tr].mean(0).argmax()) for k in range(64)]); ch["kmeans64_top1"] = cm[kte]
            u = {m: Ute[ii, ch[m]] for m in METHODS}
            for m in METHODS:
                pf = [float(u[m][fam[te] == g].mean()) for g in range(F)]; per[m][L]["qw"].append(float(u[m].mean())); per[m][L]["eq"].append(float(np.mean(pf))); per[m][L]["fam"].append(pf)
            agree[L]["knn100_vs_cell"].append(float((ch["knn100"] == ch["cell_lookup"]).mean())); agree[L]["pred_vs_cell"].append(float((ch["predicted_cell_lookup"] == ch["cell_lookup"]).mean())); agree[L]["cell_vs_family"].append(float((ch["cell_lookup"] == ch["family_lookup"]).mean())); agree[L]["knn100_vs_family"].append(float((ch["knn100"] == ch["family_lookup"]).mean())); agree[L]["random_vs_cell"].append(float((ch["random_cell_100"] == ch["cell_lookup"]).mean()))
            shares[L]["cell_lookup"] += np.bincount(ch["cell_lookup"], minlength=M) / len(te) / len(SEEDS); shares[L]["knn100"] += np.bincount(ch["knn100"], minlength=M) / len(te) / len(SEEDS)
            for key, (a, b) in {"cell-knn100": ("cell_lookup", "knn100"), "family-knn100": ("family_lookup", "knn100"), "cell-family": ("cell_lookup", "family_lookup"), "pred-cell": ("predicted_cell_lookup", "cell_lookup")}.items():
                d = u[a] - u[b]; bq = boot(d, 1000, rng); df = np.array([d[fam[te] == g].mean() for g in range(F)]); be = boot(df, 1000, rng)
                P = paired[L][key]; P["qw_mean"].append(float(d.mean())); P["qw_lo"].append(float(np.percentile(bq, 2.5))); P["qw_hi"].append(float(np.percentile(bq, 97.5))); P["eq_mean"].append(float(df.mean())); P["eq_lo"].append(float(np.percentile(be, 2.5))); P["eq_hi"].append(float(np.percentile(be, 97.5)))
        print("seed", seed, "done", time.time() - t0, flush=True)
    def ms(v): return [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
    res["results"] = {m: {L: {"qw": ms(per[m][L]["qw"]), "eq": ms(per[m][L]["eq"]), "per_family_mean": dict(zip(fn, np.mean(per[m][L]["fam"], 0).round(4).tolist()))} for L in per[m]} for m in METHODS}
    res["agreement"] = {L: {k: float(np.mean(v)) for k, v in a.items()} for L, a in agree.items()}; res["predicted_cell_accuracy"] = pred_acc
    res["paired"] = {L: {k: {"qw": [float(np.mean(P["qw_mean"])), float(np.mean(P["qw_lo"])), float(np.mean(P["qw_hi"])), int(sum(lo > 0 for lo in P["qw_lo"])), int(sum(hi < 0 for hi in P["qw_hi"]))], "eq": [float(np.mean(P["eq_mean"])), float(np.mean(P["eq_lo"])), float(np.mean(P["eq_hi"])), int(sum(lo > 0 for lo in P["eq_lo"])), int(sum(hi < 0 for hi in P["eq_hi"]))]} for k, P in d_.items()} for L, d_ in paired.items()}
    res["model_shares"] = {L: {m: dict(zip(names, v.round(3).tolist())) for m, v in s.items()} for L, s in shares.items()}
    md = [f"# {EXP} (SPROUT)", "", f"Rows {n}; families " + ", ".join(f"{f} {int((fam==g).sum())}" for g, f in enumerate(fn)) + f"; cells {ncell} (topic sources {res['topic_sources']}). Strongest single model {names[strong]} (mean reward {rbar[strong]:.3f}, mean cost {cbar[strong]*1e3:.2f}e-3; RouterBench GPT-4 3.25e-3, so lambda here ~ {3.25e-3/cbar[strong]:.2f} x RouterBench's lambda at equal lambda*c_strongest). Predicted-cell accuracy per split: " + ", ".join(f"{a:.3f}" for a in pred_acc) + ".", ""]
    for agg, nm in (("qw", "per query"), ("eq", "equal weight over families")):
        md += [f"## Utility, {nm} (mean over 5 splits)", "", "router | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
        for m in METHODS: md.append(f"{m} | " + " | ".join(f"{res['results'][m][f'{l:g}'][agg][0]:.3f}" for l in LAMS))
        md.append("")
    md += ["## Agreement (share of test queries with the same model)", "", "λ | kNN100=cell | pred=cell | cell=family | kNN100=family | random-cell=cell", "---|---|---|---|---|---"] + [f"{L} | " + " | ".join(f"{res['agreement'][L][k]:.3f}" for k in ("knn100_vs_cell", "pred_vs_cell", "cell_vs_family", "knn100_vs_family", "random_vs_cell")) for L in res["agreement"]] + [""]
    md += ["## Paired differences (first minus second; mean over splits of point estimate [mean 95% CI]; n of 5 splits with CI > 0 / < 0)", ""]
    for key in ("cell-knn100", "family-knn100", "cell-family", "pred-cell"):
        md += [f"### {key}", "", "λ | per query | splits +/- | equal weight | splits +/-", "---|---|---|---|---"] + [f"{L} | {v[key]['qw'][0]:+.4f} [{v[key]['qw'][1]:+.4f}, {v[key]['qw'][2]:+.4f}] | {v[key]['qw'][3]}+ / {v[key]['qw'][4]}- | {v[key]['eq'][0]:+.4f} [{v[key]['eq'][1]:+.4f}, {v[key]['eq'][2]:+.4f}] | {v[key]['eq'][3]}+ / {v[key]['eq'][4]}-" for L, v in res["paired"].items()] + [""]
    for lam in (5.0, 50.0):
        L = f"{lam:g}"; md += [f"## Per family at λ={L} (mean over splits)", "", "family | " + " | ".join(METHODS), "---|" + "---|" * len(METHODS)] + [f"{f} | " + " | ".join(f"{res['results'][m][L]['per_family_mean'][f]:.3f}" for m in METHODS) for f in fn] + [""]
        md += [f"model shares at λ={L}: cell lookup " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(res["model_shares"][L]["cell_lookup"].items(), key=lambda t: -t[1])[:4]) + "; kNN-100 " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(res["model_shares"][L]["knn100"].items(), key=lambda t: -t[1])[:4]), ""]
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
