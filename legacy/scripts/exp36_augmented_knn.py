"""exp36: kNN on the semantic embedding augmented with difficulty features read from a frozen LLM.

Space: z = [ unit(GTE) , w * B / sqrt(d_B) ] where B is a standardised difficulty block (d_B dims)
and w a weight chosen per (lambda, k) on an inner 80/20 split of the training rows (grid W).  Every
block is computed from training outcomes only; training-row values are cross-fitted (5 folds) so
that index and query carry out-of-sample predictions alike.  Reader: Qwen2.5-7B-Instruct, layer
21, last token (exp35), unless --reader/--layer say otherwise.
Blocks:
  diff        ridge-predicted solve count from the reader state (1-d)
  diff_gte    the same predicted from GTE itself (1-d control: what the semantic space already has)
  permodel    ridge-predicted success of every pool model from the reader state (M-d)
  joint       diff + permodel
  surprise    prompt log-prob / entropy scalars of the reader, raw (5-d, no regression)
  lens        solve count predicted from the reader's vocabulary projection at the last token (1-d)
  surface     log token counts (raw)
  pca64       unsupervised PCA-64 of the reader state (no outcome used)
  all         diff + permodel + surprise + lens + surface
  only_*      the block alone, without the semantic embedding (diff, permodel, all)
Routers: kNN k in {10, 40, 100} on each space; references: plain GTE kNN, family lookup,
(family, topic) cell lookup, best single model, oracle; and the (family, predicted-difficulty
quintile) lookup built from the diff block (cells below 50 training rows fall back to the family).
5 stratified 70/30 splits, 15-lambda grid, both aggregations, AIQ, paired bootstraps on split 0.
Writes results[_root]/difficulty_llm/exp36_augmented_knn/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, lengths_features, topic_str  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, aiq, split, summarise  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp36_augmented_knn"
KS = (10, 40, 100); W = (0.5, 1.0, 2.0, 4.0); REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight"); N_BOOT = 1000; MIN_CELL = 50; NBIN = 5
BLOCKS = ("diff", "diff_gte", "permodel", "joint", "surprise", "lens", "surface", "pca64", "all"); ONLY = ("diff", "permodel", "all")


def unit(a):
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def topk(q, index, k):
    nb = np.zeros((len(q), k), dtype=np.int64)
    for s in range(0, len(q), 2048):
        sims = q[s:s + 2048] @ index.T; top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1); nb[s:s + 2048] = np.take_along_axis(top, order, axis=1)
    return nb


def ridge_fit_predict(Xtr, Ytr, Xte, onehot_tr, onehot_te):
    sc = StandardScaler().fit(Xtr); m = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(np.column_stack([onehot_tr, sc.transform(Xtr)]), Ytr)
    return m.predict(np.column_stack([onehot_te, sc.transform(Xte)]))


def crossfit(X, Y, onehot, tr, te, fam, seed):
    """out-of-fold predictions on the training rows (5 stratified folds), full-fit predictions on the test rows."""
    ptr = np.zeros((len(tr),) + Y.shape[1:], dtype=np.float32)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, fam[tr]):
        ptr[b] = ridge_fit_predict(X[tr[a]], Y[tr[a]], X[tr[b]], onehot[tr[a]], onehot[tr[b]])
    return ptr, ridge_fit_predict(X[tr], Y[tr], X[te], onehot[tr], onehot[te]).astype(np.float32)


def boot(d, fte, F, rng):
    q = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    groups = [np.flatnonzero(fte == g) for g in range(F)]; groups = [g for g in groups if len(g)]
    f = [np.mean([d[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
    return {"per_query": {"point": float(d.mean()), "ci95": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))]},
            "per_family_equal_weight": {"point": float(np.mean([d[g].mean() for g in groups])), "ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]}}


def cell_lookup(cells_tr, cells_te, fam_tr, fam_te, Utr, F):
    fam_best = np.array([Utr[fam_tr == g].mean(0).argmax() for g in range(F)])
    best = {}
    for cl in np.unique(cells_tr):
        m = cells_tr == cl
        if m.sum() >= MIN_CELL: best[cl] = int(Utr[m].mean(0).argmax())
    return np.array([best.get(cl, fam_best[g]) for cl, g in zip(cells_te, fam_te)])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", default="qwen2.5-7b-instruct"); ap.add_argument("--layer", type=int, default=21); ap.add_argument("--pool", default="last")
    ap.add_argument("--wgrid", default=None, help="weights separated by commas or colons, e.g. 0.05:0.1:0.25:0.5:1:2:4"); ap.add_argument("--tag", default="", help="suffix of the output folder"); a = ap.parse_args()
    global W, OUT
    if a.wgrid: W = tuple(float(x) for x in a.wgrid.replace(":", ",").split(","))   # colon-separated also accepted (sbatch --export splits on commas)
    OUT = RESULTS_ROOT / "difficulty_llm" / (EXP + (("_" + a.tag) if a.tag else ""))
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); M = panel.m; onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0); solved = (r >= SUCCESS_THRESHOLD).astype(np.float32); y = solved.sum(1)
    g = unit(load_rep(find_rep(PANEL, REP), n, panel.keep))
    HID = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / a.reader
    def load(p):
        arr = np.load(p, mmap_mode="r"); arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr; assert arr.shape[0] == n, (p, arr.shape); return np.asarray(arr, dtype=np.float32)
    h = load(HID / f"L{a.layer}_{a.pool}.npy"); lens = load(HID / "logit_lens_last.npy"); surprise = load(ANALYSIS_ROOT / "features" / PANEL / "prompt" / "qwen7b_hidden" / "prompt_logprob_stats.npz") if False else None
    st = np.load(ANALYSIS_ROOT / "features" / PANEL / "prompt" / "qwen7b_hidden" / "prompt_logprob_stats.npz")["stats"]; surprise = (st[panel.keep] if panel.keep is not None and st.shape[0] == len(panel.keep) else st).astype(np.float32)
    surface, _ = lengths_features(panel); surface = surface.astype(np.float32); topics = np.array([str(t) for t in topic_str(panel)], dtype=object)
    per = {}; boots = {}; chosen_w = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte, ftr = fam[te], fam[tr]; dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        # ---- blocks (training rows cross-fitted, test rows out of sample)
        B = {}
        B["diff"] = crossfit(h, y, onehot, tr, te, fam, seed); B["diff_gte"] = crossfit(g, y, onehot, tr, te, fam, seed); B["lens"] = crossfit(lens, y, onehot, tr, te, fam, seed)
        B["permodel"] = crossfit(h, solved, onehot, tr, te, fam, seed)
        B["surprise"] = (surprise[tr], surprise[te]); B["surface"] = (surface[tr], surface[te])
        sc = StandardScaler().fit(h[tr]); p = PCA(64, random_state=0).fit(sc.transform(h[tr])); B["pca64"] = (p.transform(sc.transform(h[tr])), p.transform(sc.transform(h[te])))
        cat = lambda ks: (np.column_stack([np.atleast_2d(B[k][0].T).T for k in ks]), np.column_stack([np.atleast_2d(B[k][1].T).T for k in ks]))  # noqa: E731
        B["joint"] = cat(("diff", "permodel")); B["all"] = cat(("diff", "permodel", "surprise", "lens", "surface"))
        for k in list(B):
            btr, bte = (np.asarray(v).reshape(len(v), -1) for v in B[k]); s = StandardScaler().fit(btr); d = btr.shape[1]
            B[k] = ((s.transform(btr) / np.sqrt(d)).astype(np.float32), (s.transform(bte) / np.sqrt(d)).astype(np.float32))
        print(f"seed {seed}: blocks ready ({time.time() - t0:.0f}s)", flush=True)
        # ---- inner split of the training rows for the weight
        itr, ite = split(len(tr), ftr, 100 + seed); itr, ite = np.sort(itr), np.sort(ite)
        spaces = {"plain": (g[tr], g[te])}
        for k in BLOCKS:
            for w in W: spaces[f"{k}@w{w:g}"] = (np.column_stack([g[tr], w * B[k][0]]), np.column_stack([g[te], w * B[k][1]]))
        for k in ONLY: spaces[f"only_{k}"] = (unit(B[k][0]), unit(B[k][1]))
        inner_nb = {name: topk(vtr[ite], vtr[itr], max(KS)) for name, (vtr, vte) in spaces.items() if "@w" in name}
        outer_nb = {name: topk(vte, vtr, max(KS)) for name, (vtr, vte) in spaces.items()}
        # difficulty-bin cells from the diff block (training quantiles per family)
        dtr, dte = B["diff"][0][:, 0], B["diff"][1][:, 0]; bins_tr = np.zeros(len(tr), int); bins_te = np.zeros(len(te), int)
        for gi in range(F):
            qs = np.quantile(dtr[ftr == gi], np.linspace(0, 1, NBIN + 1)[1:-1]) if (ftr == gi).sum() else np.array([])
            bins_tr[ftr == gi] = np.searchsorted(qs, dtr[ftr == gi]); bins_te[fte == gi] = np.searchsorted(qs, dte[fte == gi])
        cells_diff_tr = np.array([f"{names[gi]}#d{b}" for gi, b in zip(ftr, bins_tr)]); cells_diff_te = np.array([f"{names[gi]}#d{b}" for gi, b in zip(fte, bins_te)])
        res = {}; pq = {}; curves = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]; Uitr = Utr[itr]
            choices = {"family_lookup": np.array([Utr[ftr == gi].mean(0).argmax() for gi in range(F)])[fte], "cell_lookup": cell_lookup(topics[tr], topics[te], ftr, fte, Utr, F),
                       "diffbin_lookup": cell_lookup(cells_diff_tr, cells_diff_te, ftr, fte, Utr, F), "best_single": np.full(len(te), int(Utr.mean(0).argmax())), "oracle": U[te].argmax(1)}
            for k in KS:
                choices[f"plain_knn{k}"] = Utr[outer_nb["plain"][:, :k]].mean(1).argmax(1)
                for o in ONLY: choices[f"only_{o}_knn{k}"] = Utr[outer_nb[f"only_{o}"][:, :k]].mean(1).argmax(1)
                for b in BLOCKS:
                    # pick w on the inner split: mean utility of the inner test rows
                    iu = {w: U[tr[itr]][inner_nb[f"{b}@w{w:g}"][:, :k]].mean(1).argmax(1) for w in W}
                    iu = {w: float(U[tr[ite]][np.arange(len(ite)), ch].mean()) for w, ch in iu.items()}; w = max(iu, key=iu.get); chosen_w.setdefault(f"{b}_knn{k}", {}).setdefault(str(lam), []).append(w)
                    choices[f"{b}_knn{k}"] = Utr[outer_nb[f"{b}@w{w:g}"][:, :k]].mean(1).argmax(1)
            for name, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); pq[(name, lam)] = s.pop("_per_query_utility"); res.setdefault(name, {})[str(lam)] = s
                curves.setdefault(name, []).append([s["all"]["per_query"]["cost"], s["all"]["per_query"]["reward"]])
        max_cost = max(np.array(v)[:, 0].max() for v in curves.values()); per[seed] = {"results": res, "aiq": {name: aiq(np.array(v), max_cost) for name, v in curves.items()}}
        if seed == SEEDS[0]:
            rng = np.random.default_rng(0)
            for lam in REPORT:
                for k in (10, 100):
                    for b in ("diff", "permodel", "joint", "surprise", "all", "pca64", "diff_gte"):
                        boots[f"{b}_knn{k} - plain_knn{k} @ λ={lam:g}"] = boot(pq[(f"{b}_knn{k}", lam)] - pq[(f"plain_knn{k}", lam)], fte, F, rng)
                    boots[f"all_knn{k} - cell_lookup @ λ={lam:g}"] = boot(pq[(f"all_knn{k}", lam)] - pq[("cell_lookup", lam)], fte, F, rng)
                boots[f"diffbin_lookup - cell_lookup @ λ={lam:g}"] = boot(pq[("diffbin_lookup", lam)] - pq[("cell_lookup", lam)], fte, F, rng)
                boots[f"diffbin_lookup - family_lookup @ λ={lam:g}"] = boot(pq[("diffbin_lookup", lam)] - pq[("family_lookup", lam)], fte, F, rng)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); methods = list(per[SEEDS[0]]["results"])
    def agg(name, lam, subset, mode, key):
        v = [per[s]["results"][name][str(lam)][subset][mode][key] for s in SEEDS]; return float(np.mean(v)), float(min(v)), float(max(v))
    out = {**panel.provenance(), "experiment": EXP, "representation": REP, "reader": a.reader, "layer": a.layer, "pool": a.pool, "ks": list(KS), "weights": list(W), "seeds": list(SEEDS), "methods": methods, "blocks": list(BLOCKS),
           "by_lambda": {str(lam): {name: {subset: {mode: {key: agg(name, lam, subset, mode, key) for key in ("utility", "reward", "cost")} for mode in AGGS} for subset in ("all", "disagreement")} for name in methods} for lam in LAMBDAS},
           "per_family": {str(lam): {name: {names[gi]: float(np.mean([per[s]["results"][name][str(lam)]["per_family"][gi]["utility"] for s in SEEDS])) for gi in range(F)} for name in methods} for lam in REPORT},
           "aiq": {name: [per[s]["aiq"][name] for s in SEEDS] for name in methods}, "bootstrap_split0": boots, "chosen_w": chosen_w, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# {EXP} — kNN on GTE ⊕ difficulty features from frozen {a.reader} (L{a.layer}, {a.pool}) — {panel.name}", "",
          "z = [unit(GTE), w·block/√d]; block standardised on training rows, training-row values cross-fitted (5 folds); w ∈ {0.5, 1, 2, 4} chosen per (λ, k) on an inner 80/20 split of the training rows. 5 splits, mean over splits; CIs = paired bootstraps on split 0.", ""]
    cols = ["family_lookup", "cell_lookup", "diffbin_lookup"] + [f"plain_knn{k}" for k in (10, 100)] + [f"{b}_knn{k}" for k in (10, 100) for b in BLOCKS] + [f"only_{o}_knn{k}" for k in (10, 100) for o in ONLY] + ["best_single", "oracle"]
    for mode in AGGS:
        md += [f"## Utility, {mode}", "", "method | " + " | ".join(f"λ={lam:g}" for lam in REPORT), "---|" + "---|" * len(REPORT)]
        for name in cols: md.append(f"{name} | " + " | ".join(f"{agg(name, lam, 'all', mode, 'utility')[0]:.4f}" for lam in REPORT))
        md.append("")
    md += ["## AIQ (mean over splits)", "", "method | AIQ", "---|---"] + [f"{name} | {np.mean(out['aiq'][name]):.4f}" for name in cols] + [""]
    md += ["## Paired bootstraps, split 0 (per query / equal weight, 95% CI)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    md += ["", "## Chosen weights (mode over splits) at the report λ", "", "space | " + " | ".join(f"λ={lam:g}" for lam in REPORT), "---|" + "---|" * len(REPORT)]
    for b in BLOCKS:
        for k in (10, 100):
            md.append(f"{b}_knn{k} | " + " | ".join(f"{max(set(chosen_w[f'{b}_knn{k}'][str(lam)]), key=chosen_w[f'{b}_knn{k}'][str(lam)].count):g}" for lam in REPORT))
    md += ["", "## Per-family utility at λ = 0 and 100 (mean over splits)", ""]
    for lam in (0.0, 100.0):
        md += [f"### λ = {lam:g}", "", "method | " + " | ".join(names), "---|" + "---|" * F]
        for name in ["family_lookup", "cell_lookup", "diffbin_lookup", "plain_knn10", "diff_knn10", "permodel_knn10", "all_knn10", "plain_knn100", "diff_knn100", "permodel_knn100", "all_knn100"]:
            md.append(f"{name} | " + " | ".join(f"{out['per_family'][str(lam)][name][names[gi]]:.4f}" for gi in range(F)))
        md.append("")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
