"""exp26 step 2 (CPU job): the in-domain suite on the templated representation ("prompt") and the deformatted one ("core"), side by side,
under both cost conventions (recorded / corrected).  See experiments/exp26_deformatting.yaml.
Same rows, splits, decoders and metrics as exp01 / exp08 / exp14 / exp18 / exp20."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME  # noqa: E402
from deformat import PANEL_DIR  # noqa: E402

OUT = OUT_OVERRIDE or HERE; KS = (5, 10, 40, 100); MIN_CELL = 50; N_BOOT = 1000; AGGS = ("per_query", "per_family_equal_weight"); REPORT_LAMS = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0)


def unit(a): return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def purity(labels, truth):
    F = truth.max() + 1; cont = np.zeros((labels.max() + 1, F), int)
    for cl in range(labels.max() + 1): cont[cl] = np.bincount(truth[labels == cl], minlength=F)
    return float(cont.max(1).sum() / len(labels)), cont


def main() -> None:
    panel = load_panel("routerbench-0shot"); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); n_solved = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
    COSTS = {"recorded": np.nan_to_num(panel.costs.astype(np.float64), nan=0.0), "corrected": np.load(PANEL_DIR / "costs_corrected.npy")}
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str); rec = rec[panel.keep] if panel.keep is not None else rec
    src = np.array(["wikihow" if re.search(r"\[(header|title|step|substeps)\]", str(t)) else "activitynet" for t in panel.prompts])
    subj = np.array([s_.split(".")[0] for s_ in rec]); cell_str = np.where(ds == "mmlu", subj, np.where(ds == "hellaswag", "hellaswag/" + src, ds))
    cell_names, cell = np.unique(cell_str, return_inverse=True); cell_fam = np.array([fam[cell == k][0] for k in range(len(cell_names))])
    X = {"prompt": load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32), "core": np.load(PANEL_DIR / f"{REP_NAME}.npy").astype(np.float32)}
    U = {k: unit(v) for k, v in X.items()}; REPS = list(X)
    res = {"geometry": {}, "recovery": {}, "routing": {}, "difficulty": {}, "ci": {}}

    def agg(v, fte, mode): return float(v.mean()) if mode == "per_query" else float(np.mean([v[fte == g].mean() for g in range(F) if (fte == g).any()]))
    def boot(v, fte, mode, rng):
        if mode == "per_query": vals = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
        else:
            groups = [np.flatnonzero(fte == g) for g in range(F)]; groups = [g for g in groups if len(g)]
            vals = [np.mean([v[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    # ---------------- geometry (full panel): cosine structure, 10-NN family purity, K-means as in exp01 (K = 7, seeds 0-2), within-family K-means (exp04-style)
    rng0 = np.random.default_rng(0); samp = np.sort(rng0.choice(n, 3000, replace=False)); mm = np.flatnonzero(ds == "mmlu"); hs = np.flatnonzero(ds == "hellaswag")
    _, subj_idx = np.unique(subj[mm], return_inverse=True); _, src_idx = np.unique(src[hs], return_inverse=True)
    for k, u in U.items():
        S = u[samp] @ u.T; same = fam[samp][:, None] == fam[None, :]; S[np.arange(len(samp)), samp] = -2; same[np.arange(len(samp)), samp] = False
        nn10 = np.argpartition(-S, 10, axis=1)[:, :10]
        g = {"mean_cos_same_family": float(S[same].mean()), "mean_cos_other_family": float(S[~same].mean()), "nn10_same_family_share": float((fam[nn10] == fam[samp][:, None]).mean()),
             "nn10_same_family_share_by_family": {names[f_]: float((fam[nn10][fam[samp] == f_] == f_).mean()) for f_ in range(F)}, "kmeans7": []}
        for seed in (0, 1, 2):
            lab = KMeans(n_clusters=7, n_init=3, random_state=seed).fit_predict(u); pur, cont = purity(lab, fam)
            g["kmeans7"].append({"seed": seed, "nmi": float(normalized_mutual_info_score(fam, lab)), "ari": float(adjusted_rand_score(fam, lab)), "purity": pur, "contingency_rows_clusters_cols_families": cont.tolist()})
        lab = KMeans(n_clusters=57, n_init=3, random_state=0).fit_predict(u[mm]); g["kmeans57_mmlu_vs_subject"] = {"nmi": float(normalized_mutual_info_score(subj_idx, lab)), "purity": purity(lab, subj_idx)[0]}
        lab = KMeans(n_clusters=2, n_init=3, random_state=0).fit_predict(u[hs]); g["kmeans2_hellaswag_vs_source"] = {"nmi": float(normalized_mutual_info_score(src_idx, lab)), "purity": purity(lab, src_idx)[0]}
        res["geometry"][k] = g; print(f"geometry {k} done", flush=True)

    # ---------------- per split: recovery, difficulty, neighbours; routing under both cost conventions
    per = {k: {"cell_acc": [], "fam_acc": [], "mmlu_subj_acc": [], "hs_src_acc": [], "r2b": [], "r2b_fam": [], "r2b_knn40": [], "spear": []} for k in REPS}
    routing = {cc: {k: {} for k in REPS} for cc in COSTS}; ci = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte = fam[te]; idx = np.arange(len(te)); y = n_solved[te]; store = {}
        floor = np.array([n_solved[tr][fam[tr] == g_].mean() for g_ in range(F)])[fte]
        for k in REPS:
            x, u = X[k], U[k]; sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
            pc = LogisticRegression(C=1.0, max_iter=3000).fit(xtr, cell[tr]).predict(xte)
            per[k]["cell_acc"].append(float(np.mean(pc == cell[te]))); per[k]["fam_acc"].append(float(np.mean(cell_fam[pc] == fte)))
            per[k]["mmlu_subj_acc"].append(float(np.mean(pc[ds[te] == "mmlu"] == cell[te][ds[te] == "mmlu"]))); per[k]["hs_src_acc"].append(float(np.mean(pc[ds[te] == "hellaswag"] == cell[te][ds[te] == "hellaswag"])))
            pn = RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, n_solved[tr]).predict(xte)
            per[k]["r2b"].append(float(1 - ((y - pn) ** 2).sum() / ((y - floor) ** 2).sum())); per[k]["spear"].append(float(spearmanr(y, pn).correlation))
            per[k]["r2b_fam"].append({names[g_]: float(1 - ((y - pn)[fte == g_] ** 2).sum() / ((y - floor)[fte == g_] ** 2).sum()) for g_ in range(F)})
            nb = tr[NearestNeighbors(n_neighbors=max(KS), metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
            pk = n_solved[nb[:, :40]].mean(1); per[k]["r2b_knn40"].append(float(1 - ((y - pk) ** 2).sum() / ((y - floor) ** 2).sum())); store[k] = {"nb": nb, "pn": pn}
        if seed == SEEDS[0]:
            d_ = ((y - store["prompt"]["pn"]) ** 2) - ((y - store["core"]["pn"]) ** 2)
            ci["difficulty:ridge sse(prompt)-sse(core)|per_query"] = boot(d_, fte, "per_query", np.random.default_rng(0)); ci["difficulty:ridge sse(prompt)-sse(core)|per_family_equal_weight"] = boot(d_, fte, "per_family_equal_weight", np.random.default_rng(0))
        for cc, c in COSTS.items():
            for lam in LAMBDAS:
                Ul = r - lam * c; Ute, Utr = Ul[te], Ul[tr]; ftab = np.array([Utr[fam[tr] == g_].mean(0).argmax() for g_ in range(F)]); ch_fam = ftab[fte]; ch_cell = ch_fam.copy()
                for g_ in np.unique(cell[te]):
                    rows = cell[tr] == g_
                    if rows.sum() >= MIN_CELL: ch_cell[cell[te] == g_] = Utr[rows].mean(0).argmax()
                u_fam, u_cell, u_or = Ute[idx, ch_fam], Ute[idx, ch_cell], Ute.max(1)
                uk = {k: {kk: Ute[idx, Ul[store[k]["nb"][:, :kk]].mean(1).argmax(1)] for kk in KS} for k in REPS}
                for k in REPS:
                    d = routing[cc][k].setdefault(f"{lam:g}", {})
                    for mode in AGGS:
                        for kk in KS: d.setdefault(f"knn{kk}|{mode}", []).append(agg(uk[k][kk], fte, mode))
                        d.setdefault(f"family_lookup|{mode}", []).append(agg(u_fam, fte, mode)); d.setdefault(f"cell_lookup|{mode}", []).append(agg(u_cell, fte, mode)); d.setdefault(f"oracle|{mode}", []).append(agg(u_or, fte, mode))
                    for g_ in range(F): d.setdefault(f"knn100|family:{names[g_]}", []).append(float(uk[k][100][fte == g_].mean())); d.setdefault(f"cell_lookup|family:{names[g_]}", []).append(float(u_cell[fte == g_].mean()))
                if seed == SEEDS[0]:
                    for mode in AGGS:
                        ci[f"{cc}|{lam:g}|knn100:core-prompt|{mode}"] = boot(uk["core"][100] - uk["prompt"][100], fte, mode, np.random.default_rng(0))
                        ci[f"{cc}|{lam:g}|knn40:core-prompt|{mode}"] = boot(uk["core"][40] - uk["prompt"][40], fte, mode, np.random.default_rng(0))
                        ci[f"{cc}|{lam:g}|knn100:core-cell_lookup|{mode}"] = boot(uk["core"][100] - u_cell, fte, mode, np.random.default_rng(0))
        print(f"seed {seed} done", flush=True)

    # ---------------- aggregate + write
    def ms(v): return {"mean": float(np.mean(v)), "sd": float(np.std(v))}
    for k in REPS:
        res["recovery"][k] = {m: ms(per[k][m]) for m in ("cell_acc", "fam_acc", "mmlu_subj_acc", "hs_src_acc")}
        res["difficulty"][k] = {"ridge_r2_beyond_family": ms(per[k]["r2b"]), "knn40_r2_beyond_family": ms(per[k]["r2b_knn40"]), "ridge_spearman": ms(per[k]["spear"]),
                                "ridge_r2_beyond_family_per_family": {g_: float(np.mean([d[g_] for d in per[k]["r2b_fam"]])) for g_ in names}, "ridge_r2_beyond_family_equal_weight": float(np.mean([np.mean(list(d.values())) for d in per[k]["r2b_fam"]]))}
    res["routing"] = {cc: {k: {lam: {m: ms(v) for m, v in d.items()} for lam, d in routing[cc][k].items()} for k in REPS} for cc in COSTS}; res["ci"] = ci
    (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp26_deformatting", **res}, indent=1))
    md = ["# exp26 — deformatted ('core') vs templated ('prompt') GTE, in domain", "", "Same 34,661 rows, 5 stratified 70/30 splits, same decoders and metrics as exp01/exp08/exp14/exp18/exp20. Mean ± sd over splits; CI = bootstrap on split 0 (query-level / family-stratified). Costs: recorded (release) and corrected (template tokens × per-model input price, results/deformatting/cost_study.md).", ""]
    md += ["## Geometry (full panel; 3,000-row sample for cosines and 10-NN)", "", "rep | mean cos same family | other family | 10-NN same-family share | K-means-7 NMI / ARI / purity (mean of 3 seeds) | K-means-57 on MMLU: NMI / purity vs subject | K-means-2 on HellaSwag: NMI / purity vs source", "---|---|---|---|---|---|---"]
    for k, g in res["geometry"].items():
        km = g["kmeans7"]; md.append(f"{k} | {g['mean_cos_same_family']:.3f} | {g['mean_cos_other_family']:.3f} | {g['nn10_same_family_share']:.3f} | {np.mean([q['nmi'] for q in km]):.3f} / {np.mean([q['ari'] for q in km]):.3f} / {np.mean([q['purity'] for q in km]):.4f} | {g['kmeans57_mmlu_vs_subject']['nmi']:.3f} / {g['kmeans57_mmlu_vs_subject']['purity']:.3f} | {g['kmeans2_hellaswag_vs_source']['nmi']:.3f} / {g['kmeans2_hellaswag_vs_source']['purity']:.4f}")
    md += ["", "10-NN same-family share by family: " + "; ".join(f"{k}: " + ", ".join(f"{f_} {v:.2f}" for f_, v in g["nn10_same_family_share_by_family"].items()) for k, g in res["geometry"].items()), ""]
    for k, g in res["geometry"].items():
        md += [f"K-means-7 contingency, {k}, seed 0 (rows clusters, cols " + ", ".join(names) + "):", "```"] + [" ".join(f"{v:6d}" for v in row) for row in g["kmeans7"][0]["contingency_rows_clusters_cols_families"]] + ["```", ""]
    md += ["## Recovery of family and cell (exp20 classifier recipe)", "", "rep | cell accuracy (64) | implied family accuracy | MMLU subject accuracy | HellaSwag source accuracy", "---|---|---|---|---"]
    for k, g in res["recovery"].items(): md.append(f"{k} | {g['cell_acc']['mean']:.4f} ± {g['cell_acc']['sd']:.4f} | {g['fam_acc']['mean']:.4f} | {g['mmlu_subj_acc']['mean']:.4f} | {g['hs_src_acc']['mean']:.4f}")
    md += ["", "## Difficulty read-out (exp18 recipe, target n_solved)", "", "rep | ridge R²_beyond_family per query | equal weight | kNN-40 R²_beyond_family | Spearman | per family (ridge)", "---|---|---|---|---|---"]
    for k, g in res["difficulty"].items(): md.append(f"{k} | {g['ridge_r2_beyond_family']['mean']:+.3f} ± {g['ridge_r2_beyond_family']['sd']:.3f} | {g['ridge_r2_beyond_family_equal_weight']:+.3f} | {g['knn40_r2_beyond_family']['mean']:+.3f} | {g['ridge_spearman']['mean']:.3f} | " + ", ".join(f"{f_} {v:+.2f}" for f_, v in g["ridge_r2_beyond_family_per_family"].items()))
    md += ["", f"CI of per-query SSE(prompt) − SSE(core) for ridge (positive = core better): per query {ci['difficulty:ridge sse(prompt)-sse(core)|per_query']}, per family {ci['difficulty:ridge sse(prompt)-sse(core)|per_family_equal_weight']}", ""]
    for cc in COSTS:
        for mode in AGGS:
            md += [f"## Routing, cost = {cc}, {mode}", "", "λ | kNN-5 prompt / core | kNN-10 prompt / core | kNN-40 prompt / core | kNN-100 prompt / core | family lookup | cell lookup | oracle | kNN-100 core − prompt (CI) | kNN-40 core − prompt (CI) | kNN-100 core − cell lookup (CI)", "---|---|---|---|---|---|---|---|---|---|---"]
            for lam in LAMBDAS:
                L = f"{lam:g}"; P, C_ = res["routing"][cc]["prompt"][L], res["routing"][cc]["core"][L]
                row = [f"{P[f'knn{kk}|{mode}']['mean']:.4f} / {C_[f'knn{kk}|{mode}']['mean']:.4f}" for kk in KS]
                d1 = C_[f"knn100|{mode}"]["mean"] - P[f"knn100|{mode}"]["mean"]; d4 = C_[f"knn40|{mode}"]["mean"] - P[f"knn40|{mode}"]["mean"]; d2 = C_[f"knn100|{mode}"]["mean"] - P[f"cell_lookup|{mode}"]["mean"]
                md.append(f"{L} | " + " | ".join(row) + f" | {P[f'family_lookup|{mode}']['mean']:.4f} | {P[f'cell_lookup|{mode}']['mean']:.4f} | {P[f'oracle|{mode}']['mean']:.4f} | {d1:+.4f} {ci[f'{cc}|{L}|knn100:core-prompt|{mode}']} | {d4:+.4f} {ci[f'{cc}|{L}|knn40:core-prompt|{mode}']} | {d2:+.4f} {ci[f'{cc}|{L}|knn100:core-cell_lookup|{mode}']}")
            md.append("")
        md += [f"## Per family, cost = {cc}: kNN-100 core − prompt (mean over splits)", "", "λ | " + " | ".join(names), "---|" + "---|" * F]
        for lam in REPORT_LAMS:
            L = f"{lam:g}"; md.append(f"{L} | " + " | ".join(f"{res['routing'][cc]['core'][L][f'knn100|family:{g_}']['mean'] - res['routing'][cc]['prompt'][L][f'knn100|family:{g_}']['mean']:+.4f}" for g_ in names))
        md.append("")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
