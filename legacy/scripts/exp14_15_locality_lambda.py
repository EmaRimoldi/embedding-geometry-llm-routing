"""exp14/exp15: neighbour sets as routers under u_λ — kNN vs random same-cell vs oracle set, across λ.

Usage: exp14_15_locality_lambda.py --cell family|topic   (see experiments/exp14_*.yaml, exp15_*.yaml)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

KS = (5, 10, 40, 100); DRAWS = 5


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", choices=["family", "topic"], required=True); a = ap.parse_args()
    EXP = "exp14_locality_lambda_family" if a.cell == "family" else "exp15_locality_lambda_topic"
    panel = load_panel(PANEL); spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep); u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); n = panel.n; kmax = max(KS)
    if a.cell == "family":
        cell = fam.copy()
    else:
        cn = topic_str(panel, "record_prefix"); cnames = sorted(set(cn)); cell = np.array([cnames.index(s_) for s_ in cn])
    C = int(cell.max()) + 1
    dis_all = (r.max(1) - r.min(1)) > 1e-9

    per_seed = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr, te = split(n, fam, seed); tr = np.sort(tr); te = np.sort(te); fam_te = fam[te]; cell_tr, cell_te = cell[tr], cell[te]; dis = dis_all[te]
        # kNN neighbours (λ-independent)
        nb_knn = np.zeros((len(te), kmax), dtype=np.int64)
        for s0 in range(0, len(te), 2048):
            e0 = min(len(te), s0 + 2048); sims = u[te[s0:e0]] @ u[tr].T
            top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
            nb_knn[s0:e0] = tr[np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)]
        # random same-cell sets (λ-independent), capped
        nb_rand = np.zeros((DRAWS, len(te), kmax), dtype=np.int64); avail = np.zeros(len(te), dtype=np.int64)
        cell_rows = {g: tr[cell_tr == g] for g in range(C)}
        fam_rows = {g: tr[fam[tr] == g] for g in range(F)}
        nb_fill = np.zeros((DRAWS, len(te), kmax), dtype=np.int64)     # same-cell rows, remainder filled from the same family
        for g in range(C):
            trg = cell_rows[g]; teg = np.flatnonzero(cell_te == g)
            if len(teg) == 0 or len(trg) == 0: continue
            kk = min(kmax, len(trg)); avail[teg] = kk
            fg = fam[te[teg[0]]]; others = np.setdiff1d(fam_rows[fg], trg)
            for d in range(DRAWS):
                keys = rng.random((len(teg), len(trg)))
                pick = trg[np.argpartition(keys, kk - 1, axis=1)[:, :kk]] if kk < len(trg) else trg[np.argsort(keys, axis=1)]
                nb_rand[d, teg, :kk] = pick; nb_fill[d, teg, :kk] = pick
                if kk < kmax and len(others) >= kmax - kk:
                    keys2 = rng.random((len(teg), len(others)))
                    nb_fill[d, teg, kk:] = others[np.argpartition(keys2, kmax - kk - 1, axis=1)[:, :kmax - kk]]
                elif kk < kmax:
                    nb_fill[d, teg, kk:] = pick[:, :1]
        res = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]; Ute = U[te]; orc = U.argmax(1)      # oracle model per row
            table = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)])
            cell_table = np.array([Utr[cell_tr == g].mean(0).argmax() if (cell_tr == g).sum() >= 50 else table[fam[te[cell_te == g][0]]] if (cell_te == g).any() else 0 for g in range(C)])
            # oracle neighbour set: same-cell training rows whose oracle model equals the query's (capped, random subset if more than kmax)
            nb_best = np.zeros((len(te), kmax), dtype=np.int64); avail_best = np.zeros(len(te), dtype=np.int64)
            for g in range(C):
                trg = cell_rows[g]; teg = np.flatnonzero(cell_te == g)
                if len(teg) == 0 or len(trg) == 0: continue
                orc_tr = orc[trg]
                for m in np.unique(orc[te[teg]]):
                    q = teg[orc[te[teg]] == m]; cand = trg[orc_tr == m]
                    if len(cand) == 0:
                        cand = trg                                               # no same-oracle row in the cell: fall back to the cell
                    kk = min(kmax, len(cand)); avail_best[q] = kk
                    keys = rng.random((len(q), len(cand)))
                    nb_best[q, :kk] = cand[np.argpartition(keys, kk - 1, axis=1)[:, :kk]] if kk < len(cand) else cand[np.argsort(keys, axis=1)]
            def route_and_score(nb, cnt=None):
                out = {}
                for k in KS:
                    if cnt is None:
                        mean_u = U[nb[:, :k]].mean(1)
                        agree = (orc[nb[:, :k]] == orc[te][:, None]).mean(1)
                    else:
                        kk = np.minimum(k, cnt); m = np.arange(k)[None, :] < kk[:, None]
                        mean_u = (U[nb[:, :k]] * m[:, :, None]).sum(1) / kk[:, None]
                        agree = ((orc[nb[:, :k]] == orc[te][:, None]) * m).sum(1) / kk
                    ch = mean_u.argmax(1); real = Ute[np.arange(len(te)), ch]
                    out[k] = {"utility": real, "agree": agree}
                return out
            R = {"knn": route_and_score(nb_knn), "best_possible": route_and_score(nb_best, avail_best)}
            rand_runs = [route_and_score(nb_rand[d], avail) for d in range(DRAWS)]
            R["random_same_cell"] = {k: {key: np.mean([rr[k][key] for rr in rand_runs], axis=0) for key in ("utility", "agree")} for k in KS}
            fill_runs = [route_and_score(nb_fill[d]) for d in range(DRAWS)]
            R["random_cell_fill_family"] = {k: {key: np.mean([rr[k][key] for rr in fill_runs], axis=0) for key in ("utility", "agree")} for k in KS}
            look = Ute[np.arange(len(te)), table[fam_te]]; orac = Ute.max(1); clook = Ute[np.arange(len(te)), cell_table[cell_te]]
            res[str(lam)] = {}
            for k in KS:
                res[str(lam)][str(k)] = {}
                for subset, mask in (("all", np.ones(len(te), bool)), ("disagreement", dis)):
                    d_ = {}
                    for name in ("knn", "random_same_cell", "random_cell_fill_family", "best_possible"):
                        v = R[name][k]["utility"]; g_ = R[name][k]["agree"]
                        d_[name] = {"utility_per_query": float(v[mask].mean()), "utility_per_family": float(np.mean([v[mask & (fam_te == g)].mean() for g in range(F) if (mask & (fam_te == g)).sum()])),
                                    "decision_agreement": float(g_[mask].mean())}
                    d_["family_lookup"] = {"utility_per_query": float(look[mask].mean()), "utility_per_family": float(np.mean([look[mask & (fam_te == g)].mean() for g in range(F) if (mask & (fam_te == g)).sum()]))}
                    d_["cell_lookup"] = {"utility_per_query": float(clook[mask].mean()), "utility_per_family": float(np.mean([clook[mask & (fam_te == g)].mean() for g in range(F) if (mask & (fam_te == g)).sum()]))}
                    d_["oracle"] = {"utility_per_query": float(orac[mask].mean())}
                    res[str(lam)][str(k)][subset] = d_
        per_seed[seed] = res; print(f"seed {seed} done", flush=True)

    def agg(lam, k, subset, name, key):
        v = [per_seed[s][str(lam)][str(k)][subset][name][key] for s in SEEDS]; return float(np.mean(v)), float(min(v)), float(max(v))
    out = {**panel.provenance(), "experiment": EXP, "cell": a.cell, "n_cells": C, "ks": list(KS), "lambdas": LAMBDAS, "seeds": list(SEEDS),
           "results": {str(lam): {str(k): {subset: {name: {key: agg(lam, k, subset, name, key) for key in per_seed[SEEDS[0]][str(lam)][str(k)][subset][name]}
                                                      for name in per_seed[SEEDS[0]][str(lam)][str(k)][subset]} for subset in ("all", "disagreement")} for k in KS} for lam in LAMBDAS}}
    md = [f"# {EXP}", "", f"Reference cell: {a.cell} ({C} cells). 5 stratified 70/30 splits; neighbour sets from training rows; values = mean over splits (ranges in result.json).", ""]
    for subset in ("all", "disagreement"):
        for aggr in ("utility_per_query", "utility_per_family"):
            md += [f"## Realised utility, subset: {subset}, aggregation: **{aggr}**", "", "λ | k | kNN | random same-cell | random cell, family fill | best possible | family lookup | cell lookup | oracle", "---|---|---|---|---|---|---|---|---"]
            for lam in (0.0, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0):
                for k in KS:
                    row = out["results"][str(lam)][str(k)][subset]
                    md.append(f"{lam:g} | {k} | {row['knn'][aggr][0]:.4f} | {row['random_same_cell'][aggr][0]:.4f} | {row['random_cell_fill_family'][aggr][0]:.4f} | {row['best_possible'][aggr][0]:.4f} | {row['family_lookup'][aggr][0]:.4f} | {row['cell_lookup'][aggr][0]:.4f} | " + (f"{row['oracle']['utility_per_query'][0]:.4f}" if aggr == "utility_per_query" else ""))
            md.append("")
    md += ["## Decision agreement (fraction of the set sharing the query's oracle model), all rows, per query", "", "λ | k | kNN | random same-cell | best possible", "---|---|---|---|---"]
    for lam in (0.0, 10.0, 100.0, 1000.0):
        for k in KS:
            row = out["results"][str(lam)][str(k)]["all"]
            md.append(f"{lam:g} | {k} | {row['knn']['decision_agreement'][0]:.4f} | {row['random_same_cell']['decision_agreement'][0]:.4f} | {row['best_possible']['decision_agreement'][0]:.4f}")
    d = RESULTS_ROOT / "outcome_locality" / EXP; d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(out, indent=1)); (d / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
