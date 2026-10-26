"""exp29: is "solved by almost every model" a usable signal for routing to the cheapest model?
(a) Descriptive: among queries Mistral-7B solves, how many models solve them; P(Mistral solves | count); lift.
(b) Two-stage router: predict the solve count with the exp18 ridge on GTE (fit on the training fold), send the query to Mistral-7B if
    the prediction >= tau, else use the (family, topic) cell lookup; tau swept; utility on test rows at several lambdas, both aggregations,
    against the cell lookup, kNN-100, the same rule with the TRUE count (upper bound), and the oracle.
Writes results/recovering_difficulty/exp29_easy_to_mistral/."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
OUT = RESULTS_ROOT / "recovering_difficulty" / "exp29_easy_to_mistral"; LAMS = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0); TAUS = None; MIN_CELL = 50; N_BOOT = 1000; N_BOOT = 1000   # TAUS = (M-4 .. M), set in main from the number of models (7..11 on RouterBench); N_BOOT = 1000


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    models = [str(m).split("/")[-1] for m in panel.models]; mi = models.index(CHEAP_MODEL)
    global TAUS; M = len(models); TAUS = tuple(range(M - 4, M + 1)); hi = M - 2   # 'almost every model' = all but two (9 of 11 on RouterBench)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0); ns = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64); msolve = r[:, mi] >= SUCCESS_THRESHOLD
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    cell_str = topic_str(panel); _, cell = np.unique(cell_str, return_inverse=True)
    # (a) descriptive
    desc = {"P_mistral_solves_given_count": {int(k): float(msolve[ns == k].mean()) for k in range(M + 1)}, "count_distribution_given_mistral_solves": {int(k): float((ns[msolve] == k).mean()) for k in range(M + 1)},
            "almost_all_threshold": hi, "share_of_mistral_successes_with_count_ge_9": float((ns[msolve] >= hi).mean()), "share_of_queries_with_count_ge_9": float((ns >= hi).mean()), "P_mistral_solves": float(msolve.mean()),
            "P_mistral_solves_given_count_ge_9": float(msolve[ns >= hi].mean()), "P_mistral_solves_given_count_le_8": float(msolve[ns <= hi - 1].mean()),
            "per_family": {names[g]: {"P_mistral_solves": float(msolve[fam == g].mean()), "P_given_ge9": float(msolve[(fam == g) & (ns >= hi)].mean()) if ((fam == g) & (ns >= hi)).any() else None, "share_ge9": float((ns[fam == g] >= hi).mean())} for g in range(F)}}
    # (b) two-stage router
    def agg(v, fte, mode): return float(v.mean()) if mode == "per_query" else float(np.mean([v[fte == g].mean() for g in range(F) if (fte == g).any()]))
    def boot(v, fte, mode, rng):
        if mode == "per_query": vals = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
        else:
            groups = [np.flatnonzero(fte == g) for g in range(F)]; groups = [g for g in groups if len(g)]; vals = [np.mean([v[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    res = {}; ci = {}; prec = {t: [] for t in TAUS}; cov = {t: [] for t in TAUS}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte = fam[te]; idx = np.arange(len(te))
        sc = StandardScaler().fit(x[tr]); pn = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(x[tr]), ns[tr]).predict(sc.transform(x[te]))
        nb = tr[NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
        for t in TAUS: easy = pn >= t; cov[t].append(float(easy.mean())); prec[t].append(float(msolve[te][easy].mean()) if easy.any() else float("nan"))
        for lam in LAMS:
            Ul = r - lam * c; Utr, Ute = Ul[tr], Ul[te]; ftab = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)]); ch_cell = ftab[fte].copy()
            for g in np.unique(cell[te]):
                rows = cell[tr] == g
                if rows.sum() >= MIN_CELL: ch_cell[cell[te] == g] = Utr[rows].mean(0).argmax()
            u_cell = Ute[idx, ch_cell]; u_knn = Ute[idx, Ul[nb].mean(1).argmax(1)]; u_or = Ute.max(1); L = f"{lam:g}"; d = res.setdefault(L, {})
            for mode in ("per_query", "per_family_equal_weight"):
                d.setdefault(f"cell_lookup|{mode}", []).append(agg(u_cell, fte, mode)); d.setdefault(f"knn100|{mode}", []).append(agg(u_knn, fte, mode)); d.setdefault(f"oracle|{mode}", []).append(agg(u_or, fte, mode))
                for t in TAUS:
                    u_rule = np.where(pn >= t, Ute[:, mi], u_cell); u_true = np.where(ns[te] >= t, Ute[:, mi], u_cell)
                    d.setdefault(f"rule_pred_tau{t}|{mode}", []).append(agg(u_rule, fte, mode)); d.setdefault(f"rule_true_tau{t}|{mode}", []).append(agg(u_true, fte, mode))
                    if seed == SEEDS[0]: ci[f"{L}|rule_pred_tau{t}-cell|{mode}"] = boot(u_rule - u_cell, fte, mode, np.random.default_rng(0))
        print(f"seed {seed} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); ms = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
    out = {**panel.provenance(), "experiment": "exp29_easy_to_mistral", "descriptive": desc, "coverage_pred_ge_tau": {t: ms(v) for t, v in cov.items()}, "precision_mistral_solves_given_pred_ge_tau": {t: ms(v) for t, v in prec.items()},
           "routing": {L: {k: ms(v) for k, v in d.items()} for L, d in res.items()}, "ci_split0": ci}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# exp29 — is 'solved by almost every model' a usable signal to route to the cheap model ({CHEAP_MODEL})?", "", "## (a) Descriptive (whole panel)", "",
          f"P(Mistral solves) = {desc['P_mistral_solves']:.3f}. Among the queries Mistral solves, the share solved by >= {hi} of the {M} models is {desc['share_of_mistral_successes_with_count_ge_9']:.3f}; such queries are {desc['share_of_queries_with_count_ge_9']:.3f} of the panel. P(Mistral solves | count >= 9) = {desc['P_mistral_solves_given_count_ge_9']:.3f}; P(Mistral solves | count <= 8) = {desc['P_mistral_solves_given_count_le_8']:.3f}.", "",
          "count of models that solve the query | share of the panel | P(Mistral solves | count) | share of Mistral's successes", "---|---|---|---"]
    md += [f"{k} | {float((ns == k).mean()):.3f} | {desc['P_mistral_solves_given_count'][k]:.2f} | {desc['count_distribution_given_mistral_solves'][k]:.3f}" for k in range(M + 1)]
    md += ["", f"family | P({CHEAP_MODEL} solves) | share of rows with count >= {hi} | P({CHEAP_MODEL} solves | count >= {hi})", "---|---|---|---"] + [f"{g} | {v['P_mistral_solves']:.2f} | {v['share_ge9']:.2f} | {('%.2f' % v['P_given_ge9']) if v['P_given_ge9'] is not None else '–'}" for g, v in desc["per_family"].items()]
    md += ["", "## (b) Two-stage rule on proper splits: predicted count (exp18 ridge on GTE) >= tau -> Mistral-7B, else cell lookup", "",
           "tau | coverage (share of test rows sent to Mistral) | precision (Mistral solves them)", "---|---|---"] + [f"{t} | {cov[t] and np.mean(cov[t]):.3f} | {np.nanmean(prec[t]):.3f}" for t in TAUS]
    for mode in ("per_query", "per_family_equal_weight"):
        md += ["", f"### Utility, {mode} (mean over 5 splits; CI on split 0 for rule − cell lookup)", "", "λ | cell lookup | kNN-100 | oracle | " + " | ".join(f"rule τ={t} (pred) [CI vs cell]" for t in TAUS) + " | " + " | ".join(f"rule τ={t} (true count)" for t in TAUS), "---|---|---|---|" + "---|" * (2 * len(TAUS))]
        for lam in LAMS:
            L = f"{lam:g}"; d = out["routing"][L]
            md.append(f"{L} | {d[f'cell_lookup|{mode}']['mean']:.4f} | {d[f'knn100|{mode}']['mean']:.4f} | {d[f'oracle|{mode}']['mean']:.4f} | " + " | ".join(f"{d[f'rule_pred_tau{t}|{mode}']['mean']:.4f} [{ci[f'{L}|rule_pred_tau{t}-cell|{mode}'][0]:+.4f}, {ci[f'{L}|rule_pred_tau{t}-cell|{mode}'][1]:+.4f}]" for t in TAUS) + " | " + " | ".join(f"{d[f'rule_true_tau{t}|{mode}']['mean']:.4f}" for t in TAUS))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
