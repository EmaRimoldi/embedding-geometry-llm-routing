"""exp33: does a topic label help routing once EVERY family has one?  LLMRouterBench only.

Topic labels: the released ones (MMLU-Pro 14, ArenaHard 3) and the Qwen2.5-32B-Instruct labels
of llmrouterbench/topics_llm/ (every family, taxonomy per family, assigned from the raw question
only).  Cells = family x topic, family fallback below MIN_CELL training rows.  Five stratified
70/30 splits; both aggregations; paired bootstraps on split 0; within-family permutation null.
A. lookups: family / released cell / LLM cell / LLM cell predicted from GTE, kNN-100, random
   same-LLM-cell rows, oracle, per lambda.
B. permutation test of the LLM-cell gain over the family lookup, pooled and per family.
C. topic ceiling: sum over cells of share x (cell-best reward - family-best model's reward in
   the cell), with whole-panel winners (a ceiling, not a router), released vs LLM cells.
D. train-selected isolation: cells whose training-fold best model differs from the family's;
   LLM-cell gain on their test rows only, with the permutation null.
E. recoverability: multinomial logistic regression GTE -> LLM cell.
Writes llmrouterbench/results/performance-cost/outcome_locality/exp33_llm_topics/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, topic_str  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp33_llm_topics"
OUT = RESULTS_ROOT / "outcome_locality" / EXP
LAMS = (0.0, 1.0, 10.0, 100.0, 1000.0)
MIN_CELL = 50
N_PERM_PER_SEED = 20          # 20 x 5 splits = 100 permutation draws
N_BOOT = 1000
AGGS = ("per_query", "per_family_equal_weight")


def lookup(U, tr, te, fam, cell, F):
    """Best model per cell on the training rows, family fallback below MIN_CELL rows."""
    ftab = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)])
    ch = ftab[fam[te]].copy()
    for g in np.unique(cell[te]):
        rows = cell[tr] == g
        if rows.sum() >= MIN_CELL:
            ch[cell[te] == g] = U[tr][rows].mean(0).argmax()
    return ch


def agg(v, fte, F, mode):
    if mode == "per_query":
        return float(v.mean())
    return float(np.mean([v[fte == g].mean() for g in range(F) if (fte == g).any()]))


def boot(d, fte, F, rng):
    q = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    groups = [np.flatnonzero(fte == g) for g in range(F)]
    groups = [g for g in groups if len(g)]
    f = [np.mean([d[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
    return {"per_query": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))],
            "per_family_equal_weight": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]}


def permute_within_family(cell, fam, F, rng):
    pc = cell.copy()
    for g in range(F):
        rows = np.flatnonzero(fam == g)
        pc[rows] = cell[rows][rng.permutation(len(rows))]
    return pc


def main() -> None:
    panel = load_panel(PANEL)
    n = panel.n; fam = panel.dataset_index(); names = panel.datasets; F = len(names)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    c = np.nan_to_num(panel.costs, nan=0.0)
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    T = json.load(open(ANALYSIS_ROOT / "llmrouterbench" / "topics_llm" / f"{PANEL}.json"))
    llm_topic = np.array(T["topic"], dtype=object)
    if panel.keep is not None and T["rows"] == len(panel.keep):
        llm_topic = llm_topic[panel.keep]          # labels built on the full release; row-filtered panel (minilm/)
    assert len(llm_topic) == n
    labels = {"released": np.array([str(t) for t in topic_str(panel)], dtype=object),
              "llm": llm_topic}
    cells = {k: np.unique(v, return_inverse=True)[1] for k, v in labels.items()}
    has_topic = {k: np.array(["/" in str(t) for t in v]) for k, v in labels.items()}
    llm_cell_fam = np.array([fam[cells["llm"] == k_][0] for k_ in range(cells["llm"].max() + 1)])

    # ---- C. topic ceiling (whole-panel winners)
    ceiling = {}
    for k, cell in cells.items():
        ceiling[k] = {}
        for lam in (0.0, 10.0, 100.0):
            U = r - lam * c
            per_fam = {}
            for g in range(F):
                rows = np.flatnonzero(fam == g)
                fb = int(U[rows].mean(0).argmax())
                gain = 0.0
                for cl in np.unique(cell[rows]):
                    cr = rows[cell[rows] == cl]
                    m = U[cr].mean(0)
                    gain += len(cr) / len(rows) * (m.max() - m[fb])
                per_fam[names[g]] = float(gain)
            ceiling[k][f"{lam:g}"] = {"per_family": per_fam,
                                      "per_query": float(sum(per_fam[names[g]] * (fam == g).mean() for g in range(F))),
                                      "per_family_equal_weight": float(np.mean(list(per_fam.values())))}

    res, ci = {}, {}
    scopes = ["all"] + list(names)
    obs = {k: {f"{lam:g}": {mode: {sc: [] for sc in scopes} for mode in AGGS} for lam in LAMS} for k in cells}
    perm = {k: {f"{lam:g}": {mode: {sc: [] for sc in scopes} for mode in AGGS} for lam in LAMS} for k in cells}
    iso = {f"{lam:g}": {"gain": [], "share": [], "null": []} for lam in LAMS}
    recov = {"cell_accuracy": [], "implied_family_accuracy": [], "per_family_accuracy": []}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te)
        fte = fam[te]; idx = np.arange(len(te))
        # kNN-100 neighbours among training rows
        nb = np.zeros((len(te), 100), dtype=np.int64)
        for s in range(0, len(te), 2048):
            sims = u[te[s:s + 2048]] @ u[tr].T
            top = np.argpartition(-sims, 100, axis=1)[:, :100]
            nb[s:s + 2048] = tr[top]
        # E. recoverability of the LLM cell from GTE
        sc_ = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc_.transform(x[tr]), cells["llm"][tr])
        pred = clf.predict(sc_.transform(x[te]))
        recov["cell_accuracy"].append(float((pred == cells["llm"][te]).mean()))
        recov["implied_family_accuracy"].append(float((llm_cell_fam[pred] == fte).mean()))
        recov["per_family_accuracy"].append({names[g]: float((pred[fte == g] == cells["llm"][te][fte == g]).mean()) for g in range(F)})
        # random 100 training rows of the same LLM cell (family pool below MIN_CELL)
        rand_nb = np.zeros((len(te), 100), dtype=np.int64)
        for cl in np.unique(cells["llm"][te]):
            rows_te = np.flatnonzero(cells["llm"][te] == cl)
            pool = tr[cells["llm"][tr] == cl]
            if len(pool) < MIN_CELL:
                pool = tr[fam[tr] == fte[rows_te[0]]]
            rand_nb[rows_te] = rng.choice(pool, size=(len(rows_te), 100), replace=True)
        # permuted label sets, drawn once per seed and reused at every lambda
        perms = {k: [permute_within_family(cells[k], fam, F, rng) for _ in range(N_PERM_PER_SEED)] for k in cells}
        for lam in LAMS:
            L = f"{lam:g}"; U = r - lam * c; Ute = U[te]
            ftab = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)])
            ch = {"family": ftab[fte]}
            for k, cell in cells.items():
                ch[k] = lookup(U, tr, te, fam, cell, F)
            tab = {}
            for cl in np.unique(cells["llm"][tr]):
                rows = cells["llm"][tr] == cl
                if rows.sum() >= MIN_CELL:
                    tab[cl] = int(U[tr][rows].mean(0).argmax())
            ch["llm_predicted"] = np.array([tab.get(int(p_), int(ftab[llm_cell_fam[p_]])) for p_ in pred])
            util = {k: Ute[idx, v] for k, v in ch.items()}
            util["knn100"] = Ute[idx, U[nb].mean(1).argmax(1)]
            util["random_same_llm_cell_100"] = Ute[idx, U[rand_nb].mean(1).argmax(1)]
            util["oracle"] = Ute.max(1)
            for k, v in util.items():
                for mode in AGGS:
                    res.setdefault(L, {}).setdefault(f"{k}|{mode}", []).append(agg(v, fte, F, mode))
            if seed == SEEDS[0]:
                for k in ("released", "llm", "llm_predicted", "knn100"):
                    ci[f"{L}|{k}-family"] = boot(util[k] - util["family"], fte, F, np.random.default_rng(0))
                ci[f"{L}|llm-knn100"] = boot(util["llm"] - util["knn100"], fte, F, np.random.default_rng(0))
                ci[f"{L}|llm-released"] = boot(util["llm"] - util["released"], fte, F, np.random.default_rng(0))
            # B. observed gains and permutation null, both label sets
            for k, cell in cells.items():
                d = util[k] - util["family"]
                for mode in AGGS:
                    obs[k][L][mode]["all"].append(agg(d, fte, F, mode))
                    for g in range(F):
                        obs[k][L][mode][names[g]].append(float(d[fte == g].mean()))
                for pc in perms[k]:
                    dp = Ute[idx, lookup(U, tr, te, fam, pc, F)] - util["family"]
                    for mode in AGGS:
                        perm[k][L][mode]["all"].append(agg(dp, fte, F, mode))
                        for g in range(F):
                            perm[k][L][mode][names[g]].append(float(dp[fte == g].mean()))
            # D. train-selected isolation on the LLM cells
            def selected(cell_arr):
                sel = np.zeros(len(te), bool)
                for cl in np.unique(cell_arr[tr]):
                    rows = cell_arr[tr] == cl
                    if rows.sum() >= MIN_CELL and U[tr][rows].mean(0).argmax() != ftab[fam[tr][rows][0]]:
                        sel |= cell_arr[te] == cl
                return sel
            sel = selected(cells["llm"])
            iso[L]["share"].append(float(sel.mean()))
            iso[L]["gain"].append(float((util["llm"] - util["family"])[sel].mean()) if sel.any() else float("nan"))
            for pc in perms["llm"]:
                selp = selected(pc)
                dp = Ute[idx, lookup(U, tr, te, fam, pc, F)] - util["family"]
                iso[L]["null"].append(float(dp[selp].mean()) if selp.any() else 0.0)
        print(f"seed {seed} done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    ms = lambda v: float(np.mean(v))  # noqa: E731
    B = {k: {L: {mode: {sc: {"observed": ms(obs[k][L][mode][sc]), "null_mean": ms(perm[k][L][mode][sc]),
                             "null_97.5": float(np.percentile(perm[k][L][mode][sc], 97.5)),
                             "p": float(np.mean(np.array(perm[k][L][mode][sc]) >= ms(obs[k][L][mode][sc])))}
                        for sc in scopes} for mode in AGGS} for L in obs[k]} for k in cells}
    D = {L: {"share_of_test_rows": ms(v["share"]), "gain_on_selected": float(np.nanmean(v["gain"])), "null_mean": ms(v["null"]),
             "null_97.5": float(np.percentile(v["null"], 97.5)), "p": float(np.mean(np.array(v["null"]) >= np.nanmean(v["gain"])))}
         for L, v in iso.items()}
    E = {"cell_accuracy": ms(recov["cell_accuracy"]), "implied_family_accuracy": ms(recov["implied_family_accuracy"]),
         "per_family": {g: ms([d[g] for d in recov["per_family_accuracy"]]) for g in names}}
    out = {**panel.provenance(), "experiment": EXP,
           "labels": {k: {"n_cells": int(cells[k].max() + 1), "rows_with_topic": int(has_topic[k].sum())} for k in cells},
           "llm_labeller": T["labeller"], "llm_agreement_with_released": T["agreement_with_released"], "llm_label_counts": T["counts"],
           "A_utility": {L: {k: ms(v) for k, v in d.items()} for L, d in res.items()}, "A_ci_split0": ci,
           "B_permutation": B, "C_ceiling": ceiling, "D_isolation": D, "E_recoverability": E}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))

    md = [f"# {EXP} — topic cells from an LLM labeller on every family", "",
          f"Labeller: {Path(T['labeller']).name} on origin_query; {out['labels']['llm']['n_cells']} LLM cells "
          f"({out['labels']['llm']['rows_with_topic']} rows labelled) vs {out['labels']['released']['n_cells']} released cells "
          f"({out['labels']['released']['rows_with_topic']} rows). Agreement of the LLM with the released labels: "
          + ", ".join(f"{k} {v['agree']:.3f}" for k, v in T["agreement_with_released"].items())
          + f". 5 splits; CI = paired bootstrap, split 0; null = within-family permutation of the labels ({N_PERM_PER_SEED * len(SEEDS)} draws).", ""]
    for mode in AGGS:
        md += [f"## A. Utility, {mode}", "",
               "λ | family | released cell | LLM cell | LLM cell (predicted from GTE) | kNN-100 | random same-LLM-cell 100 | oracle | LLM − family [CI] | LLM − released [CI] | LLM − kNN-100 [CI]",
               "---|---|---|---|---|---|---|---|---|---|---"]
        for lam in LAMS:
            L = f"{lam:g}"; a_ = out["A_utility"][L]
            f_ = lambda k: a_[f"{k}|{mode}"]  # noqa: E731
            md.append(f"{L} | {f_('family'):.4f} | {f_('released'):.4f} | {f_('llm'):.4f} | {f_('llm_predicted'):.4f} | {f_('knn100'):.4f} | "
                      f"{f_('random_same_llm_cell_100'):.4f} | {f_('oracle'):.4f} | "
                      f"{f_('llm') - f_('family'):+.4f} [{ci[f'{L}|llm-family'][mode][0]:+.4f}, {ci[f'{L}|llm-family'][mode][1]:+.4f}] | "
                      f"{f_('llm') - f_('released'):+.4f} [{ci[f'{L}|llm-released'][mode][0]:+.4f}, {ci[f'{L}|llm-released'][mode][1]:+.4f}] | "
                      f"{f_('llm') - f_('knn100'):+.4f} [{ci[f'{L}|llm-knn100'][mode][0]:+.4f}, {ci[f'{L}|llm-knn100'][mode][1]:+.4f}]")
        md.append("")
    md += ["## B. Permutation test of the LLM-cell gain over the family lookup (per query)", "",
           "λ | scope | observed | null mean | null 97.5th | p", "---|---|---|---|---|---"]
    for lam in LAMS:
        L = f"{lam:g}"
        for sc in scopes:
            b = B["llm"][L]["per_query"][sc]
            md.append(f"{L} | {sc} | {b['observed']:+.4f} | {b['null_mean']:+.4f} | {b['null_97.5']:+.4f} | {b['p']:.3f}")
    md += ["", "## C. Topic ceiling (whole-panel winners; share-weighted gain of the cell-best over the family-best model)", "",
           "λ | labels | per query | equal weight | per family (non-zero)", "---|---|---|---|---"]
    for k in cells:
        for L, v in ceiling[k].items():
            md.append(f"{L} | {k} | {v['per_query']:+.4f} | {v['per_family_equal_weight']:+.4f} | "
                      + ", ".join(f"{g} {x_:+.3f}" for g, x_ in v["per_family"].items() if abs(x_) > 0.0005))
    md += ["", "## D. Train-selected isolation: LLM cells whose training-fold winner differs from the family winner", "",
           "λ | share of test rows in such cells | LLM-cell gain on them | null mean | null 97.5th | p", "---|---|---|---|---|---"]
    for L, v in D.items():
        md.append(f"{L} | {v['share_of_test_rows']:.3f} | {v['gain_on_selected']:+.4f} | {v['null_mean']:+.4f} | {v['null_97.5']:+.4f} | {v['p']:.3f}")
    md += ["", f"## E. Recoverability of the LLM cell from GTE: cell accuracy {E['cell_accuracy']:.4f}, implied family accuracy "
           f"{E['implied_family_accuracy']:.4f}; per family: " + ", ".join(f"{g} {v:.3f}" for g, v in E["per_family"].items())]
    (OUT / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
