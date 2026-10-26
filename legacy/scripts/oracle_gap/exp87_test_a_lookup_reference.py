"""exp87 — Test A against the paper's cell lookup, difficulty held fixed, both panels, lam in {0, 100}.

Tier-free version of exp80/82/85 so it runs on LLMRouterBench too, where no single model dominates
(gemini-2.5-pro 0.597 vs gpt-5 0.592 panel solve rate). Reference = the (family, topic) cell lookup
fitted on each split's training rows at the given lambda (MIN_CELL=50, family fallback).
  positive  the lookup's model fails (reward < 0.5) and at least one other model solves  -> gap query
  negative  the lookup's model solves
  excluded  nobody solves
Strata: family, and (family, n_other) with n_other = number of solvers among the models OTHER than
the one the lookup chose for that query, so inside a stratum difficulty is fixed and only "was the
stored model among the solvers" is left. Ladder bow / gte / gte_l21 / reader_l21, ridge probe,
5 splits, split-0 stratified bootstrap and permutation null for gte and reader_l21.
PROVES on both benchmarks and both cost regimes whether the query text says when the stored model
fails once difficulty is removed.
Panel from E4_PANEL. Writes results/oracle_gap/exp87_test_a_lookup_reference/<panel>/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C, lookups as L
from exp85_test_a_lam0_difficulty_controlled import strat_bootstrap, strat_perm_null

EXP = "exp87_test_a_lookup_reference"; LAMS = (0.0, 100.0); N_BOOT = 1000; N_PERM = 30; NULL_REPS = ("gte", "reader_l21")


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index(); fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, _ = C.model_names(panel); S = C.solved_matrix(panel); anyone = S.any(1); n_solved = S.sum(1).astype(np.int64)
    _, cell = np.unique(np.array([str(t) for t in C.topic_str(panel)], dtype=object), return_inverse=True); n_cells = int(cell.max()) + 1
    reps = {rr: C.load_rep(rr, panel) for rr in C.REP_ORDER}; print(f"{C.PANEL}: reps loaded ({time.time()-t0:.0f}s)", flush=True)
    conds = ("family", "family_x_n_other")
    result = C.base(panel, {"experiment": EXP, "lambdas": list(LAMS), "min_cell": L.MIN_CELL, "by_lambda": {}})
    d = C.out_dir(EXP) / C.PANEL; d.mkdir(parents=True, exist_ok=True)
    md = [f"# {EXP} — Test A against the cell lookup, difficulty held fixed ({C.PANEL})", "",
          "Positive = the lookup's model fails and another model solves; negative = the lookup's model solves; lookup fitted on training rows per split. Strata: family, and (family, n_other), n_other = solvers among the models other than the stored one. AUC mean over 5 splits, pooled [equal weight over strata] (min, max).", ""]
    for lam in LAMS:
        U = L.utility(panel, lam); rows = np.arange(n)
        res = {rr: {c: {"pooled": [], "eqw": []} for c in conds} for rr in C.REP_ORDER}; split0 = {}; labinfo = []
        for seed in C.SEEDS:
            tr, te = C.split(n, fam, seed)
            fb, cb = L.fit_lookup(U[tr], cell[tr], fam[tr], F, n_cells); m_l = L.apply_lookup(fam, cell, fb, cb)
            s_l = S[rows, m_l]; y = np.full(n, -1, dtype=np.int8); y[s_l] = 0; y[(~s_l) & anyone] = 1; keep = y >= 0
            n_other = n_solved - s_l.astype(np.int64); st = {"family": fam.astype(np.int64), "family_x_n_other": fam * 100 + n_other}
            trk = tr[keep[tr]]; tek = te[keep[te]]
            labinfo.append({"seed": seed, "pos_test": int((y[tek] == 1).sum()), "neg_test": int((y[tek] == 0).sum()), "excluded_test": int((~keep[te]).sum()),
                            "lookup_choice_by_family": {fnames[g]: names[int(np.bincount(m_l[te][fam[te] == g], minlength=len(names)).argmax())] for g in range(F)}})
            for rr in C.REP_ORDER:
                s, _ = C.fit_scores(reps[rr][trk], y[trk], reps[rr][tek])
                for c in conds:
                    p, e, _ = C.strat_auc(y[tek], s, st[c][tek]); res[rr][c]["pooled"].append(p); res[rr][c]["eqw"].append(e)
                if seed == C.SEEDS[0]: split0[rr] = (s, tek, trk, y.copy(), st["family_x_n_other"].copy())
                print(f"lam {lam:g} seed {seed} {rr:10s} " + "  ".join(f"{c} {res[rr][c]['pooled'][-1]:.3f}/{res[rr][c]['eqw'][-1]:.3f}" for c in conds) + f"  ({time.time()-t0:.0f}s)", flush=True)
        def ms(v): return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
        entry = {"label_per_split": labinfo, "results": {rr: {c: {"pooled": ms(res[rr][c]["pooled"]), "eqw": ms(res[rr][c]["eqw"])} for c in conds} for rr in C.REP_ORDER}, "split0_bootstrap": {}, "split0_null": {}}
        li = labinfo[0]
        md += [f"## λ = {lam:g}", "", f"Split 0 test rows: {li['pos_test']} positive, {li['neg_test']} negative, {li['excluded_test']} excluded. Lookup by family: " + ", ".join(f"{k} → {v}" for k, v in li["lookup_choice_by_family"].items()), "",
               "representation | within family | within (family, n_other)", "---|---|---"]
        for rr in C.REP_ORDER:
            R = entry["results"][rr]; md.append(f"{C.REP_LABEL[rr]} | " + " | ".join(f"{R[c]['pooled']['mean']:.3f} [{R[c]['eqw']['mean']:.3f}] ({R[c]['pooled']['min']:.3f}, {R[c]['pooled']['max']:.3f})" for c in conds))
        result["by_lambda"][str(lam)] = entry; C.save(d, result, md)
        for rr in C.REP_ORDER:
            s, tek, trk, y, stno = split0[rr]; ci_p, ci_e = strat_bootstrap(y[tek], s, stno[tek], n=N_BOOT, seed=0)
            entry["split0_bootstrap"][rr] = {"family_x_n_other_pooled_ci95": ci_p, "eqw_ci95": ci_e}
        md += ["", "split 0 bootstrap (family, n_other), 95%: " + "; ".join(f"{C.REP_LABEL[rr]} [{v['family_x_n_other_pooled_ci95'][0]:.3f}, {v['family_x_n_other_pooled_ci95'][1]:.3f}]" for rr, v in entry["split0_bootstrap"].items())]
        C.save(d, result, md)
        for rr in NULL_REPS:
            s, tek, trk, y, stno = split0[rr]
            nulls = strat_perm_null(reps[rr][trk], y[trk], stno[trk], reps[rr][tek], y[tek], stno[tek], n=N_PERM, seed=0); obs = res[rr]["family_x_n_other"]["pooled"][0]
            entry["split0_null"][rr] = {"n_draws": len(nulls), "null_mean": float(np.mean(nulls)), "null_p97.5": float(np.percentile(nulls, 97.5)), "observed": obs, "p_value": float(np.mean(np.array(nulls) >= obs))}
            print(f"lam {lam:g} null {rr}: 97.5th {np.percentile(nulls, 97.5):.3f} observed {obs:.3f} ({time.time()-t0:.0f}s)", flush=True)
        md += ["split 0 permutation null (family, n_other): " + "; ".join(f"{C.REP_LABEL[rr]} null 97.5th {v['null_p97.5']:.3f} vs observed {v['observed']:.3f} (p={v['p_value']:.2f})" for rr, v in entry["split0_null"].items()), ""]
        C.save(d, result, md)
    result["seconds"] = time.time() - t0; C.save(d, result, md)


if __name__ == "__main__":
    main()
