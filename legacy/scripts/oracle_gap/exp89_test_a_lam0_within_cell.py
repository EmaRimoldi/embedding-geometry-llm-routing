"""exp89 — is the residual "GPT-4 fails" signal (exp85) anything beyond the paper's own cell?

exp85 holds difficulty fixed with (family, n_other) strata and the signal survives (split 0: gte 0.630,
gte_l21 0.675). The cell lookup already conditions on (family, topic); if the surviving signal is
"which topics GPT-4 fails on", it lives inside the cell key and the lookup could only use it by
switching the whole cell away from GPT-4, which never pays at lambda=0 (exp84: cell lookup 0.7928 vs
family lookup 0.7933). Strata here: family; (family, n_other); cell = (family, topic); (cell, n_other).
Same label, ladder, probe and splits as exp80/85. Bootstrap and null (reader_l21, gte, 30 draws) on
split 0 for the (cell, n_other) conditioning.
PROVES whether there is any sub-cell text signal for GPT-4 failing at fixed difficulty, i.e.
anything the cell lookup does not already see.
Writes results/oracle_gap/exp89_test_a_lam0_within_cell/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp85_test_a_lam0_difficulty_controlled import strat_bootstrap, strat_perm_null

EXP = "exp89_test_a_lam0_within_cell"; N_BOOT = 1000; N_PERM = 30; NULL_REPS = ("gte", "reader_l21")


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index(); fnames = [str(s) for s in panel.datasets]
    names, midx = C.model_names(panel); tc = C.tier_columns(midx); g4 = tc[1][0]
    S = C.solved_matrix(panel); s4 = S[:, g4]; t2 = S[:, tc[2]].any(1); n_other = S.sum(1).astype(np.int64) - s4.astype(np.int64)
    y = np.full(n, -1, dtype=np.int8); y[s4] = 0; y[(~s4) & t2] = 1; keep = y >= 0
    _, cell = np.unique(np.array([str(t) for t in C.topic_str(panel)], dtype=object), return_inverse=True)
    conds = {"family": fam.astype(np.int64), "family_x_n_other": fam * 100 + n_other, "cell": cell.astype(np.int64), "cell_x_n_other": cell * 100 + n_other}
    strata_info = {c: {"n_strata_with_>=20_rows_and_both_classes": int(sum(1 for k in np.unique(st[keep]) if (st[keep] == k).sum() >= C.MIN_STRATUM and len(np.unique(y[keep][st[keep] == k])) == 2))} for c, st in conds.items()}
    print(strata_info, flush=True)
    reps = {rr: C.load_rep(rr, panel) for rr in C.REP_ORDER}; print(f"reps loaded ({time.time()-t0:.0f}s)", flush=True)
    res = {rr: {c: {"pooled": [], "eqw": []} for c in conds} for rr in C.REP_ORDER}; split0 = {}
    for seed in C.SEEDS:
        tr, te = C.split(n, fam, seed); trk = tr[keep[tr]]; tek = te[keep[te]]
        for rr in C.REP_ORDER:
            s, _ = C.fit_scores(reps[rr][trk], y[trk], reps[rr][tek])
            for c, st in conds.items():
                p, e, _ = C.strat_auc(y[tek], s, st[tek]); res[rr][c]["pooled"].append(p); res[rr][c]["eqw"].append(e)
            if seed == C.SEEDS[0]: split0[rr] = (s, tek, trk)
            print(f"seed {seed} {rr:10s} " + "  ".join(f"{c} {res[rr][c]['pooled'][-1]:.3f}/{res[rr][c]['eqw'][-1]:.3f}" for c in conds) + f"  ({time.time()-t0:.0f}s)", flush=True)
    def ms(v): return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    result = C.base(panel, {"experiment": EXP, "label": "pos: GPT-4 fails and >=1 tier-2 solves; neg: GPT-4 solves", "strata_info": strata_info,
                            "results": {rr: {c: {"pooled": ms(res[rr][c]["pooled"]), "eqw": ms(res[rr][c]["eqw"])} for c in conds} for rr in C.REP_ORDER}})
    def md_main():
        L = [f"# {EXP} — Test A at λ=0 inside the paper's cell ({panel.name})", "", "Same label as exp80/85. Strata add the (family, topic) cell used by the paper's lookup. AUC mean over 5 splits, pooled [equal weight over strata] (min, max). Strata with fewer than 20 rows or one class are dropped: " + ", ".join(f"{c}: {v['n_strata_with_>=20_rows_and_both_classes']}" for c, v in strata_info.items()) + " usable strata.", "",
             "representation | " + " | ".join(f"within {c}" for c in conds), "---|" + "---|" * len(conds)]
        for rr in C.REP_ORDER:
            R = result["results"][rr]; L.append(f"{C.REP_LABEL[rr]} | " + " | ".join(f"{R[c]['pooled']['mean']:.3f} [{R[c]['eqw']['mean']:.3f}] ({R[c]['pooled']['min']:.3f}, {R[c]['pooled']['max']:.3f})" for c in conds))
        return L
    d = C.out_dir(EXP); C.save(d, result, md_main())
    stc = conds["cell_x_n_other"]; result["split0_bootstrap_cell_x_n_other"] = {}; result["split0_null_cell_x_n_other"] = {}
    for rr in C.REP_ORDER:
        s, tek, trk = split0[rr]; ci_p, ci_e = strat_bootstrap(y[tek], s, stc[tek], n=N_BOOT, seed=0)
        result["split0_bootstrap_cell_x_n_other"][rr] = {"pooled_ci95": ci_p, "eqw_ci95": ci_e}; print(f"bootstrap {rr}: {ci_p} ({time.time()-t0:.0f}s)", flush=True)
    md = md_main() + ["", "split 0 bootstrap (cell, n_other), 95%: " + "; ".join(f"{C.REP_LABEL[rr]} [{v['pooled_ci95'][0]:.3f}, {v['pooled_ci95'][1]:.3f}]" for rr, v in result["split0_bootstrap_cell_x_n_other"].items())]
    C.save(d, result, md)
    for rr in NULL_REPS:
        s, tek, trk = split0[rr]; nulls = strat_perm_null(reps[rr][trk], y[trk], stc[trk], reps[rr][tek], y[tek], stc[tek], n=N_PERM, seed=0); obs = res[rr]["cell_x_n_other"]["pooled"][0]
        result["split0_null_cell_x_n_other"][rr] = {"n_draws": len(nulls), "null_mean": float(np.mean(nulls)), "null_p97.5": float(np.percentile(nulls, 97.5)), "observed": obs, "p_value": float(np.mean(np.array(nulls) >= obs))}
        print(f"null {rr}: 97.5th {np.percentile(nulls, 97.5):.3f} observed {obs:.3f} ({time.time()-t0:.0f}s)", flush=True)
    md += ["split 0 permutation null (cell, n_other), labels shuffled inside each training stratum: " + "; ".join(f"{C.REP_LABEL[rr]} null 97.5th {v['null_p97.5']:.3f} vs observed {v['observed']:.3f} (p={v['p_value']:.2f})" for rr, v in result["split0_null_cell_x_n_other"].items())]
    result["seconds"] = time.time() - t0; C.save(d, result, md)


if __name__ == "__main__":
    main()
