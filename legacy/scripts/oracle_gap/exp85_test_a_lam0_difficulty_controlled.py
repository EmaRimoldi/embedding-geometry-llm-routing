"""exp85 — Test A at lambda = 0 with difficulty held fixed.

exp80 asks whether "GPT-4 fails and a tier-2 model solves" is readable from the text, within family.
That AUC can be inflated by difficulty: GPT-4 fails more often on queries that everybody fails, and
the reader hidden state reads the solve count (exp30 B, 0.677 for GPT-4). Stratifying by n_solved is
NOT clean, because GPT-4 contributes 1 to the count on every negative and 0 on every positive.
Here the stratum is (family, n_other) with n_other = number of solvers among the ten non-GPT-4 models.
Inside a stratum every query was solved by the same number of other models, and the only question
left is whether GPT-4 was among the solvers: that is the "which model" question with difficulty
removed. Same label as exp80 (positive = GPT-4 fails and >=1 tier-2 solves; negative = GPT-4 solves;
excluded = GPT-4 fails and no tier-2 solves), same 4-representation ladder, ridge probe and splits.
Reported: within-family AUC (replicates exp80), within (family, n_solved), within (family, n_other),
each pooled and equal-weight over strata; split-0 bootstrap and permutation null for the
(family, n_other) conditioning, gte and reader_l21.
PROVES whether the part of exp80's signal that survives is difficulty (drops to ~0.5 here) or a
genuine per-model idiosyncrasy readable from the text (stays above 0.5).
Writes results/oracle_gap/exp85_test_a_lam0_difficulty_controlled/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

EXP = "exp85_test_a_lam0_difficulty_controlled"; N_BOOT = 1000; N_PERM = 50; NULL_REPS = ("gte", "reader_l21")


def strat_bootstrap(y, s, strata, n=1000, seed=0):
    """Bootstrap of strat_auc (pooled, eqw) resampling rows within each stratum."""
    rng = np.random.default_rng(seed); u = np.unique(strata); idx_by = {k: np.flatnonzero(strata == k) for k in u}
    P, E = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in idx_by.values()])
        p, e, _ = C.strat_auc(y[idx], s[idx], strata[idx]); P.append(p); E.append(e)
    return [float(np.nanpercentile(P, 2.5)), float(np.nanpercentile(P, 97.5))], [float(np.nanpercentile(E, 2.5)), float(np.nanpercentile(E, 97.5))]


def strat_perm_null(xtr, ytr, sttr, xte, yte, stte, n=50, seed=0):
    """Labels shuffled inside each TRAIN stratum, probe refitted, strat_auc on test — the null for a probe
    that can read the stratum perfectly but nothing else."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        yp = ytr.copy()
        for k in np.unique(sttr):
            m = np.flatnonzero(sttr == k); yp[m] = yp[rng.permutation(m)]
        if yp.min() == yp.max(): continue
        s, _ = C.fit_scores(xtr, yp, xte); out.append(C.strat_auc(yte, s, stte)[0])
    return out


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index(); fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); tc = C.tier_columns(midx); g4 = tc[1][0]
    S = C.solved_matrix(panel); s4 = S[:, g4]; t2 = S[:, tc[2]].any(1)
    n_solved = S.sum(1).astype(np.int64); n_other = n_solved - s4.astype(np.int64)
    y = np.full(n, -1, dtype=np.int8); y[s4] = 0; y[(~s4) & t2] = 1; keep = y >= 0
    st_fam = fam.astype(np.int64); st_ns = fam * 100 + n_solved; st_no = fam * 100 + n_other
    print(f"label: pos={int((y==1).sum())} neg={int((y==0).sum())} excluded={int((~keep).sum())}", flush=True)
    # how the label sits inside n_other strata (is there anything left to predict?)
    comp = {}
    for k in range(0, 11):
        m = keep & (n_other == k)
        if m.sum(): comp[k] = {"rows": int(m.sum()), "pos_rate": float((y[m] == 1).mean())}
    reps = {rr: C.load_rep(rr, panel) for rr in C.REP_ORDER}
    print(f"reps loaded ({time.time()-t0:.0f}s)", flush=True)
    conds = {"family": st_fam, "family_x_n_solved": st_ns, "family_x_n_other": st_no}
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
    result = C.base(panel, {"experiment": EXP, "label": "pos: GPT-4 fails and >=1 tier-2 solves; neg: GPT-4 solves", "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
                            "label_by_n_other": comp, "results": {rr: {c: {"pooled": ms(res[rr][c]["pooled"]), "eqw": ms(res[rr][c]["eqw"])} for c in conds} for rr in C.REP_ORDER}})
    def md_main():
        L = [f"# {EXP} — Test A at λ=0 with difficulty held fixed ({panel.name})", "",
             "Same label as exp80. Strata: family (exp80's conditioning); (family, n_solved) — biased because GPT-4 is inside the count; (family, n_other) with n_other = solvers among the ten non-GPT-4 models — the clean one. AUC = mean over 5 splits, pooled [equal weight over strata].", "",
             "n_other | rows | positive rate", "---|---|---"] + [f"{k} | {v['rows']} | {v['pos_rate']:.3f}" for k, v in comp.items()] + ["",
             "representation | within family | within (family, n_solved) | within (family, n_other)", "---|---|---|---"]
        for rr in C.REP_ORDER:
            R = result["results"][rr]
            L.append(f"{C.REP_LABEL[rr]} | " + " | ".join(f"{R[c]['pooled']['mean']:.3f} [{R[c]['eqw']['mean']:.3f}] (min {R[c]['pooled']['min']:.3f}, max {R[c]['pooled']['max']:.3f})" for c in conds))
        return L
    d = C.out_dir(EXP); C.save(d, result, md_main())
    result["split0_bootstrap_family_x_n_other"] = {}; result["split0_null_family_x_n_other"] = {}
    for rr in C.REP_ORDER:
        s, tek, trk = split0[rr]; ci_p, ci_e = strat_bootstrap(y[tek], s, st_no[tek], n=N_BOOT, seed=0)
        result["split0_bootstrap_family_x_n_other"][rr] = {"pooled_ci95": ci_p, "eqw_ci95": ci_e}; print(f"bootstrap {rr}: {ci_p} ({time.time()-t0:.0f}s)", flush=True)
    md = md_main() + ["", "## Split 0 bootstrap, (family, n_other), 95%", "", "representation | pooled CI | equal-weight CI", "---|---|---"]
    md += [f"{C.REP_LABEL[rr]} | [{v['pooled_ci95'][0]:.3f}, {v['pooled_ci95'][1]:.3f}] | [{v['eqw_ci95'][0]:.3f}, {v['eqw_ci95'][1]:.3f}]" for rr, v in result["split0_bootstrap_family_x_n_other"].items()]
    C.save(d, result, md)
    for rr in NULL_REPS:
        s, tek, trk = split0[rr]
        nulls = strat_perm_null(reps[rr][trk], y[trk], st_no[trk], reps[rr][tek], y[tek], st_no[tek], n=N_PERM, seed=0)
        obs = res[rr]["family_x_n_other"]["pooled"][0]
        result["split0_null_family_x_n_other"][rr] = {"n_draws": len(nulls), "null_mean": float(np.mean(nulls)), "null_p97.5": float(np.percentile(nulls, 97.5)), "observed": obs, "p_value": float(np.mean(np.array(nulls) >= obs))}
        print(f"null {rr}: 97.5th {np.percentile(nulls, 97.5):.3f} observed {obs:.3f} ({time.time()-t0:.0f}s)", flush=True)
    md += ["", f"## Split 0 permutation null, (family, n_other) ({N_PERM} refitted draws, labels shuffled inside each training stratum)", "", "representation | null mean | null 97.5th | observed | p", "---|---|---|---|---"]
    md += [f"{C.REP_LABEL[rr]} | {v['null_mean']:.3f} | {v['null_p97.5']:.3f} | {v['observed']:.3f} | {v['p_value']:.3f}" for rr, v in result["split0_null_family_x_n_other"].items()]
    result["seconds"] = time.time() - t0; C.save(d, result, md)


if __name__ == "__main__":
    main()
