"""exp82 — Test A at lambda = 100: is "the lookup's model fails here" readable from the text?

At lambda=100 the cell lookup is no longer GPT-4 in five of seven families and the gap is spread
over tiers 2-4 (exp30 attribution), so the reference must be the lookup's ACTUAL choice per cell,
fitted on the training rows of each split (best mean training utility per (family, topic) cell,
family fallback below 50 rows, exactly the paper's cell lookup).
Two labels, both against that lookup:
  label 1 (reward gap)  positive = lookup's model fails and some model solves; negative = lookup's
                        model solves; excluded = nobody solves. Direct analogue of exp80.
  label 2 (any gap)     positive = oracle utility > lookup utility on this query; negative = equal.
Both labels are computed on training and test rows with the training-fitted lookup.
Ladder, probe, conditioning, bootstrap and null as exp80. Also records the realised gap and its
reward/cost split on the test rows as a check against exp30 (0.204 = 0.165 reward + 0.039 cost).
Writes results/oracle_gap/exp82_test_a_lam100/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C, lookups as L
from sklearn.metrics import roc_auc_score

EXP = "exp82_test_a_lam100"; LAM = 100.0; N_BOOT = 1000; N_PERM = 50; NULL_REPS = ("gte", "reader_l21")


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index()
    fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); S = C.solved_matrix(panel); anyone = S.any(1)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs.astype(np.float64), nan=0.0)
    U = L.utility(panel, LAM); u_o = U.max(1); m_o = U.argmax(1)
    _, cell = np.unique(np.array([str(t) for t in C.topic_str(panel)], dtype=object), return_inverse=True); n_cells = int(cell.max()) + 1
    reps = {rr: C.load_rep(rr, panel) for rr in C.REP_ORDER}
    print(f"reps loaded ({time.time()-t0:.0f}s)", flush=True)

    res = {lab: {rr: {"pooled": [], "eqw": [], "uncond": [], "per_family": [], "prec10": []} for rr in C.REP_ORDER} for lab in ("reward_gap", "any_gap")}
    gapchk = []; split0 = {}
    for seed in C.SEEDS:
        tr, te = C.split(n, fam, seed)
        fam_best, cell_best = L.fit_lookup(U[tr], cell[tr], fam[tr], F, n_cells)
        m_l = L.apply_lookup(fam, cell, fam_best, cell_best)
        u_l = U[np.arange(n), m_l]; r_l = r[np.arange(n), m_l]
        y1 = np.full(n, -1, dtype=np.int8); y1[r_l >= C.SUCCESS_THRESHOLD] = 0; y1[(r_l < C.SUCCESS_THRESHOLD) & anyone] = 1
        y2 = (u_o > u_l + 1e-9).astype(np.int8)
        # realised gap on the test rows, split into reward and cost parts
        g = u_o[te] - u_l[te]; rg = r[te, m_o[te]] - r_l[te]; cg = -LAM * (c[te, m_o[te]] - c[te, m_l[te]])
        gapchk.append({"gap": float(g.mean()), "reward_part": float(rg.mean()), "cost_part": float(cg.mean()),
                       "share_lookup_fails_some_solves": float((y1[te] == 1).mean()), "share_any_gap": float(y2[te].mean()),
                       "lookup_choice_by_family": {fnames[gg]: names[int(np.bincount(m_l[te][fam[te] == gg], minlength=len(names)).argmax())] for gg in range(F)}})
        for lab, y in (("reward_gap", y1), ("any_gap", y2)):
            keep = y >= 0; trk = tr[keep[tr]]; tek = te[keep[te]]
            for rr in C.REP_ORDER:
                s, _ = C.fit_scores(reps[rr][trk], y[trk], reps[rr][tek])
                p, e, per = C.family_auc(y[tek], s, fam[tek])
                R = res[lab][rr]; R["pooled"].append(p); R["eqw"].append(e); R["per_family"].append(per)
                R["uncond"].append(float(roc_auc_score(y[tek], s))); R["prec10"].append(C.precision_at(y[tek], s, 0.10))
                if seed == C.SEEDS[0]: split0[(lab, rr)] = (s, tek)
            if seed == C.SEEDS[0]: split0[(lab, "y")] = y
        print(f"seed {seed}: gap {gapchk[-1]['gap']:.4f} = reward {gapchk[-1]['reward_part']:.4f} + cost {gapchk[-1]['cost_part']:.4f}; " +
              "  ".join(f"{lab}:{rr} {res[lab][rr]['pooled'][-1]:.3f}" for lab in res for rr in ("gte", "reader_l21")) + f"  ({time.time()-t0:.0f}s)", flush=True)

    def ms(v): return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    def agg_family(lab, rr):
        return {fnames[gg]: (float(np.mean([pf.get(gg) for pf in res[lab][rr]["per_family"] if pf.get(gg) is not None])) if any(pf.get(gg) is not None for pf in res[lab][rr]["per_family"]) else None) for gg in range(F)}
    result = C.base(panel, {"experiment": EXP, "lambda": LAM, "gap_check_per_split": gapchk,
        "results": {lab: {rr: {"within_family_auc_pooled": ms(res[lab][rr]["pooled"]), "within_family_auc_eqw": ms(res[lab][rr]["eqw"]),
                               "unconditional_auc": ms(res[lab][rr]["uncond"]), "per_family_auc_mean_over_splits": agg_family(lab, rr),
                               "precision_at_top10pct": {"precision": ms([p[0] for p in res[lab][rr]["prec10"]]), "recall": ms([p[1] for p in res[lab][rr]["prec10"]]),
                                                         "base_rate": ms([p[2] for p in res[lab][rr]["prec10"]])}} for rr in C.REP_ORDER} for lab in res}})

    def md_main():
        g0 = gapchk[0]
        Lm = [f"# {EXP} — Test A at λ=100 against the paper's cell lookup ({panel.name})", "",
              f"Reference: the (family, topic) cell lookup fitted on each split's training rows at λ=100. Split-0 realised gap on test rows: {g0['gap']:.4f} = reward {g0['reward_part']:.4f} + cost {g0['cost_part']:.4f} (exp30: 0.204 = 0.165 + 0.039). Lookup's model fails and some model solves on {g0['share_lookup_fails_some_solves']:.3f} of test rows; any utility gap on {g0['share_any_gap']:.3f}.", "",
              "Split-0 lookup choice by family: " + ", ".join(f"{k} → {v}" for k, v in g0["lookup_choice_by_family"].items()), ""]
        for lab, title in (("reward_gap", "Label 1, reward gap: lookup's model fails and some model solves, vs lookup's model solves"),
                           ("any_gap", "Label 2, any gap: oracle utility exceeds the lookup's on this query")):
            Lm += [f"## {title}", "", "representation | within-family AUC, pooled | equal weight | unconditional | precision @ top 10% (recall) | base rate", "---|---|---|---|---|---"]
            for rr in C.REP_ORDER:
                R = result["results"][lab][rr]
                Lm.append(f"{C.REP_LABEL[rr]} | {R['within_family_auc_pooled']['mean']:.3f} [{R['within_family_auc_pooled']['min']:.3f}, {R['within_family_auc_pooled']['max']:.3f}] | {R['within_family_auc_eqw']['mean']:.3f} | {R['unconditional_auc']['mean']:.3f} | {R['precision_at_top10pct']['precision']['mean']:.3f} ({R['precision_at_top10pct']['recall']['mean']:.3f}) | {R['precision_at_top10pct']['base_rate']['mean']:.3f}")
            Lm += ["", "family | " + " | ".join(C.REP_ORDER), "---|" + "---|" * len(C.REP_ORDER)]
            Lm += [f"{fnames[gg]} | " + " | ".join(C.f3(result['results'][lab][rr]['per_family_auc_mean_over_splits'][fnames[gg]]) for rr in C.REP_ORDER) for gg in range(F)]
            Lm.append("")
        return Lm
    d = C.out_dir(EXP); C.save(d, result, md_main())

    result["split0_bootstrap"] = {}; result["split0_permutation_null"] = {}
    y1 = split0[("reward_gap", "y")]
    for rr in C.REP_ORDER:
        s, tek = split0[("reward_gap", rr)]
        ci_p, ci_e = C.bootstrap_family_auc(y1[tek], s, fam[tek], n=N_BOOT, seed=0)
        result["split0_bootstrap"][rr] = {"reward_gap_within_family_auc_pooled_ci95": ci_p, "eqw_ci95": ci_e}
        print(f"bootstrap {rr}: {ci_p}  ({time.time()-t0:.0f}s)", flush=True)
    md = md_main() + ["## Split 0 bootstrap, label 1 (95%, family-stratified)", "", "representation | pooled CI | equal-weight CI", "---|---|---"]
    md += [f"{C.REP_LABEL[rr]} | [{v['reward_gap_within_family_auc_pooled_ci95'][0]:.3f}, {v['reward_gap_within_family_auc_pooled_ci95'][1]:.3f}] | [{v['eqw_ci95'][0]:.3f}, {v['eqw_ci95'][1]:.3f}]" for rr, v in result["split0_bootstrap"].items()]
    C.save(d, result, md)
    tr0, _ = C.split(n, fam, C.SEEDS[0]); keep = y1 >= 0; tr0 = tr0[keep[tr0]]
    for rr in NULL_REPS:
        s, tek = split0[("reward_gap", rr)]
        nulls = C.permutation_null(reps[rr][tr0], y1[tr0], fam[tr0], reps[rr][tek], y1[tek], fam[tek], n=N_PERM, seed=0)
        obs = res["reward_gap"][rr]["pooled"][0]
        result["split0_permutation_null"][rr] = {"n_draws": N_PERM, "null_mean": float(np.mean(nulls)), "null_p97.5": float(np.percentile(nulls, 97.5)), "observed": obs, "p_value": float(np.mean(np.array(nulls) >= obs))}
        print(f"null {rr}: 97.5th {np.percentile(nulls,97.5):.3f} observed {obs:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    md += ["", f"## Split 0 permutation null, label 1 ({N_PERM} refitted draws)", "", "representation | null mean | null 97.5th | observed | p", "---|---|---|---|---"]
    md += [f"{C.REP_LABEL[rr]} | {v['null_mean']:.3f} | {v['null_p97.5']:.3f} | {v['observed']:.3f} | {v['p_value']:.3f}" for rr, v in result["split0_permutation_null"].items()]
    result["seconds"] = time.time() - t0; C.save(d, result, md)


if __name__ == "__main__":
    main()
