"""exp88 — does the "GPT-4 fails" signal (exp80) buy any utility once it has to route? lambda = 0.

exp84: a lookup keyed by (family, true tier of the oracle's model) scores 0.866 vs the cell lookup's
0.793 at lambda=0, i.e. 58% of the gap, and at lambda=0 "tier >= 2" is exactly "GPT-4 fails". exp80
measures how readable that label is (AUC). Here the probe score is used to route:
   route to the family's best tier-2 fallback if score > tau, else GPT-4.
tau is chosen on a 20% calibration slice of the TRAINING rows (probe fitted on the other 80%), never
on test. The fallback per family = tier-2 model with the highest mean training reward on training
rows where GPT-4 fails (the population the switch is meant for). Reported on the test rows: mean
reward per query [equal weight over families] for GPT-4 always, the cell lookup, the probe router
for each representation, the test-optimal tau (an upper bound no router can reach), the
(family, true tier) lookup, and the oracle; and share of the gap captured.
PROVES whether an AUC of 0.6-0.75 for "GPT-4 fails" is worth anything in the units the paper uses.
Writes results/oracle_gap/exp88_probe_to_router_lam0/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C, lookups as L

EXP = "exp88_probe_to_router_lam0"; N_TAU = 60; CAL_FRAC = 0.2


def agg(v, fam, F):
    return float(v.mean()), float(np.mean([v[fam == g].mean() for g in range(F) if (fam == g).any()]))


def route_reward(score, tau, r_g4, r_fb):
    return np.where(score > tau, r_fb, r_g4)


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index(); fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); tc = C.tier_columns(midx); g4 = tc[1][0]; t2cols = tc[2]
    S = C.solved_matrix(panel); s4 = S[:, g4]; t2 = S[:, t2cols].any(1)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); r_g4 = r[:, g4]; r_or = r.max(1)
    y = np.full(n, -1, dtype=np.int8); y[s4] = 0; y[(~s4) & t2] = 1
    _, cell = np.unique(np.array([str(t) for t in C.topic_str(panel)], dtype=object), return_inverse=True); n_cells = int(cell.max()) + 1
    M = len(names); tier_of = np.zeros(M, dtype=np.int64)
    for t, cols in tc.items():
        for cc in cols: tier_of[cc] = t
    is_max = r >= r_or[:, None] - 1e-12; tier_strong = np.where(is_max, tier_of[None, :], 99).min(1)
    reps = {rr: C.load_rep(rr, panel) for rr in C.REP_ORDER}; print(f"reps loaded ({time.time()-t0:.0f}s)", flush=True)
    per_split = []
    for seed in C.SEEDS:
        tr, te = C.split(n, fam, seed); rng = np.random.default_rng(1000 + seed)
        # calibration slice of the training rows, stratified by family
        cal_mask = np.zeros(n, dtype=bool)
        for g in range(F):
            idx = tr[fam[tr] == g]; cal_mask[rng.choice(idx, size=int(round(CAL_FRAC * len(idx))), replace=False)] = True
        fit = tr[~cal_mask[tr]]; cal = tr[cal_mask[tr]]
        # fallback per family: best tier-2 model on training rows where GPT-4 fails
        fb = np.zeros(F, dtype=np.int64)
        for g in range(F):
            m = (fam[tr] == g) & (~s4[tr]); pop = tr[m] if m.sum() >= 20 else tr[fam[tr] == g]
            fb[g] = t2cols[int(r[pop][:, t2cols].mean(0).argmax())]
        r_fb = r[np.arange(n), fb[fam]]
        # references
        cb_f, cb_c = L.fit_lookup(r[tr], cell[tr], fam[tr], F, n_cells); m_cell = L.apply_lookup(fam, cell, cb_f, cb_c)
        k_ft = L.combine_keys(fam, tier_strong); kb_f, kb = L.fit_lookup(r[tr], k_ft[tr], fam[tr], F, int(k_ft.max()) + 1); m_ft = L.apply_lookup(fam, k_ft, kb_f, kb)
        row = {"seed": seed, "fallback_by_family": {fnames[g]: names[fb[g]] for g in range(F)},
               "gpt4_always": agg(r_g4[te], fam[te], F), "cell_lookup": agg(r[te, m_cell[te]], fam[te], F),
               "family_true_tier": agg(r[te, m_ft[te]], fam[te], F), "oracle": agg(r_or[te], fam[te], F),
               "fallback_always": agg(r_fb[te], fam[te], F), "probe": {}}
        fitk = fit[y[fit] >= 0]
        for rr in C.REP_ORDER:
            s_all, alpha = C.fit_scores(reps[rr][fitk], y[fitk], np.vstack([reps[rr][cal], reps[rr][te]]))
            s_cal, s_te = s_all[:len(cal)], s_all[len(cal):]
            taus = np.quantile(s_cal, np.linspace(0.0, 1.0, N_TAU))
            cal_pq = [route_reward(s_cal, t, r_g4[cal], r_fb[cal]).mean() for t in taus]
            cal_eq = [agg(route_reward(s_cal, t, r_g4[cal], r_fb[cal]), fam[cal], F)[1] for t in taus]
            tau_pq = taus[int(np.argmax(cal_pq))]; tau_eq = taus[int(np.argmax(cal_eq))]
            te_pq = [route_reward(s_te, t, r_g4[te], r_fb[te]).mean() for t in taus]
            te_eq = [agg(route_reward(s_te, t, r_g4[te], r_fb[te]), fam[te], F)[1] for t in taus]
            row["probe"][rr] = {"alpha": float(alpha),
                                "router": (float(route_reward(s_te, tau_pq, r_g4[te], r_fb[te]).mean()), agg(route_reward(s_te, tau_eq, r_g4[te], r_fb[te]), fam[te], F)[1]),
                                "share_routed_to_fallback": float((s_te > tau_pq).mean()),
                                "test_optimal_tau_upper_bound": (float(max(te_pq)), float(max(te_eq)))}
            print(f"seed {seed} {rr:10s} router {row['probe'][rr]['router'][0]:.4f} [{row['probe'][rr]['router'][1]:.4f}]  routed {row['probe'][rr]['share_routed_to_fallback']:.3f}  test-opt {max(te_pq):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        per_split.append(row)
        print(f"seed {seed}: gpt4 {row['gpt4_always'][0]:.4f} cell {row['cell_lookup'][0]:.4f} true-tier {row['family_true_tier'][0]:.4f} oracle {row['oracle'][0]:.4f}", flush=True)
    def mean2(key, sub=None):
        vals = [(r_[sub][key] if sub else r_[key]) for r_ in per_split]; return (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
    cl = mean2("cell_lookup"); orc = mean2("oracle")
    summary = {"gpt4_always": mean2("gpt4_always"), "cell_lookup": cl, "fallback_always": mean2("fallback_always"), "family_true_tier": mean2("family_true_tier"), "oracle": orc, "probe": {}}
    for rr in C.REP_ORDER:
        rt = (float(np.mean([r_["probe"][rr]["router"][0] for r_ in per_split])), float(np.mean([r_["probe"][rr]["router"][1] for r_ in per_split])))
        ub = (float(np.mean([r_["probe"][rr]["test_optimal_tau_upper_bound"][0] for r_ in per_split])), float(np.mean([r_["probe"][rr]["test_optimal_tau_upper_bound"][1] for r_ in per_split])))
        summary["probe"][rr] = {"router": rt, "test_optimal_tau_upper_bound": ub, "share_routed_to_fallback": float(np.mean([r_["probe"][rr]["share_routed_to_fallback"] for r_ in per_split])),
                                "gain_over_cell": (rt[0] - cl[0], rt[1] - cl[1]), "share_of_gap_captured": ((rt[0] - cl[0]) / (orc[0] - cl[0]), (rt[1] - cl[1]) / (orc[1] - cl[1]))}
    result = C.base(panel, {"experiment": EXP, "lambda": 0.0, "cal_frac": CAL_FRAC, "n_tau": N_TAU, "summary": summary, "per_split": per_split})
    md = [f"# {EXP} — routing on the 'GPT-4 fails' probe score, λ=0 ({panel.name})", "",
          "Route to the family's tier-2 fallback if probe score > τ, else GPT-4; τ chosen on a 20% calibration slice of the training rows. Mean reward on test rows over 5 splits, per query [equal weight over families]. 'captured' = (router − cell lookup)/(oracle − cell lookup).", "",
          f"Split-0 fallback by family: " + ", ".join(f"{k} → {v}" for k, v in per_split[0]["fallback_by_family"].items()), "",
          "router | reward | gain over cell lookup | share of gap captured", "---|---|---|---",
          f"GPT-4 always | {summary['gpt4_always'][0]:.4f} [{summary['gpt4_always'][1]:.4f}] | |",
          f"cell lookup | {cl[0]:.4f} [{cl[1]:.4f}] | |",
          f"tier-2 fallback always | {summary['fallback_always'][0]:.4f} [{summary['fallback_always'][1]:.4f}] | |"]
    for rr in C.REP_ORDER:
        P = summary["probe"][rr]
        md.append(f"probe router, {C.REP_LABEL[rr]} | {P['router'][0]:.4f} [{P['router'][1]:.4f}] | {P['gain_over_cell'][0]:+.4f} [{P['gain_over_cell'][1]:+.4f}] | {P['share_of_gap_captured'][0]:.2f} [{P['share_of_gap_captured'][1]:.2f}] (routes {P['share_routed_to_fallback']:.2f} to fallback; test-optimal τ would give {P['test_optimal_tau_upper_bound'][0]:.4f})")
    md += [f"(family, true tier) lookup — perfect knowledge of 'GPT-4 fails' | {summary['family_true_tier'][0]:.4f} [{summary['family_true_tier'][1]:.4f}] | {summary['family_true_tier'][0]-cl[0]:+.4f} | {(summary['family_true_tier'][0]-cl[0])/(orc[0]-cl[0]):.2f}",
           f"oracle | {orc[0]:.4f} [{orc[1]:.4f}] | {orc[0]-cl[0]:+.4f} | 1.00"]
    result["seconds"] = time.time() - t0; C.save(C.out_dir(EXP), result, md)


if __name__ == "__main__":
    main()
