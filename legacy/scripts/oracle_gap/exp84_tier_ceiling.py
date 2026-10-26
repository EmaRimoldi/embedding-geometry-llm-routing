"""exp84 — the tier ceiling: what is knowing the oracle's TIER worth? (embedder-agnostic)

For each lambda in {0, 10, 100, 1000} and each split: the TRUE tier of the oracle's model on each
query, used as a lookup key together with the family (and, in a second variant, with the released
topic cell). The lookup stores the best model by mean TRAINING utility per key, family fallback
below 50 rows. Test-row utility is compared with the family lookup, the (family, topic) cell lookup
and the oracle. "Share of gap captured" = (tier lookup - cell lookup) / (oracle - cell lookup).
No embedding is involved, so every number here is a property of the benchmark's outcome table.
At lambda=0 the oracle ties across every solver; both tie-breaks are reported: strongest rung
(lowest tier number among the solvers) and cheapest rung (highest tier number).
PROVES how much of the gap sits at the rung level, where a coarse ordered label could in principle
recover it, versus inside a rung, where exp81 measures whether anything is readable. Comparators
from exp43/exp44: true scalar difficulty as key +0.003 over the cell lookup at lambda=100, true
best model +0.200.
Writes results/oracle_gap/exp84_tier_ceiling/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C, lookups as L

EXP = "exp84_tier_ceiling"; LAMS = (0.0, 10.0, 100.0, 1000.0)


def agg(v, fam, F):
    return float(v.mean()), float(np.mean([v[fam == g].mean() for g in range(F) if (fam == g).any()]))


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index()
    fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); M = len(names); tc = C.tier_columns(midx)
    tier_of = np.zeros(M, dtype=np.int64)
    for t, cols in tc.items():
        for cc in cols: tier_of[cc] = t
    _, cell = np.unique(np.array([str(t) for t in C.topic_str(panel)], dtype=object), return_inverse=True); n_cells = int(cell.max()) + 1
    rows = np.arange(n)
    out = {}
    for lam in LAMS:
        U = L.utility(panel, lam); u_o = U.max(1)
        # oracle's tier per query, two tie-breaks
        is_max = U >= (u_o[:, None] - 1e-12)
        tiers_mat = np.where(is_max, tier_of[None, :], 99)
        tier_strong = tiers_mat.min(1)
        tiers_mat2 = np.where(is_max, tier_of[None, :], 0)
        tier_cheap = tiers_mat2.max(1)
        variants = {"strongest_rung": tier_strong, "cheapest_rung": tier_cheap} if lam == 0.0 else {"oracle_tier": tier_strong}
        rec = {"tier_distribution_of_oracle": {}, "per_split": []}
        for vname, tier in variants.items():
            rec["tier_distribution_of_oracle"][vname] = {int(t): float((tier == t).mean()) for t in range(1, 5)}
        for seed in C.SEEDS:
            tr, te = C.split(n, fam, seed)
            fam_best, _ = L.fit_lookup(U[tr], fam[tr], fam[tr], F, F)
            m_fam = fam_best[fam]
            cb_f, cb_c = L.fit_lookup(U[tr], cell[tr], fam[tr], F, n_cells); m_cell = L.apply_lookup(fam, cell, cb_f, cb_c)
            row = {"seed": seed, "family_lookup": agg(U[te, m_fam[te]], fam[te], F), "cell_lookup": agg(U[te, m_cell[te]], fam[te], F),
                   "oracle": agg(u_o[te], fam[te], F), "best_single": None, "tier_keys": {}}
            bs = int(U[tr].mean(0).argmax()); row["best_single"] = agg(U[te, bs], fam[te], F)
            for vname, tier in variants.items():
                k_ft = L.combine_keys(fam, tier); kb_f, kb = L.fit_lookup(U[tr], k_ft[tr], fam[tr], F, int(k_ft.max()) + 1); m_ft = L.apply_lookup(fam, k_ft, kb_f, kb)
                k_ct = L.combine_keys(cell, tier); kb_f2, kb2 = L.fit_lookup(U[tr], k_ct[tr], fam[tr], F, int(k_ct.max()) + 1); m_ct = L.apply_lookup(fam, k_ct, kb_f2, kb2)
                # what does the (family, tier) cell actually store, and how often does it match the oracle's model?
                same_as_oracle = float((U[te, m_ft[te]] >= u_o[te] - 1e-12).mean())
                row["tier_keys"][vname] = {"family+tier": agg(U[te, m_ft[te]], fam[te], F), "cell+tier": agg(U[te, m_ct[te]], fam[te], F),
                                           "family+tier_choice_is_oracle_optimal_share": same_as_oracle}
            rec["per_split"].append(row)
        out[str(lam)] = rec
        print(f"lam {lam:g} done ({time.time()-t0:.0f}s)", flush=True)

    def mean_pq(rows, path):
        vals = [r for r in rows]
        return float(np.mean([eval("r" + path)[0] for r in vals])), float(np.mean([eval("r" + path)[1] for r in vals]))
    result = C.base(panel, {"experiment": EXP, "lambdas": list(LAMS), "min_cell": L.MIN_CELL, "by_lambda": {}})
    md = [f"# {EXP} — the tier ceiling: what is knowing the oracle's tier worth? ({panel.name})", "",
          "Lookups fitted on training rows, evaluated on test rows, mean over 5 splits; per query [equal weight over families]. 'captured' = (tier lookup − cell lookup) / (oracle − cell lookup). Tiers: 1 GPT-4; 2 Claude-v2, Yi-34B, Claude-v1, GPT-3.5, Claude-instant; 3 Mixtral, WizardLM; 4 Llama-2, Mistral, Code-Llama.", "",
          "λ | key | family lookup | cell lookup | (family, true tier) | (cell, true tier) | oracle | gain over cell | share of gap captured | tier choice oracle-optimal on",
          "---|---|---|---|---|---|---|---|---|---"]
    for lam in LAMS:
        rec = out[str(lam)]; rows = rec["per_split"]
        fl = mean_pq(rows, "['family_lookup']"); cl = mean_pq(rows, "['cell_lookup']"); orc = mean_pq(rows, "['oracle']")
        entry = {"family_lookup": fl, "cell_lookup": cl, "oracle": orc, "best_single": mean_pq(rows, "['best_single']"), "tier_distribution_of_oracle": rec["tier_distribution_of_oracle"], "tier_keys": {}}
        for vname in rows[0]["tier_keys"]:
            ft = mean_pq(rows, f"['tier_keys']['{vname}']['family+tier']"); ct = mean_pq(rows, f"['tier_keys']['{vname}']['cell+tier']")
            same = float(np.mean([r["tier_keys"][vname]["family+tier_choice_is_oracle_optimal_share"] for r in rows]))
            cap = (ft[0] - cl[0]) / (orc[0] - cl[0]) if orc[0] > cl[0] else float("nan")
            cap_e = (ft[1] - cl[1]) / (orc[1] - cl[1]) if orc[1] > cl[1] else float("nan")
            entry["tier_keys"][vname] = {"family+tier": ft, "cell+tier": ct, "gain_over_cell": (ft[0] - cl[0], ft[1] - cl[1]), "share_of_gap_captured": (cap, cap_e), "choice_oracle_optimal_share": same}
            md.append(f"{lam:g} | {vname} | {fl[0]:.4f} [{fl[1]:.4f}] | {cl[0]:.4f} [{cl[1]:.4f}] | {ft[0]:.4f} [{ft[1]:.4f}] | {ct[0]:.4f} [{ct[1]:.4f}] | {orc[0]:.4f} [{orc[1]:.4f}] | {ft[0]-cl[0]:+.4f} [{ft[1]-cl[1]:+.4f}] | {cap:.2f} [{cap_e:.2f}] | {same:.3f}")
        result["by_lambda"][str(lam)] = entry
    md += ["", "## Where the oracle's model sits (share of queries per tier)", "", "λ | tie-break | tier 1 | tier 2 | tier 3 | tier 4", "---|---|---|---|---|---"]
    for lam in LAMS:
        for vname, dist in out[str(lam)]["tier_distribution_of_oracle"].items():
            md.append(f"{lam:g} | {vname} | " + " | ".join(f"{dist[t]:.3f}" for t in range(1, 5)))
    md += ["", "Comparators from exp43/exp44 at λ=100 over the cell lookup: true scalar difficulty +0.003; true best model +0.200; oracle +0.205."]
    result["seconds"] = time.time() - t0
    C.save(C.out_dir(EXP), result, md)


if __name__ == "__main__":
    main()
