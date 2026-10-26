"""exp80 — Test A at lambda = 0: is "leave GPT-4" readable from the text?

Label (RouterBench, released outcomes only):
  positive  GPT-4 fails (reward < 0.5) AND at least one tier-2 model solves  -> the gap query
  negative  GPT-4 solves
  excluded  GPT-4 fails and no tier-2 model solves: nothing at tier 2 rescues these, so they are
            not a routing decision; removed from train and test alike.
At lambda = 0 the positives are 76% of the oracle-minus-lookup gap (exp30 part A attribution).

Representations, a ladder: hashed bag-of-words | emitted GTE | GTE's own L21 | Qwen2.5-7B reader L21.
Probe: RidgeCV on the standardised representation, the study's fixed alpha grid, fit on training rows.
Conditioning: AUC inside each family (pooled by rows, and equal weight over families). Family and
difficulty conditioning is NOT used here: GPT-4 contributes 0 to the solve count on every positive
and 1 on every negative, so the count partly encodes the label.
Also: unconditional AUC for reference, precision/recall at the top 10% of scores, family-stratified
query bootstrap on split 0, and a within-family permutation null (labels shuffled inside each family
on the TRAINING rows, probe refitted) on split 0 for GTE and the reader.
Writes results/oracle_gap/exp80_test_a_lam0/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from sklearn.metrics import roc_auc_score

EXP = "exp80_test_a_lam0"; N_BOOT = 1000; N_PERM = 50; NULL_REPS = ("gte", "reader_l21")


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index()
    fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); S = C.solved_matrix(panel); tc = C.tier_columns(midx)
    g4 = S[:, tc[1][0]]; t2 = S[:, tc[2]].any(1)
    y = np.full(n, -1, dtype=np.int8); y[g4] = 0; y[~g4 & t2] = 1
    keep = y >= 0
    print(f"label: pos={int((y==1).sum())} neg={int((y==0).sum())} excluded={int((~keep).sum())}", flush=True)
    reps = {r: C.load_rep(r, panel) for r in C.REP_ORDER}
    print("reps loaded", {r: reps[r].shape for r in reps}, f"({time.time()-t0:.0f}s)", flush=True)

    res = {r: {"pooled": [], "eqw": [], "uncond": [], "per_family": [], "prec10": [], "alpha": []} for r in C.REP_ORDER}
    split0 = {}
    for seed in C.SEEDS:
        tr, te = C.split(n, fam, seed); tr = tr[keep[tr]]; te = te[keep[te]]
        for r in C.REP_ORDER:
            s, alpha = C.fit_scores(reps[r][tr], y[tr], reps[r][te])
            p, e, per = C.family_auc(y[te], s, fam[te])
            res[r]["pooled"].append(p); res[r]["eqw"].append(e); res[r]["per_family"].append(per)
            res[r]["uncond"].append(float(roc_auc_score(y[te], s)))
            res[r]["prec10"].append(C.precision_at(y[te], s, 0.10)); res[r]["alpha"].append(alpha)
            if seed == C.SEEDS[0]:
                split0[r] = s
            print(f"seed {seed} {r:10s} within-family AUC {p:.3f} (eqw {e:.3f})  uncond {res[r]['uncond'][-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if seed == C.SEEDS[0]:
            split0["te"] = te

    def ms(v): return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    def agg_family(r):
        out = {}
        for g in range(F):
            vals = [pf.get(g) for pf in res[r]["per_family"] if pf.get(g) is not None]
            out[fnames[g]] = float(np.mean(vals)) if vals else None
        return out
    te0 = split0["te"]
    n_pos_fam = {fnames[g]: int(((fam[te0] == g) & (y[te0] == 1)).sum()) for g in range(F)}
    n_fam = {fnames[g]: int((fam[te0] == g).sum()) for g in range(F)}

    result = C.base(panel, {"experiment": EXP, "lambda": 0.0,
        "label": {"positive": "gpt-4 fails and >=1 tier-2 model solves", "negative": "gpt-4 solves",
                  "excluded": "gpt-4 fails and no tier-2 model solves", "n_pos": int((y==1).sum()),
                  "n_neg": int((y==0).sum()), "n_excluded": int((~keep).sum())},
        "results": {r: {"within_family_auc_pooled": ms(res[r]["pooled"]), "within_family_auc_eqw": ms(res[r]["eqw"]),
                        "unconditional_auc": ms(res[r]["uncond"]), "per_family_auc_mean_over_splits": agg_family(r),
                        "precision_at_top10pct": {"precision": ms([p[0] for p in res[r]["prec10"]]),
                                                  "recall": ms([p[1] for p in res[r]["prec10"]]),
                                                  "base_rate": ms([p[2] for p in res[r]["prec10"]])},
                        "ridge_alpha": res[r]["alpha"]} for r in C.REP_ORDER},
        "split0_test_positives_per_family": n_pos_fam, "split0_test_rows_per_family": n_fam})

    def md_main():
        L = [f"# {EXP} — Test A at λ=0: is 'leave GPT-4' readable from the text? ({panel.name})", "",
             f"Positive = GPT-4 fails and at least one tier-2 model solves ({result['label']['n_pos']} rows); negative = GPT-4 solves ({result['label']['n_neg']}); excluded = GPT-4 fails and no tier-2 model solves ({result['label']['n_excluded']}). Ridge on the standardised representation; AUC of the ridge score inside each family. Mean over 5 splits [min, max].", "",
             "representation | within-family AUC, pooled | equal weight over families | unconditional | precision @ top 10% (recall) | base rate",
             "---|---|---|---|---|---"]
        for r in C.REP_ORDER:
            R = result["results"][r]
            L.append(f"{C.REP_LABEL[r]} | {R['within_family_auc_pooled']['mean']:.3f} [{R['within_family_auc_pooled']['min']:.3f}, {R['within_family_auc_pooled']['max']:.3f}] | {R['within_family_auc_eqw']['mean']:.3f} | {R['unconditional_auc']['mean']:.3f} | {R['precision_at_top10pct']['precision']['mean']:.3f} ({R['precision_at_top10pct']['recall']['mean']:.3f}) | {R['precision_at_top10pct']['base_rate']['mean']:.3f}")
        L += ["", "## Per family (within-family AUC, mean over splits; split-0 test positives / rows)", "",
              "family | " + " | ".join(C.REP_ORDER) + " | pos / rows", "---|" + "---|" * len(C.REP_ORDER) + "---"]
        for g in range(F):
            L.append(f"{fnames[g]} | " + " | ".join(C.f3(result['results'][r]['per_family_auc_mean_over_splits'][fnames[g]]) for r in C.REP_ORDER) + f" | {n_pos_fam[fnames[g]]} / {n_fam[fnames[g]]}")
        return L

    d = C.out_dir(EXP); C.save(d, result, md_main())   # checkpoint before the slow parts

    # split 0: bootstrap CIs for every representation
    yb = y[te0]; fb = fam[te0]
    result["split0_bootstrap"] = {}
    for r in C.REP_ORDER:
        ci_p, ci_e = C.bootstrap_family_auc(yb, split0[r], fb, n=N_BOOT, seed=0)
        result["split0_bootstrap"][r] = {"within_family_auc_pooled_ci95": ci_p, "within_family_auc_eqw_ci95": ci_e}
        print(f"bootstrap {r}: pooled {ci_p}  eqw {ci_e}  ({time.time()-t0:.0f}s)", flush=True)
    md = md_main() + ["", "## Split 0, 95% family-stratified query bootstrap", "", "representation | pooled AUC CI | equal-weight AUC CI", "---|---|---"]
    md += [f"{C.REP_LABEL[r]} | [{result['split0_bootstrap'][r]['within_family_auc_pooled_ci95'][0]:.3f}, {result['split0_bootstrap'][r]['within_family_auc_pooled_ci95'][1]:.3f}] | [{result['split0_bootstrap'][r]['within_family_auc_eqw_ci95'][0]:.3f}, {result['split0_bootstrap'][r]['within_family_auc_eqw_ci95'][1]:.3f}]" for r in C.REP_ORDER]
    C.save(d, result, md)

    # split 0: within-family permutation null, probe refitted each draw
    tr0, _ = C.split(n, fam, C.SEEDS[0]); tr0 = tr0[keep[tr0]]
    result["split0_permutation_null"] = {}
    for r in NULL_REPS:
        nulls = C.permutation_null(reps[r][tr0], y[tr0], fam[tr0], reps[r][te0], yb, fb, n=N_PERM, seed=0)
        obs = res[r]["pooled"][0]
        result["split0_permutation_null"][r] = {"n_draws": N_PERM, "null_mean": float(np.mean(nulls)), "null_p97.5": float(np.percentile(nulls, 97.5)),
                                                "observed": obs, "p_value": float(np.mean(np.array(nulls) >= obs))}
        print(f"null {r}: mean {np.mean(nulls):.3f} 97.5th {np.percentile(nulls,97.5):.3f} observed {obs:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    md += ["", f"## Split 0, within-family permutation null ({N_PERM} draws, labels shuffled inside each family on the training rows, probe refitted)", "",
           "representation | null mean | null 97.5th | observed | p", "---|---|---|---|---"]
    md += [f"{C.REP_LABEL[r]} | {v['null_mean']:.3f} | {v['null_p97.5']:.3f} | {v['observed']:.3f} | {v['p_value']:.3f}" for r, v in result["split0_permutation_null"].items()]
    result["seconds"] = time.time() - t0
    C.save(d, result, md)


if __name__ == "__main__":
    main()
