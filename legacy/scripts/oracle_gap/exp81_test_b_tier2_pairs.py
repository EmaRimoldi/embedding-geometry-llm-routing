"""exp81 — Test B: inside tier 2, is "which model solves" readable from the text?

For each of the 10 pairs (A, B) of tier-2 models, keep only the queries where EXACTLY ONE of the
two solves (the disagreement set); everything both solve or both fail carries no decision.
Label: 1 = A solves and B fails; 0 = B solves and A fails.
Conditioning: within family, and within family x difficulty. The second is legitimate here because
the pair contributes exactly 1 to the solve count in either class, so the count carries no label.
Representations: the 4-rung ladder. Probe and splits as exp80. Per-pair per-rep AUC, mean over 5
splits; per-family AUC per pair (specialisation, if any, will show on one family, likely MBPP);
family-stratified bootstrap on split 0 for every pair; within-family permutation null on split 0
for GTE and the reader, 20 refitted draws per pair.
PROVES whether the within-tier residual of the gap is readable: this is the variance claim.
Writes results/oracle_gap/exp81_test_b_tier2_pairs/.
"""
from __future__ import annotations
import itertools, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

EXP = "exp81_test_b_tier2_pairs"; N_BOOT = 1000; N_PERM = 20; NULL_REPS = ("gte", "reader_l21")


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index()
    fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); S = C.solved_matrix(panel); n_solved = S.sum(1)
    pairs = list(itertools.combinations(C.TIERS[2], 2))
    reps = {r: C.load_rep(r, panel) for r in C.REP_ORDER}
    print(f"{len(pairs)} pairs; reps loaded ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for a, b in pairs:
        key = f"{a} vs {b}"; ia, ib = midx[a], midx[b]
        mask = S[:, ia] != S[:, ib]; y = S[:, ia].astype(np.int8)
        rec = {"n_disagree": int(mask.sum()), "share_of_panel": float(mask.mean()), "base_rate_A_solves": float(y[mask].mean()),
               "per_rep": {r: {"fam": [], "fam_diff": [], "per_family": [], "alpha": []} for r in C.REP_ORDER}, "split0": {}}
        s0 = {}
        for seed in C.SEEDS:
            tr, te = C.split(n, fam, seed); tr = tr[mask[tr]]; te = te[mask[te]]
            for r in C.REP_ORDER:
                s, alpha = C.fit_scores(reps[r][tr], y[tr], reps[r][te])
                p, _, per = C.family_auc(y[te], s, fam[te]); pd, _, _ = C.family_difficulty_auc(y[te], s, fam[te], n_solved[te])
                R = rec["per_rep"][r]; R["fam"].append(p); R["fam_diff"].append(pd); R["per_family"].append(per); R["alpha"].append(alpha)
                if seed == C.SEEDS[0]: s0[r] = s
            if seed == C.SEEDS[0]: rec["split0"]["te"] = te
        te0 = rec["split0"]["te"]
        for r in C.REP_ORDER:
            ci_p, _ = C.bootstrap_family_auc(y[te0], s0[r], fam[te0], n=N_BOOT, seed=0)
            rec["split0"][r] = {"within_family_auc_ci95": ci_p}
        rec["split0"]["n_test"] = int(len(te0)); del rec["split0"]["te"]
        print(f"{key:45s} n={rec['n_disagree']:5d}  " + "  ".join(f"{r} {np.mean(rec['per_rep'][r]['fam']):.3f}" for r in C.REP_ORDER) + f"  ({time.time()-t0:.0f}s)", flush=True)
        out[key] = rec

    def ms(v): return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    def agg_family(rec, r):
        d = {}
        for g in range(F):
            vals = [pf.get(g) for pf in rec["per_rep"][r]["per_family"] if pf.get(g) is not None]
            d[fnames[g]] = float(np.mean(vals)) if vals else None
        return d
    results = {}
    for key, rec in out.items():
        results[key] = {"n_disagree": rec["n_disagree"], "share_of_panel": rec["share_of_panel"], "base_rate_A_solves": rec["base_rate_A_solves"],
                        "reps": {r: {"within_family_auc": ms(rec["per_rep"][r]["fam"]), "within_family_x_difficulty_auc": ms(rec["per_rep"][r]["fam_diff"]),
                                     "per_family_auc": agg_family(rec, r), "split0_bootstrap_ci95": rec["split0"][r]["within_family_auc_ci95"],
                                     "ridge_alpha": rec["per_rep"][r]["alpha"]} for r in C.REP_ORDER}}
    result = C.base(panel, {"experiment": EXP, "tier": 2, "pairs": [f"{a} vs {b}" for a, b in pairs],
                            "label": "1 = first model solves and second fails; 0 = the reverse; queries where both solve or both fail are excluded",
                            "results": results})

    def md_main():
        L = [f"# {EXP} — Test B: inside tier 2, is 'which model solves' readable? ({panel.name})", "",
             "Each row is a pair of tier-2 models restricted to the queries where exactly one solves. AUC of the ridge score inside each family, mean over 5 splits. Second number in each cell: inside family x difficulty strata.", "",
             "pair | disagreement rows (share) | " + " | ".join(C.REP_ORDER) + " | split-0 CI, reader",
             "---|---|" + "---|" * len(C.REP_ORDER) + "---"]
        for key, R in results.items():
            cells = " | ".join(f"{R['reps'][r]['within_family_auc']['mean']:.3f} / {R['reps'][r]['within_family_x_difficulty_auc']['mean']:.3f}" for r in C.REP_ORDER)
            ci = R["reps"]["reader_l21"]["split0_bootstrap_ci95"]
            L.append(f"{key} | {R['n_disagree']} ({R['share_of_panel']:.2f}) | {cells} | [{ci[0]:.3f}, {ci[1]:.3f}]")
        mean_over_pairs = {r: float(np.mean([results[k]["reps"][r]["within_family_auc"]["mean"] for k in results])) for r in C.REP_ORDER}
        L += ["", "Mean over the ten pairs, within-family AUC: " + ", ".join(f"{C.REP_LABEL[r]} {mean_over_pairs[r]:.3f}" for r in C.REP_ORDER), "",
              "## Per family, reader L21 (mean over splits); the place specialisation would show", "",
              "pair | " + " | ".join(fnames), "---|" + "---|" * F]
        for key, R in results.items():
            L.append(f"{key} | " + " | ".join(C.f3(R["reps"]["reader_l21"]["per_family_auc"][f]) for f in fnames))
        L += ["", "## Per family, emitted GTE (mean over splits)", "", "pair | " + " | ".join(fnames), "---|" + "---|" * F]
        for key, R in results.items():
            L.append(f"{key} | " + " | ".join(C.f3(R["reps"]["gte"]["per_family_auc"][f]) for f in fnames))
        return L
    d = C.out_dir(EXP); C.save(d, result, md_main())   # checkpoint before the null

    result["split0_permutation_null"] = {}
    for a, b in pairs:
        key = f"{a} vs {b}"; ia, ib = midx[a], midx[b]; mask = S[:, ia] != S[:, ib]; y = S[:, ia].astype(np.int8)
        tr0, te0 = C.split(n, fam, C.SEEDS[0]); tr0 = tr0[mask[tr0]]; te0 = te0[mask[te0]]
        result["split0_permutation_null"][key] = {}
        for r in NULL_REPS:
            nulls = C.permutation_null(reps[r][tr0], y[tr0], fam[tr0], reps[r][te0], y[te0], fam[te0], n=N_PERM, seed=0)
            obs = out[key]["per_rep"][r]["fam"][0]
            result["split0_permutation_null"][key][r] = {"n_draws": N_PERM, "null_mean": float(np.mean(nulls)), "null_p97.5": float(np.percentile(nulls, 97.5)),
                                                          "observed": obs, "p_value": float(np.mean(np.array(nulls) >= obs))}
        print(f"null {key:45s} " + "  ".join(f"{r} obs {result['split0_permutation_null'][key][r]['observed']:.3f} null97.5 {result['split0_permutation_null'][key][r]['null_p97.5']:.3f}" for r in NULL_REPS) + f"  ({time.time()-t0:.0f}s)", flush=True)
    md = md_main() + ["", f"## Split 0, within-family permutation null ({N_PERM} refitted draws per pair)", "", "pair | " + " | ".join(f"{r}: observed / null 97.5th / p" for r in NULL_REPS), "---|" + "---|" * len(NULL_REPS)]
    for key in result["split0_permutation_null"]:
        md.append(f"{key} | " + " | ".join(f"{v['observed']:.3f} / {v['null_p97.5']:.3f} / {v['p_value']:.2f}" for v in result["split0_permutation_null"][key].values()))
    result["seconds"] = time.time() - t0
    C.save(d, result, md)


if __name__ == "__main__":
    main()
