"""exp11 - the held-out study re-read on cost-quality frontiers (no refits; saved per-lambda tables only).

Sources (read-only): heldout_generalization_20260920/aggregates/<panel>_family.csv (classical baselines,
query-weighted and macro per lambda), .../results/<panel>/family_transfer_routers/fold_method_table.csv
(per family per lambda), the fold JSONs of the study's prior-based methods (per family per lambda, seed-
averaged where seeds exist), and this folder's exp10 result.json (this series' routers). Every method
contributes one (mean cost, mean reward) point per lambda of the 15-point grid, per held-out family.
Metrics, following the published conventions:
  AIQ            RouterBench: area under the non-decreasing convex hull of (cost, reward), / max_cost,
                 max_cost = mean cost of the priciest single model on the same rows (rb.probes.sweep.aiq)
  AIQ_log        same hull integrated over log10(cost) between the cheapest and priciest single model
  APGR           RouteLLM-style: performance gap recovered, (reward - r_cheapest) / (r_strongest -
                 r_cheapest), read off the hull at cost fractions 0..1 between the cheapest and the
                 strongest single model, averaged over the fractions (their APGR over call fraction,
                 with cost in place of calls, since the routers are multi-model)
  PerfGain / CostSave   LLMRouterBench: best-reward point vs family-balanced best single; 1 - cost of
                 the cheapest point with reward >= best single / cost of best single
Aggregations: query-weighted (points pooled over held-out rows) and macro (per-family AIQ, averaged).
Noise: per-family dispersion of AIQ vs of utility at lambda = 50 and 100 (std across families divided by
the between-method range of the same quantity), and rank stability: mean Kendall tau of the method
ranking between adjacent lambdas of the sweep vs mean pairwise-order agreement of AIQ rankings over
1,000 family-bootstrap resamples.
Writes results/exp11_frontier_reproduction/<panel>/{result.json, result.md, frontier.png}.
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import kendalltau
import common as C
from rb.probes.sweep import LAMBDAS, aiq

EXP = "exp11_frontier_reproduction"; STUDY = C.E4 / "heldout_generalization_20260920"; LK = [f"{l:g}" for l in LAMBDAS]


def hull(pts):
    pts = pts[np.argsort(pts[:, 0])]; h = []
    for cc, rr in pts:
        if h and rr < h[-1][1]: continue
        h.append([cc, rr])
    return np.array(h)


def aiq_log(pts, cmin, cmax):
    h = hull(pts); x = np.log10(np.clip(h[:, 0], cmin, cmax)); y = h[:, 1]
    x = np.r_[np.log10(cmin), x, np.log10(cmax)]; y = np.r_[y[0], y, y[-1]]
    o = np.argsort(x); return float(np.trapezoid(y[o], x[o]) / (np.log10(cmax) - np.log10(cmin)))


def aiq_range(pts, cmin, cmax):
    """RouterBench normalisation: hull area over the common cost range [cmin, cmax] / (cmax - cmin), linear cost,
    hull extended flat to both ends of the range."""
    h = hull(pts); x = np.r_[cmin, np.clip(h[:, 0], cmin, cmax), cmax]; y = np.r_[h[0, 1], h[:, 1], h[-1, 1]]; o = np.argsort(x)
    return float(np.trapezoid(y[o], x[o]) / max(cmax - cmin, 1e-12))


def apgr(pts, c_weak, r_weak, c_strong, r_strong):
    h = hull(pts); fr = np.linspace(0, 1, 101); cs = c_weak + fr * (c_strong - c_weak)
    rr = np.interp(cs, h[:, 0], h[:, 1], left=h[0, 1], right=h[-1, 1])
    return float(np.mean((rr - r_weak) / max(r_strong - r_weak, 1e-9)))


def load_curves(panel_name, fn):
    """method -> family -> {lam_key: (cost, reward)}; plus n_test per family."""
    cur = {}; ntest = {}
    # 1. transfer routers + references (per family)
    p = STUDY / "results" / panel_name / "family_transfer_routers" / "fold_method_table.csv"
    if p.exists():
        for row in csv.DictReader(open(p)):
            m = "study:" + row["method"]; cur.setdefault(m, {}).setdefault(row["family"], {})[f"{float(row['lambda']):g}"] = (float(row["cost"]), float(row["reward"])); ntest[row["family"]] = int(row["n_test"])
    # 2. prior-based methods from fold JSONs
    def add(m, fam_name, table):
        for L, v in table.items():
            if isinstance(v, dict) and "cost" in v and "reward" in v: cur.setdefault(m, {}).setdefault(fam_name, {})[f"{float(L):g}"] = (float(v["cost"]), float(v["reward"]))
    for dname, label in (("family_lowrank_outcome_router", "study:low_rank_outcome_router"), ("family_utility_cluster_prior", "study:utility_cluster_prior"), ("family_strong_prior_residual", "study:strong_prior_residual"),
                         ("family_learned_gate", "study:learned_gate"), ("family_posterior_safe_switch", "study:posterior_safe_switch"), ("family_domain_prior_variants", "study:soft_prior_baseline_variant"), ("family_proxrouter", "study:proxrouter_k32"), ("family_shrink_v1", "study:shrink_v1")):
        acc = {}
        for f in sorted(glob.glob(str(STUDY / "results" / panel_name / dname / "fold_*.json"))):
            j = json.load(open(f)); fam_name = str(j.get("heldout_family")); res = j.get("results", {})
            if dname == "family_domain_prior_variants": res = res.get("baseline", {})
            if dname == "family_proxrouter": res = res.get("proxrouter_k32", {})
            tables = [res[s] for s in res if isinstance(res[s], dict) and set(LK[:3]) & set(res[s].keys())] if any(isinstance(res.get(k), dict) and set(LK[:3]) & set(res[k].keys()) for k in res) else [res]
            for t in tables: acc.setdefault(fam_name, []).append({L: (t[L]["cost"], t[L]["reward"]) for L in t if isinstance(t[L], dict) and "cost" in t[L]})
        for fam_name, lst in acc.items():
            keys = set.intersection(*[set(d) for d in lst]) if lst else set()
            table = {L: {"cost": float(np.mean([d[L][0] for d in lst])), "reward": float(np.mean([d[L][1] for d in lst]))} for L in keys}
            add(label, fam_name, table)
    # 2b. classical baselines of the study from aggregates/<panel>_family.csv (query-weighted and macro per lambda; no per-family curves)
    p = STUDY / "aggregates" / f"{panel_name}_family.csv"; pooled_csv = {}
    if p.exists():
        for row in csv.DictReader(open(p)):
            m = "study:" + row["method"]; L = f"{float(row['lambda']):g}"
            pooled_csv.setdefault(m, {"qw": {}, "macro": {}}); pooled_csv[m]["qw"][L] = (float(row["cost_query_weighted"]), float(row["reward_query_weighted"])); pooled_csv[m]["macro"][L] = (float(row["cost_fold_macro"]), float(row["reward_fold_macro"]))
    # 3. this series (exp10)
    p = C.RESULTS / "exp10_frontier_aiq" / panel_name / "result.json"
    if p.exists():
        j = json.load(open(p))
        if "frontier_per_family" in j:
            for m, byfam in j["frontier_per_family"].items():
                for fam_name, pts in byfam.items(): cur.setdefault("ours:" + m, {})[fam_name] = {LK[i]: (pts[i][0], pts[i][1]) for i in range(len(pts))}
    return cur, ntest, pooled_csv


def main():
    panel_name = sys.argv[1]; D = C.load(panel_name); r, c, fam, fn = D["r"], D["c"], D["fam"], D["fnames"]; F = len(fn)
    cur, ntest, pooled_csv = load_curves(panel_name, fn); ntest = {f: int((fam == fn.index(f)).sum()) for f in fn}
    cmax_f = {f: float(c[fam == fn.index(f)].mean(0).max()) for f in fn}; cmin_f = {f: float(c[fam == fn.index(f)].mean(0).min()) for f in fn}
    rbar_all = r.mean(0); cbar_all = c.mean(0); strong = int(rbar_all.argmax()); weak = int(cbar_all.argmin())
    fam_bal = int(np.mean([r[fam == g].mean(0) for g in range(F)], 0).argmax())
    methods = [m for m in cur if len(cur[m]) == F and all(len(cur[m][f]) >= len(LK) - 1 for f in fn)]
    res = {"methods": methods, "per_family_aiq": {}, "aiq_macro": {}, "aiq_qw": {}, "aiq_log_qw": {}, "apgr_qw": {}, "perfgain": {}, "costsave": {}, "utility_qw": {}, "per_family_utility": {}}
    W = np.array([ntest[f] for f in fn], float); W /= W.sum()
    for m in methods:
        per = []; pooled = []
        for L in LK:
            pts = np.array([cur[m][f].get(L, cur[m][f][list(cur[m][f])[0]]) for f in fn]); pooled.append((W @ pts[:, 0], W @ pts[:, 1]))
        pooled = np.array(pooled); res["aiq_qw"][m] = aiq(pooled, float(cbar_all.max()))
        res["aiq_log_qw"][m] = aiq_log(pooled, float(cbar_all.min()), float(cbar_all.max()))
        res.setdefault("aiq_range_qw", {})[m] = aiq_range(pooled, float(cbar_all.min()), float(cbar_all.max()))
        res["apgr_qw"][m] = apgr(pooled, float(cbar_all[weak]), float(rbar_all[weak]), float(cbar_all[strong]), float(rbar_all[strong]))
        bs_r, bs_c = float(rbar_all[fam_bal]), float(cbar_all[fam_bal]); res["perfgain"][m] = float(pooled[:, 1].max() / bs_r - 1)
        ok = pooled[pooled[:, 1] >= bs_r - 1e-12]; res["costsave"][m] = (float(1 - ok[:, 0].min() / bs_c) if len(ok) else None)
        res["per_family_aiq"][m] = {f: aiq(np.array([cur[m][f][L] for L in LK if L in cur[m][f]]), cmax_f[f]) for f in fn}
        res["aiq_macro"][m] = float(np.mean(list(res["per_family_aiq"][m].values())))
        res["utility_qw"][m] = {L: float(pooled[i, 1] - float(L) * pooled[i, 0]) for i, L in enumerate(LK)}
        res["per_family_utility"][m] = {L: {f: cur[m][f][L][1] - float(L) * cur[m][f][L][0] for f in fn if L in cur[m][f]} for L in LK}
    # pooled-only methods (classical study baselines): AIQ on the pooled curves, no per-family statistics
    res["frontier_qw"] = {}; res["frontier_macro"] = {}; res["pooled_only_methods"] = []
    for m in methods:
        pooled = np.array([[W @ np.array([cur[m][f].get(L, cur[m][f][list(cur[m][f])[0]])[0] for f in fn]), W @ np.array([cur[m][f].get(L, cur[m][f][list(cur[m][f])[0]])[1] for f in fn])] for L in LK])
        res["frontier_qw"][m] = pooled.tolist(); res["frontier_macro"][m] = [[float(np.mean([cur[m][f][L][0] for f in fn if L in cur[m][f]])), float(np.mean([cur[m][f][L][1] for f in fn if L in cur[m][f]]))] for L in LK]
    for m, d in pooled_csv.items():
        if m in methods or any(L not in d["qw"] for L in LK): continue
        pooled = np.array([d["qw"][L] for L in LK]); pm = np.array([d["macro"][L] for L in LK]); res["pooled_only_methods"].append(m)
        res["frontier_qw"][m] = pooled.tolist(); res["frontier_macro"][m] = pm.tolist()
        res["aiq_qw"][m] = aiq(pooled, float(cbar_all.max())); res["aiq_range_qw"][m] = aiq_range(pooled, float(cbar_all.min()), float(cbar_all.max())); res["aiq_log_qw"][m] = aiq_log(pooled, float(cbar_all.min()), float(cbar_all.max()))
        res["aiq_macro"][m] = aiq(pm, float(np.mean([cmax_f[f] for f in fn]))); res["apgr_qw"][m] = apgr(pooled, float(cbar_all[weak]), float(rbar_all[weak]), float(cbar_all[strong]), float(rbar_all[strong]))
        bs_r, bs_c = float(rbar_all[fam_bal]), float(cbar_all[fam_bal]); res["perfgain"][m] = float(pooled[:, 1].max() / bs_r - 1); ok = pooled[pooled[:, 1] >= bs_r - 1e-12]; res["costsave"][m] = (float(1 - ok[:, 0].min() / bs_c) if len(ok) else None)
        res["utility_qw"][m] = {L: float(pooled[i, 1] - float(L) * pooled[i, 0]) for i, L in enumerate(LK)}; res["per_family_aiq"][m] = {}
    # noise: dispersion and rank stability
    def rank_vec(key_fn): return np.array([key_fn(m) for m in methods])
    rs = np.random.default_rng(0); B = 1000; agree = []
    A = np.array([[res["per_family_aiq"][m][f] for f in fn] for m in methods])       # methods x families
    base_order = np.argsort(-A.mean(1))
    for _ in range(B):
        idx = rs.integers(0, F, F); o = np.argsort(-A[:, idx].mean(1)); agree.append(kendalltau(base_order.argsort(), o.argsort()).correlation)
    taus = []
    for i in range(len(LK) - 1):
        u1 = rank_vec(lambda m: res["utility_qw"][m][LK[i]]); u2 = rank_vec(lambda m: res["utility_qw"][m][LK[i + 1]]); taus.append(kendalltau(u1, u2).correlation)
    def disp(vals_by_method):
        per = np.array([[vals_by_method[m][f] for f in fn] for m in methods]); within = per.std(1).mean(); between = per.mean(1).max() - per.mean(1).min(); return float(within / max(between, 1e-9))
    res["noise"] = {"family_bootstrap_kendall_tau_of_AIQ_ranking_mean": float(np.nanmean(agree)), "adjacent_lambda_kendall_tau_of_utility_ranking": [float(t) for t in taus],
                    "adjacent_lambda_kendall_tau_mean": float(np.nanmean(taus)), "dispersion_AIQ": disp(res["per_family_aiq"]),
                    "dispersion_utility_lam50": disp({m: res["per_family_utility"][m]["50"] for m in methods}), "dispersion_utility_lam100": disp({m: res["per_family_utility"][m]["100"] for m in methods})}
    out = C.provenance(D); out.update({"experiment": EXP, "lambdas": LK, **res})
    order = sorted(list(res["aiq_qw"]), key=lambda m: -res["aiq_qw"][m])
    md = [f"# {EXP} ({panel_name})", "", "Held-out-family results of the study and of this series re-read as cost-quality frontiers (no refits). AIQ = RouterBench hull area / priciest-model cost; AIQ_log over log10 cost; APGR = RouteLLM-style gap recovered between cheapest and strongest single model, averaged over cost fractions; PerfGain/CostSave = LLMRouterBench vs family-balanced best single.", "",
          "method | AIQ qw (repo: /max cost) | AIQ qw (RouterBench: / cost range) | AIQ macro (repo) | AIQ_log qw | APGR qw (adapted) | PerfGain | CostSave | u(λ=50) qw | u(λ=100) qw | AIQ per-family min..max", "---|---|---|---|---|---|---|---|---|---|---"]
    for m in order:
        pf = res["per_family_aiq"][m] or {"-": float("nan")}; md.append(f"{m} | {res['aiq_qw'][m]:.4f} | {res['aiq_range_qw'][m]:.4f} | {res['aiq_macro'][m]:.4f} | {res['aiq_log_qw'][m]:.4f} | {res['apgr_qw'][m]:.3f} | {res['perfgain'][m]:+.3f} | " + (f"{res['costsave'][m]:.3f}" if res["costsave"][m] is not None else "n/a") + f" | {res['utility_qw'][m]['50']:.3f} | {res['utility_qw'][m]['100']:.3f} | {min(pf.values()):.3f}..{max(pf.values()):.3f}")
    nz = res["noise"]
    md += ["", "## noise", "", f"- family-dispersion (mean within-method std across families / between-method range): AIQ {nz['dispersion_AIQ']:.2f}; utility λ=50 {nz['dispersion_utility_lam50']:.2f}; utility λ=100 {nz['dispersion_utility_lam100']:.2f}",
           f"- rank stability: Kendall τ of the method ranking between adjacent λ of the sweep, mean {nz['adjacent_lambda_kendall_tau_mean']:.2f} (per step: " + ", ".join(f"{t:.2f}" for t in nz["adjacent_lambda_kendall_tau_of_utility_ranking"]) + f"); Kendall τ of the AIQ ranking under family bootstrap, mean {nz['family_bootstrap_kendall_tau_of_AIQ_ranking_mean']:.2f}", "",
           "## per-family AIQ", "", "method | " + " | ".join(fn), "---|" + "---|" * F]
    for m in order:
        if res["per_family_aiq"][m]: md.append(f"{m} | " + " | ".join(f"{res['per_family_aiq'][m][f]:.3f}" for f in fn))
    d = C.RESULTS / EXP / panel_name; C.save(d, out, md)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5)); show = [m for m in order if any(k in m for k in ("ours:", "knn100", "best_single", "family_ceiling", "oracle", "soft_prior", "proxrouter", "low_rank", "learned_gate"))]
        for m in show:
            pts = np.array([[W @ np.array([cur[m][f][L][0] for f in fn]), W @ np.array([cur[m][f][L][1] for f in fn])] for L in LK if all(L in cur[m][f] for f in fn)])
            ax.plot(pts[:, 0] * 1e3, pts[:, 1], marker="o", ms=3, lw=1, label=m)
        ax.set_xscale("log"); ax.set_xlabel("mean cost per query (e-3 USD, log)"); ax.set_ylabel("mean reward"); ax.set_title(f"{panel_name}: held-out family, query-weighted frontiers"); ax.grid(alpha=.3); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(d / "frontier.png", dpi=150)
    except Exception as e: print("figure skipped:", e)


if __name__ == "__main__":
    main()
