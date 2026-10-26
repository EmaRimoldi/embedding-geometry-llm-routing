"""exp12 - the in-domain baselines re-read on cost-quality frontiers (saved curves; one recomputed lookup).

Sources (read-only), per panel: exp08 basic_model_tests (kNN k=5/10/40/100, family lookup
'best_model_per_family', classify-then-family-lookup, references best_single and oracle) and the exp64
baselines (K-means-64 utility top-1/top-3, RouteLLM-MF with training-selected threshold, GraphRouter
performance-first / balance / cost-first, Avengers-Pro alpha frontier). All are means over the same
five family-stratified 70/30 splits (seeds 0-4) at the 15-lambda grid (Avengers: over alpha). The paper's
(family, topic) cell lookup (min 50 training rows, family fallback) has no saved curve and is recomputed
here on the same splits. Metrics: AIQ with the exp64 convention (max_cost = priciest operating point of
any method; comparable to the baseline tables) and the RouterBench convention (priciest single model);
AIQ over log10 cost; RouteLLM-style APGR between cheapest and strongest single model; LLMRouterBench
PerfGain / CostSave vs best single. Query-weighted and equal-weight-over-families curves.
Writes results/exp12_indomain_frontier/<panel>/{result.json, result.md, frontier.png}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from rb.panelctx import topic_str
from rb.probes.sweep import LAMBDAS, SEEDS, split, aiq

EXP = "exp12_indomain_frontier"; MIN_CELL = 50
ROOTS = {"routerbench-0shot": (C.E4 / "results" / "basic_model_tests", C.E4 / "results" / "routerbench_baselines_exp64" / "baselines"),
         "llmrouterbench-performance-cost": (C.E4 / "llmrouterbench" / "results" / "performance-cost" / "basic_model_tests", C.E4 / "llmrouterbench" / "results" / "performance-cost" / "baselines")}


def curve_from_by_lambda(bl, agg):
    pts = []
    for lam in LAMBDAS:
        v = bl.get(str(lam)) or bl.get(f"{lam:g}") or bl.get(str(float(lam)))
        if v is None: return None
        a = v["all"][agg]; pts.append([float(a["cost"][0]) if isinstance(a["cost"], list) else float(a["cost"]), float(a["reward"][0]) if isinstance(a["reward"], list) else float(a["reward"])])
    return np.array(pts)


def hull(pts):
    pts = pts[np.argsort(pts[:, 0])]; h = []
    for cc, rr in pts:
        if h and rr < h[-1][1]: continue
        h.append([cc, rr])
    return np.array(h)


def aiq_log(pts, cmin, cmax):
    h = hull(pts); x = np.r_[np.log10(cmin), np.log10(np.clip(h[:, 0], cmin, cmax)), np.log10(cmax)]; y = np.r_[h[0, 1], h[:, 1], h[-1, 1]]; o = np.argsort(x)
    return float(np.trapezoid(y[o], x[o]) / (np.log10(cmax) - np.log10(cmin)))


def aiq_range(pts, cmin, cmax):
    """RouterBench normalisation: hull area over the common cost range [cmin, cmax] / (cmax - cmin), linear cost,
    hull extended flat to both ends of the range."""
    h = hull(pts); x = np.r_[cmin, np.clip(h[:, 0], cmin, cmax), cmax]; y = np.r_[h[0, 1], h[:, 1], h[-1, 1]]; o = np.argsort(x)
    return float(np.trapezoid(y[o], x[o]) / max(cmax - cmin, 1e-12))


def apgr(pts, cw, rw, cs, rs):
    h = hull(pts); fr = np.linspace(0, 1, 101); cc = cw + fr * (cs - cw); rr = np.interp(cc, h[:, 0], h[:, 1], left=h[0, 1], right=h[-1, 1]); return float(np.mean((rr - rw) / max(rs - rw, 1e-9)))


def main():
    panel_name = sys.argv[1]; D = C.load(panel_name); r, c, fam, fn, names = D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, n = len(fn), len(r)
    basic, base64 = ROOTS[panel_name]; curves = {"per_query": {}, "per_family_equal_weight": {}}
    def take(label, path, key="by_lambda"):
        if not Path(path).exists(): print("missing", path); return
        j = json.load(open(path)); bl = j[key] if key in j else None
        if bl is None: return
        for agg in curves:
            cv = curve_from_by_lambda(bl, agg)
            if cv is not None: curves[agg][label] = cv
    refs = json.load(open(basic / "references.json"))["references"]
    for lab in ("best_single", "oracle"):
        for agg in curves: curves[agg][lab] = curve_from_by_lambda(refs[lab], agg)
    for k in (5, 10, 40, 100): take(f"knn{k}", basic / "knn" / f"knn{k}.json")
    take("family_lookup", basic / "best_model_per_family" / "best_model_per_family.json"); take("classify_family_lookup", basic / "classify_per_family_best_model" / "classify_per_family_best_model.json")
    for lab in ("kmeans64_utility_top1", "kmeans64_utility_top3", "routellm_mf_train_selected_threshold", "graphrouter_performance_first", "graphrouter_balance", "graphrouter_cost_first"): take(lab, base64 / lab / "result.json")
    av = base64 / "avengerspro_k64_top1" / "result.json"
    if av.exists(): curves["per_query"]["avengerspro_k64_top1 (alpha frontier)"] = np.array(json.load(open(av))["frontier_mean_over_seeds"])
    # recomputed (family, topic) cell lookup on the same splits
    _, cell = np.unique(np.array([str(t) for t in topic_str(D["panel"])], dtype=object), return_inverse=True); ncell = int(cell.max()) + 1
    pq, pf = [], []
    for lam in LAMBDAS:
        U = r - lam * c; a_q, a_f = [], []
        for seed in SEEDS:
            tr, te = split(n, fam, seed); fb = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)]); cb = np.full(ncell, -1)
            for k in np.unique(cell[tr]):
                m = cell[tr] == k
                if m.sum() >= MIN_CELL: cb[k] = U[tr][m].mean(0).argmax()
            ch = cb[cell[te]]; ch = np.where(ch < 0, fb[fam[te]], ch); cc, rr = c[te, ch], r[te, ch]
            a_q.append([cc.mean(), rr.mean()]); a_f.append([np.mean([cc[fam[te] == g].mean() for g in range(F)]), np.mean([rr[fam[te] == g].mean() for g in range(F)])])
        pq.append(np.mean(a_q, 0)); pf.append(np.mean(a_f, 0))
    curves["per_query"]["cell_lookup (recomputed)"] = np.array(pq); curves["per_family_equal_weight"]["cell_lookup (recomputed)"] = np.array(pf)
    cbar, rbar = c.mean(0), r.mean(0); strong, weak = int(rbar.argmax()), int(cbar.argmin()); bs = int(rbar.argmax()) if panel_name == "routerbench-0shot" else int(rbar.argmax())
    out = {"aiq_exp64": {}, "aiq_routerbench": {}, "aiq_log": {}, "apgr": {}, "perfgain": {}, "costsave": {}, "aiq_macro_exp64": {}, "curves": {}}
    mc64 = max(float(cv[:, 0].max()) for cv in curves["per_query"].values()); mc_rb = float(cbar.max())
    mc64_f = max(float(cv[:, 0].max()) for cv in curves["per_family_equal_weight"].values())
    bsr, bsc = float(rbar[bs]), float(cbar[bs])
    out["cost_blind"] = {}
    for m, cv in curves["per_query"].items():
        blind = bool(np.allclose(cv, cv[0], atol=1e-9)); out["cost_blind"][m] = blind   # one operating point: AIQ not applicable (exp64 rule)
        out["aiq_exp64"][m] = float("nan") if blind else aiq(cv, mc64); out["aiq_routerbench"][m] = float("nan") if blind else aiq(cv, mc_rb); out["aiq_log"][m] = float("nan") if blind else aiq_log(cv, float(cbar.min()), mc_rb)
        out.setdefault("aiq_range", {})[m] = float("nan") if blind else aiq_range(cv, float(cbar.min()), mc_rb)
        out["apgr"][m] = apgr(cv, float(cbar[weak]), float(rbar[weak]), float(cbar[strong]), float(rbar[strong])); out["perfgain"][m] = float(cv[:, 1].max() / bsr - 1)
        ok = cv[cv[:, 1] >= bsr - 1e-12]; out["costsave"][m] = (float(1 - ok[:, 0].min() / bsc) if len(ok) else None); out["curves"][m] = cv.tolist()
    for m, cv in curves["per_family_equal_weight"].items(): out["aiq_macro_exp64"][m] = aiq(cv, mc64_f)
    out["curves_macro"] = {m: cv.tolist() for m, cv in curves["per_family_equal_weight"].items()}
    # LLMRouterBench convention: PerfGain / CostSave on DATASET-AVERAGED accuracy and cost (macro curves), reference = best single by macro accuracy
    rmac = np.array([np.mean([r[fam == g, j].mean() for g in range(F)]) for j in range(len(names))]); cmac = np.array([np.mean([c[fam == g, j].mean() for g in range(F)]) for j in range(len(names))])
    bsm = int(rmac.argmax()); out["perfgain_macro"] = {}; out["costsave_macro"] = {}; out["best_single_macro_reference"] = names[bsm]
    for m, cv in curves["per_family_equal_weight"].items():
        out["perfgain_macro"][m] = float(cv[:, 1].max() / rmac[bsm] - 1); ok = cv[cv[:, 1] >= rmac[bsm] - 1e-12]; out["costsave_macro"][m] = (float(1 - ok[:, 0].min() / cmac[bsm]) if len(ok) else None)
    res = C.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMBDAS), "max_cost_exp64": mc64, "max_cost_routerbench": mc_rb, "best_single_reference": names[bs], **out})
    order = sorted(curves["per_query"], key=lambda m: -(out["aiq_exp64"][m] if not np.isnan(out["aiq_exp64"][m]) else -1))
    md = [f"# {EXP} ({panel_name})", "", f"In-domain (5 splits) frontiers from saved curves; cell lookup recomputed. AIQ exp64 convention (max_cost = priciest operating point, {mc64*1e3:.2f}e-3) and RouterBench convention (priciest single model, {mc_rb*1e3:.2f}e-3); AIQ_log over log10 cost; APGR between cheapest ({names[weak]}) and strongest ({names[strong]}) single model; PerfGain/CostSave vs best single ({names[bs]}).", "",
          "method | AIQ (repo: /max cost) | AIQ (RouterBench: hull area / cost range, cheapest..priciest model) | AIQ_log | APGR (adapted) | PerfGain | CostSave | AIQ macro (repo)", "---|---|---|---|---|---|---|---"]
    for m in order:
        md.append(f"{m} | {out['aiq_exp64'][m]:.4f} | {out['aiq_range'][m]:.4f} | {out['aiq_log'][m]:.4f} | {out['apgr'][m]:.3f} | {out['perfgain'][m]:+.3f} | " + (f"{out['costsave'][m]:.3f}" if out["costsave"][m] is not None else "n/a") + " | " + (f"{out['aiq_macro_exp64'][m]:.4f}" if m in out["aiq_macro_exp64"] else "n/a"))
    md += ["", "## LLMRouterBench convention (dataset-averaged accuracy and cost; reference best single by macro accuracy = " + names[bsm] + ")", "", "method | PerfGain (macro) | CostSave (macro)", "---|---|---"]
    for m in order:
        if m in out["perfgain_macro"]: md.append(f"{m} | {out['perfgain_macro'][m]:+.3f} | " + (f"{out['costsave_macro'][m]:.3f}" if out["costsave_macro"][m] is not None else "n/a"))
    md += ["", "## frontier points, query-weighted (cost e-3, reward) at lambda = " + ", ".join(f"{l:g}" for l in LAMBDAS) + " (Avengers: over alpha)", ""]
    for m in order: md.append(f"- {m}: " + " ".join(f"({p[0]*1e3:.2f},{p[1]:.3f})" for p in out["curves"][m]))
    d = C.RESULTS / EXP / panel_name; C.save(d, res, md)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, agg, title in ((axs[0], "per_query", "query-weighted"), (axs[1], "per_family_equal_weight", "equal weight over families")):
            for m in order:
                if m not in curves[agg]: continue
                p = curves[agg][m]; ax.plot(p[:, 0] * 1e3, p[:, 1], marker="o", ms=3, lw=1, label=m)
            ax.set_xscale("log"); ax.set_xlabel("mean cost per query (e-3 USD, log)"); ax.set_ylabel("mean reward"); ax.set_title(f"{panel_name}: in-domain, {title}"); ax.grid(alpha=.3)
        axs[0].legend(fontsize=6, loc="lower right"); fig.tight_layout(); fig.savefig(d / "frontier.png", dpi=150)
    except Exception as e: print("figure skipped:", e)


if __name__ == "__main__":
    main()
