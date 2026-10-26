"""exp10 - cost-quality frontier and AIQ view of the held-out routers (the literature's currency).

Same strict held-out-family protocol as exp07. For every method and every lambda of the repository's
15-point grid (rb.probes.sweep.LAMBDAS), the router's decision on the held-out family gives one
(mean cost, mean reward) operating point. Frontier = those 15 points; AIQ = RouterBench's area under
the non-decreasing convex hull of the points divided by max_cost (rb.probes.sweep.aiq), with
max_cost = the mean cost of the most expensive single model on the same rows (RouterBench convention).
Two aggregations: query-weighted (points pooled over all held-out rows) and macro (curve averaged
over families, then AIQ). Also LLMRouterBench-style numbers: PerfGain = best-accuracy point / best
single - 1 and CostSave = 1 - cost of the cheapest point with reward >= best single's reward / cost of
best single (best single = family-balanced training best; both on the pooled held-out rows).
Methods (all online unless marked): safe (family-balanced best single), kNN-100, soft prior, verbosity
router (soft reward + rank-1 axis cost, raw z=0) with GTE / reader-L21 / 64-token draft features,
cost oracle (target family's true mean cost; diagnostic), family ceiling (non-deployable), oracle.
Writes results/exp10_frontier_aiq/<panel>/{result.json, result.md, frontier.png}.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp05_verbosity_axis_router import axes, coords, fit_scalar
from rb.probes.sweep import LAMBDAS, aiq

EXP = "exp10_frontier_aiq"
READER = {"routerbench-0shot": "features/routerbench-0shot/prompt/hidden/qwen2.5-7b-instruct/L21_last.npy",
          "llmrouterbench-performance-cost": "features/llmrouterbench-performance-cost/prompt/hidden/qwen2.5-7b-instruct/L21_last.npy"}


def features(D, feat, panel_name):
    ln = np.log1p(D["n_in"])[:, None].astype(np.float32)
    if feat == "gtel": return np.hstack([D["x"], ln])
    if feat == "reader":
        h = np.load(C.E4 / READER[panel_name], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(D["r"]) else h, dtype=np.float32); return np.hstack([h, ln])
    if feat == "drafts":
        p = C.RESULTS / "drafts" / panel_name / "stats.npz"
        if not p.exists(): return None
        z = np.load(p); return np.column_stack([np.log1p(z["draft_len"]), z["finished"].astype(np.float32), z["first_entropy"], z["first_margin"], z["mean_logprob"], z["mean_entropy"], ln[:, 0]]).astype(np.float32)
    raise ValueError(feat)


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name)
    x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M, n = len(fn), len(names), len(r)
    Y = np.log(c + C.EPS); xu = C.unit(x)
    feats = {k: features(D, k, panel_name) for k in ("gtel", "reader", "drafts")}; feats = {k: v for k, v in feats.items() if v is not None}
    methods = ["safe", "knn100", "soft_prior"] + [f"verbosity_{k}" for k in feats] + ["cost_oracle", "family_ceiling", "oracle"]
    choice = {m: {} for m in methods}     # method -> family -> (n_te, n_lam) model index
    for g in range(F):
        te = np.flatnonzero(fam == g); tr = np.flatnonzero(fam != g); S = [s for s in range(F) if s != g]
        w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
        rt, ct = C.family_tables(r, c, fam, S); logct = np.log(ct + C.EPS); alpha, B, G = axes(logct, 1); v = coords(Y[tr], alpha, B)[:, 0]
        P = C.fit_posterior(x, fam, S, te, True, 42); Pr, Pc = P @ rt, P @ ct
        lo, hi = G.min(), G.max(); span = hi - lo
        logc_hat = {}
        for k, X in feats.items():
            gh = np.clip(fit_scalar(X[tr], v, w)(X[te])[:, 0], lo - .25 * span, hi + .25 * span); logc_hat[k] = alpha[None, :] + gh[:, None] * B[0][None, :]
        src = np.flatnonzero(fam != g); nb = src[C.knn_mean(xu[te], xu[src], 100)]; nr, nc = r[nb].mean(1), c[nb].mean(1)
        rbar, cbar = r[te].mean(0), c[te].mean(0)
        for m in methods: choice[m][g] = np.zeros((len(te), len(LAMBDAS)), dtype=np.int64)
        for li, lam in enumerate(LAMBDAS):
            U = r[te] - lam * c[te]
            choice["safe"][g][:, li] = int((rt - lam * ct).mean(0).argmax()); choice["knn100"][g][:, li] = (nr - lam * nc).argmax(1)
            choice["soft_prior"][g][:, li] = (Pr - lam * Pc).argmax(1)
            for k in feats: choice[f"verbosity_{k}"][g][:, li] = (Pr - lam * np.exp(logc_hat[k])).argmax(1)
            choice["cost_oracle"][g][:, li] = (Pr - lam * cbar[None, :]).argmax(1); choice["family_ceiling"][g][:, li] = int((rbar - lam * cbar).argmax()); choice["oracle"][g][:, li] = U.argmax(1)
        print(fn[g], f"({time.time()-t0:.0f}s)", flush=True)
    # operating points
    rows_all = np.arange(n); max_cost_qw = float(c.mean(0).max()); best_single_qw = int(np.mean([r[fam == g].mean(0) for g in range(F)], 0).argmax())
    out = {"aiq_qw": {}, "aiq_macro": {}, "frontier_qw": {}, "frontier_macro": {}, "perfgain": {}, "costsave": {}}
    fam_te = {g: np.flatnonzero(fam == g) for g in range(F)}
    for m in methods:
        pts_q = []; pts_f = {g: [] for g in range(F)}
        for li in range(len(LAMBDAS)):
            cc = np.zeros(n); rr = np.zeros(n)
            for g in range(F):
                te = fam_te[g]; ch = choice[m][g][:, li]; cc[te] = c[te, ch]; rr[te] = r[te, ch]; pts_f[g].append([cc[te].mean(), rr[te].mean()])
            pts_q.append([cc.mean(), rr.mean()])
        pts_q = np.array(pts_q); out["frontier_qw"][m] = pts_q.tolist(); out["aiq_qw"][m] = aiq(pts_q, max_cost_qw)
        curves = np.stack([np.array(pts_f[g]) for g in range(F)]); pts_m = curves.mean(0); out["frontier_macro"][m] = pts_m.tolist()
        out.setdefault("frontier_per_family", {})[m] = {fn[g]: curves[g].tolist() for g in range(F)}
        max_cost_macro = float(np.mean([c[fam == g].mean(0).max() for g in range(F)])); out["aiq_macro"][m] = aiq(pts_m, max_cost_macro)
        bs_r, bs_c = r[:, best_single_qw].mean(), c[:, best_single_qw].mean()
        out["perfgain"][m] = float(pts_q[:, 1].max() / bs_r - 1); ok = pts_q[pts_q[:, 1] >= bs_r - 1e-12]
        out["costsave"][m] = float(1 - ok[:, 0].min() / bs_c) if len(ok) else None
    res = C.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMBDAS), "max_cost_qw": max_cost_qw, "best_single_for_perfgain": names[best_single_qw], **out})
    md = [f"# {EXP} ({panel_name})", "", "Held-out family protocol; one operating point per lambda of the 15-point grid; AIQ = RouterBench area under the convex hull / max_cost (most expensive single model). PerfGain/CostSave follow LLMRouterBench (reference = family-balanced best single, " + names[best_single_qw] + ").", "",
          "method | AIQ query-weighted | AIQ macro | PerfGain | CostSave", "---|---|---|---|---"]
    for m in methods: md.append(f"{m} | {out['aiq_qw'][m]:.4f} | {out['aiq_macro'][m]:.4f} | {out['perfgain'][m]:+.3f} | " + (f"{out['costsave'][m]:.3f}" if out["costsave"][m] is not None else "n/a"))
    md += ["", "## frontier points, query-weighted (cost e-3, reward) at lambda = " + ", ".join(f"{l:g}" for l in LAMBDAS), ""]
    for m in methods: md.append(f"- {m}: " + " ".join(f"({p[0]*1e3:.2f},{p[1]:.3f})" for p in out["frontier_qw"][m]))
    C.save(C.RESULTS / EXP / panel_name, res, md)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, key, title in ((axs[0], "frontier_qw", "query-weighted"), (axs[1], "frontier_macro", "macro over families")):
            for m in methods:
                p = np.array(out[key][m]); ax.plot(p[:, 0] * 1e3, p[:, 1], marker="o", ms=3, lw=1, label=m)
            ax.set_xscale("log"); ax.set_xlabel("mean cost per query (e-3 USD, log)"); ax.set_ylabel("mean reward"); ax.set_title(f"{panel_name}: {title}"); ax.grid(alpha=.3)
        axs[1].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(C.RESULTS / EXP / panel_name / "frontier.png", dpi=150)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
