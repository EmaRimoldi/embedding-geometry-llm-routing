"""exp26 - per-query cost prediction under held-out family: which representation and which structure transfer?
Targets Y = log(cost + eps), one column per model. Cost models, all fitted on training families only:
  transferred : soft-prior cost p.C (baseline; no regression)
  axis1       : the current router - rank-1 FAMILY-table axis, query coordinate regressed from features
  factor_k    : rank-k PER-QUERY factor model: Y_tr - mu -> SVD -> k components; ridge features -> k scores; k in 1,2,3
  full        : direct per-model ridge (M outputs)
Features: lenonly (log n_in), gtel (GTE + log n_in), reader (Qwen2.5-7B L21 last token + log n_in), drafts (64-token draft
stats + log n_in, where available). Scores on the held-out family: R^2 of log cost per model (mean; and per model group
'thinking' vs other on LLMRouterBench), and routing utility with soft-prior reward at C.LAMS; references: soft prior,
family-mean cost oracle, per-query cost oracle, family ceiling, oracle. Also an in-domain random-split R^2 (seed 0) as a
capacity reference. Usage: exp26 <panel>."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import common as C
from exp05_verbosity_axis_router import axes, coords
from exp07_calibrated_verbosity_router import READER

EXP = "exp26_cost_regression"; ALPHAS = (1.0, 10.0, 100.0, 1000.0); KS = (1, 2, 3)
THINK = ("gpt-5", "gemini-2.5-pro", "qwen3-235b-a22b-thinking-2507", "deepseek-r1-0528")


def feats(D, panel_name):
    x, r = D["x"], D["r"]; ln = np.log1p(D["n_in"])[:, None].astype(np.float32); out = {"lenonly": ln, "gtel": np.hstack([x, ln])}
    h = np.load(C.E4 / READER[panel_name], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(r) else h, dtype=np.float32); out["reader"] = np.hstack([h, ln])
    p = C.RESULTS / "drafts" / panel_name / "stats.npz"
    if p.exists():
        z = np.load(p); out["drafts"] = np.column_stack([np.log1p(z["draft_len"]), z["finished"].astype(np.float32), z["first_entropy"], z["first_margin"], z["mean_logprob"], z["mean_entropy"], ln[:, 0]]).astype(np.float32)
    return out


def ridge(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), Ytr); return m.predict(sc.transform(Xte)).reshape(len(Xte), -1)


def predictors(X, Y, tr, te, fam, S, r, c):
    """dict name -> predicted log-cost matrix on te"""
    out = {}; mu = Y[tr].mean(0)
    rt, ct = C.family_tables(r, c, fam, S); alpha, B, G = axes(np.log(ct + C.EPS), 1); v = coords(Y[tr], alpha, B)[:, 0]
    g = ridge(X[tr], v, X[te])[:, 0]; lo, hi = G.min(), G.max(); g = np.clip(g, lo - .25 * (hi - lo), hi + .25 * (hi - lo)); out["axis1"] = alpha[None, :] + g[:, None] * B[0][None, :]
    Yc = Y[tr] - mu; U_, s_, Vt = np.linalg.svd(Yc, full_matrices=False)
    for k in KS:
        V = Vt[:k].T; z = Yc @ V; zh = ridge(X[tr], z, X[te]); out[f"factor{k}"] = mu[None, :] + zh @ V.T
    out["full"] = ridge(X[tr], Y[tr], X[te]); return out


def r2(Yhat, Yte):
    ss = ((Yte - Yte.mean(0)) ** 2).sum(0); return 1 - ((Yte - Yhat) ** 2).sum(0) / np.maximum(ss, 1e-12)


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name); x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M, n = len(fn), len(names), len(r)
    FE = feats(D, panel_name); Y = np.log(c + C.EPS); think = np.array([any(t in nm for t in THINK) for nm in names])
    est_names = ["axis1"] + [f"factor{k}" for k in KS] + ["full"]
    R2 = {f: {e: [] for e in est_names} for f in FE}; R2T = {f: {e: [] for e in est_names} for f in FE}; R2ref = {"transferred": [], "family_mean_oracle": []}
    per = {}; n_by = {}
    def rec(m, L, g, v): per.setdefault(m, {}).setdefault(L, {})[fn[g]] = float(v)
    for g in range(F):
        te = np.flatnonzero(fam == g); tr = np.flatnonzero(fam != g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te); Yte = Y[te]
        P = C.fit_posterior(x, fam, S, te, True, 42); rt, ct = C.family_tables(r, c, fam, S); Pr = P @ rt; trans = np.log(P @ ct + C.EPS); fm = np.log(c[te].mean(0) + C.EPS)[None, :].repeat(len(te), 0)
        R2ref["transferred"].append(r2(trans, Yte)); R2ref["family_mean_oracle"].append(r2(fm, Yte))
        preds = {}
        for f, X in FE.items():
            pr = predictors(X, Y, tr, te, fam, S, r, c)
            for e, Yh in pr.items(): R2[f][e].append(r2(Yh, Yte)); preds[f"{f}:{e}"] = Yh
            print(fn[g], f, "done", round(time.time() - t0), flush=True)
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te))
            ch = {"soft_prior": (Pr - lam * np.exp(trans)).argmax(1), "cost_oracle_family_mean": (Pr - lam * c[te].mean(0)[None, :]).argmax(1), "cost_oracle_per_query": (Pr - lam * c[te]).argmax(1), "family_ceiling": np.full(len(te), int((r[te].mean(0) - lam * c[te].mean(0)).argmax())), "oracle": U.argmax(1), "safe": np.full(len(te), int((rt - lam * ct).mean(0).argmax()))}
            for k, Yh in preds.items(): ch[k] = (Pr - lam * np.exp(Yh)).argmax(1)
            for m, sel in ch.items(): rec(m, L, g, U[ii, sel].mean())
    # in-domain capacity reference (random split, seed 0)
    tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=fam); indom = {}
    for f, X in FE.items(): indom[f] = {e: float(np.mean(r2(Yh, Y[te]))) for e, Yh in predictors(X, Y, tr, te, fam, list(range(F)), r, c).items()}
    indom["family_mean"] = float(np.mean(r2(np.stack([np.log(c[tr][fam[tr] == fam[i]].mean(0) + C.EPS) for i in te]), Y[te])))
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in per}
    res = C.provenance(D); res.update({"experiment": EXP, "features": list(FE), "estimators": est_names, "r2_heldout_mean_over_families": {f: {e: float(np.mean(R2[f][e])) for e in est_names} for f in FE}, "r2_heldout_thinking_models": {f: {e: float(np.mean([v[think] for v in R2[f][e]])) if think.any() else None for e in est_names} for f in FE},
                "r2_heldout_other_models": {f: {e: float(np.mean([v[~think] for v in R2[f][e]])) for e in est_names} for f in FE}, "r2_reference": {k: float(np.mean(v)) for k, v in R2ref.items()}, "r2_indomain": indom, "per_family": per, "aggregate_qw_macro": agg})
    md = [f"# {EXP} ({panel_name})", "", f"Held-out family; targets log cost per model; R^2 averaged over models and held-out families (negative = worse than predicting the held-out family's mean, which is itself unavailable OOD). Thinking models: {[nm for nm, t in zip(names, think) if t]}.", "",
          "## R^2 of log cost on the held-out family (mean over models) | thinking models | other models", "", "features | " + " | ".join(est_names), "---|" + "---|" * len(est_names)]
    for f in FE: md.append(f"{f} | " + " | ".join(f"{res['r2_heldout_mean_over_families'][f][e]:.2f}" + (f" ({res['r2_heldout_thinking_models'][f][e]:.2f} / {res['r2_heldout_other_models'][f][e]:.2f})" if think.any() else "") for e in est_names))
    md += [f"reference: transferred p.C {res['r2_reference']['transferred']:.2f}; held-out family mean (oracle) {res['r2_reference']['family_mean_oracle']:.2f}", "", "## in-domain (random 70/30) R^2, capacity reference", "", "features | " + " | ".join(est_names), "---|" + "---|" * len(est_names)] + [f"{f} | " + " | ".join(f"{indom[f][e]:.2f}" for e in est_names) for f in FE] + [f"family mean {indom['family_mean']:.2f}", ""]
    md += ["## routing utility, soft-prior reward + estimated cost; query-weighted [macro]", "", "cost estimate | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)]
    order = ["safe", "soft_prior"] + [k for k in per if ":" in k] + ["cost_oracle_family_mean", "cost_oracle_per_query", "family_ceiling", "oracle"]
    for m in order: md.append(f"{m} | " + " | ".join(f"{agg[m][str(l)][0]:.3f} [{agg[m][str(l)][1]:.3f}]" for l in C.LAMS))
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
