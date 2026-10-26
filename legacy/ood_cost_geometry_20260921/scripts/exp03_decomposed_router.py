"""exp03 - decomposed transfer router with a pessimistic-cost guard, strict held-out family, nested selection.

Decision on a target query x for lambda:
    argmax_m  R_m(x)  -  lam * Cpess_m(x)
  R_m(x)      reward side: "soft"  = P(x) @ rt[:, m]   (class-balanced GTE posterior over source families)
                           "balanced" = mean_s rt[s, m]   (family-balanced global prior)
  C_m(x)      cost side: log C = beta * log chat_m(x) + (1 - beta) * log (P(x) @ ct[:, m])
              chat_m(x) = ridge prediction of log cost from GTE + log(1 + n_in), fitted on source rows
              beta = 0 recovers the held-out study's soft prior exactly; beta = 1 is fully predicted cost
  Cpess_m(x)  = exp(log C_m(x) + z * sigma_m)   with sigma_m the transfer residual std of the cost
              predictor measured by inner leave-one-source-family-out (no target information)
Grid: reward in {soft, balanced}, beta in {0, .5, 1}, z in {0, 1, 2}. For every outer fold and every
lambda the configuration is chosen by inner LOFO over the source families, lexicographically worst
family then mean (the held-out study's rule). Also reported with "online" strictness: no target
outcomes, labels, or batch statistics are used anywhere.
References: family-balanced best single, soft prior (beta=0,z=0,soft), kNN-100, in-domain family
ceiling, oracle. PROVES what predicting the cost regime from text buys on an unseen family.
"""
from __future__ import annotations
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import common as C

EXP = "exp03_decomposed_router"; ALPHAS = np.logspace(-1, 5, 13)
REWARDS = ("soft", "balanced"); BETAS = (0.0, 0.5, 1.0); ZS = (0.0, 1.0, 2.0)
GRID = list(itertools.product(REWARDS, BETAS, ZS))


def fit_cost(X, Y, tr, fam):
    w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
    sc = StandardScaler().fit(X[tr]); m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(X[tr]), Y[tr], sample_weight=w)
    return lambda rows: m.predict(sc.transform(X[rows]))


def decide(cfg, lam, Pr, Rbal, logPc, logchat, sigma):
    reward, beta, z = cfg
    R = Pr if reward == "soft" else Rbal[None, :]
    logC = beta * logchat + (1 - beta) * logPc + z * sigma[None, :]
    return (R - lam * np.exp(logC)).argmax(1)


def block(D, X, Y, sources, target_rows, sigma, seed=42):
    """Everything needed to decide on target_rows using only the given source families."""
    x, r, c, fam = D["x"], D["r"], D["c"], D["fam"]
    tr = np.flatnonzero(np.isin(fam, sources)); rt, ct = C.family_tables(r, c, fam, sources)
    P = C.fit_posterior(x, fam, sources, target_rows, True, seed)
    pred = fit_cost(X, Y, tr, fam)(target_rows)
    return {"Pr": P @ rt, "Rbal": rt.mean(0), "logPc": np.log(P @ ct + C.EPS), "logchat": pred, "sigma": sigma, "resid": Y[target_rows] - pred}


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name)
    x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M = len(fn), len(names)
    X = np.hstack([x, np.log1p(D["n_in"])[:, None].astype(np.float32)]); Y = np.log(c + C.EPS); xu = C.unit(x)
    methods = ["safe", "soft_prior", "knn100", "decomposed", "decomposed_fixed_soft_b1_z1", "family_ceiling", "oracle"]
    per = {m: {str(l): {} for l in C.LAMS} for m in methods}; n_by = {}; selected = {}; sigma_out = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        # inner LOFO: blocks for each inner target s using sources S \ {s}; sigma from inner residuals (two-pass: residuals first)
        inner = {}
        for s in S:
            Si = [q for q in S if q != s]; rows = np.flatnonzero(fam == s)
            inner[s] = block(D, X, Y, Si, rows, np.zeros(M))
        sigma = np.sqrt(np.mean(np.concatenate([inner[s]["resid"] ** 2 for s in S], axis=0), axis=0))
        for s in S: inner[s]["sigma"] = sigma
        sigma_out[fn[g]] = {names[j]: float(sigma[j]) for j in range(M)}
        outer = block(D, X, Y, S, te, sigma)
        src_rows = np.flatnonzero(fam != g); nb = src_rows[C.knn_mean(xu[te], xu[src_rows], 100)]; nr, nc = r[nb].mean(1), c[nb].mean(1)
        rt, ct = C.family_tables(r, c, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0)
        selected[fn[g]] = {}
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te))
            # inner selection
            score = {}
            for cfg in GRID:
                vals = []
                for s in S:
                    rows = np.flatnonzero(fam == s); Us = r[rows] - lam * c[rows]; ch = decide(cfg, lam, **{k: inner[s][k] for k in ("Pr", "Rbal", "logPc", "logchat", "sigma")})
                    vals.append(Us[np.arange(len(rows)), ch].mean())
                v = np.sort(vals); score[cfg] = (float(v[0]), float(v.mean()))
            best = max(GRID, key=lambda k: score[k]); selected[fn[g]][L] = {"config": list(best), "inner_worst_mean": score[best]}
            args = {k: outer[k] for k in ("Pr", "Rbal", "logPc", "logchat", "sigma")}
            ch = {"safe": np.full(len(te), int((rt - lam * ct).mean(0).argmax())),
                  "soft_prior": decide(("soft", 0.0, 0.0), lam, **args),
                  "knn100": (nr - lam * nc).argmax(1),
                  "decomposed": decide(best, lam, **args),
                  "decomposed_fixed_soft_b1_z1": decide(("soft", 1.0, 1.0), lam, **args),
                  "family_ceiling": np.full(len(te), int((rbar - lam * cbar).argmax())),
                  "oracle": U.argmax(1)}
            for m in methods: per[m][L][fn[g]] = float(U[ii, ch[m]].mean())
        print(f"{fn[g]:20s} lam100: safe {per['safe']['100.0'][fn[g]]:.3f} soft {per['soft_prior']['100.0'][fn[g]]:.3f} decomposed {per['decomposed']['100.0'][fn[g]]:.3f} (cfg {selected[fn[g]]['100.0']['config']}) fixed {per['decomposed_fixed_soft_b1_z1']['100.0'][fn[g]]:.3f} ceiling {per['family_ceiling']['100.0'][fn[g]]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in methods}
    wtl = {L: {"wins": int(sum(per["decomposed"][L][f] > per["soft_prior"][L][f] + 1e-9 for f in fn)), "losses": int(sum(per["decomposed"][L][f] < per["soft_prior"][L][f] - 1e-9 for f in fn)),
               "worst_delta": float(min(per["decomposed"][L][f] - per["soft_prior"][L][f] for f in fn))} for L in agg["decomposed"]}
    res = C.provenance(D); res.update({"experiment": EXP, "grid": [list(g_) for g_ in GRID], "per_family": per, "aggregate_qw_macro": agg, "selected": selected, "sigma_transfer_residual_std": sigma_out, "decomposed_vs_soft_prior": wtl})
    md = [f"# {EXP} ({panel_name})", "", "Strict held-out family; nested inner-LOFO selection (worst family, then mean) of reward side {soft, balanced} x cost blend beta {0,.5,1} x pessimism z {0,1,2}. Query-weighted [macro].", "",
          "lambda | " + " | ".join(methods) + " | decomposed vs soft: W/L (worst delta)", "---|" + "---|" * (len(methods) + 1)]
    for lam in C.LAMS:
        L = str(lam); md.append(f"{lam:g} | " + " | ".join(f"{agg[m][L][0]:.3f} [{agg[m][L][1]:.3f}]" for m in methods) + f" | {wtl[L]['wins']}/{wtl[L]['losses']} ({wtl[L]['worst_delta']:+.3f})")
    for lam in (10.0, 100.0, 1000.0):
        L = str(lam); md += ["", f"## per family, lambda = {lam:g}  (selected config)", "", "family | " + " | ".join(methods) + " | config", "---|" + "---|" * (len(methods) + 1)]
        for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{per[m][L][fn[g]]:.3f}" for m in methods) + f" | {selected[fn[g]][L]['config']}")
    md += ["", "## transfer residual std of log cost (inner LOFO), per model, by outer fold", ""] + [f"- {f}: " + ", ".join(f"{k} {v:.2f}" for k, v in sigma_out[f].items()) for f in fn]
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
