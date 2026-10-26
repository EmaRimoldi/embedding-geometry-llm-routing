"""exp05 - the verbosity-axis router: one scalar per query on the cost axis, model-specific slopes.

Cost structure (measured 2026-09-21, log mean-cost tables): RB additive price+verbosity R2 0.986;
LRB additive 0.62, rank-1 with model-specific slopes 0.82, rank-2 0.96. So the cost of every model on
a family is  log c[s, m] ~ alpha_m + b_m * gamma_s  (+ a second axis on LRB), and for an unseen
family only gamma (one or two scalars) has to be predicted from the text.

For every outer held-out family g (sources S):
  alpha_m  = mean over S of log ct[s, m]           (price)
  b, gamma = rank-r SVD of the centred source table (r = 1 or 2; b_m slopes, gamma_s family scores)
  v(x)     = least-squares coordinate(s) of the query's own log-cost vector on the axes (source rows)
  ghat(x)  = ridge from GTE + log(1+n_in) to v(x), fitted on source rows with equal family weight,
             clipped to the source range of gamma_s widened by 25%
  cost estimators:
     transferred  P(x) @ ct                                            (the held-out study)
     verb1        exp(alpha_m + b_m * ghat(x))
     verb2        rank-2 version
     verb_nn      the cost row of the source family whose gamma_s is nearest to ghat(x) (copies the
                  whole row, so rank-2+ residuals are kept for that family)
  pessimism z in {0, 1, 2}: ghat -> ghat + z * sigma * sign(b) applied on the cost axis, sigma = std
             of the transfer error of ghat measured by inner LOFO (no target information)
  reward side: soft posterior P(x) @ rt ("soft") or family-balanced rt.mean(0) ("balanced")
Nested selection per outer fold and lambda over {estimator} x {z} x {reward}, lexicographic worst
inner family then mean. References: safe, soft prior, cost oracle (family mean), family ceiling,
oracle. Fixed rows (no selection): verb1/soft/z=0 and verb_nn/soft/z=0. Transductive batch-mean
variants of verb1 are reported separately. PROVES whether predicting the cost axis coordinate closes
the held-out loss that exp01 attributed to cost.
"""
from __future__ import annotations
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import common as C

EXP = "exp05_verbosity_axis_router"; ALPHAS = np.logspace(-1, 5, 13)
ESTS = ("transferred", "verb1", "verb2", "verb_nn"); ZS = (0.0, 1.0, 2.0); REWARDS = ("soft", "balanced")
GRID = list(itertools.product(ESTS, ZS, REWARDS))


def axes(logct, rank):
    alpha = logct.mean(0); Tc = logct - alpha[None, :]
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    B = (Vt[:rank] * S[:rank, None]) / np.sqrt(len(logct))        # rank x M slopes
    G = U[:, :rank] * np.sqrt(len(logct))                           # S x rank family scores
    return alpha, B, G


def coords(Y, alpha, B):
    """Least-squares coordinates of each row of Y - alpha on the axes B (rank x M)."""
    A = B @ B.T; return np.linalg.solve(A, B @ (Y - alpha[None, :]).T).T


def fit_scalar(Xtr, ytr, w):
    sc = StandardScaler().fit(Xtr); m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr, sample_weight=w)
    return lambda X: m.predict(sc.transform(X)).reshape(len(X), -1)


def build(D, X, Y, S, target_rows, sigma):
    x, r, c, fam = D["x"], D["r"], D["c"], D["fam"]; tr = np.flatnonzero(np.isin(fam, S))
    w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
    rt, ct = C.family_tables(r, c, fam, S); logct = np.log(ct + C.EPS)
    P = C.fit_posterior(x, fam, S, target_rows, True, 42)
    out = {"Pr": P @ rt, "Rbal": rt.mean(0), "log_transferred": np.log(P @ ct + C.EPS), "ct_rows": logct, "sigma": sigma}
    for rank in (1, 2):
        alpha, B, G = axes(logct, rank); v = coords(Y[tr], alpha, B)
        pred = fit_scalar(X[tr], v, w)(X[target_rows])
        lo, hi = G.min(0), G.max(0); span = hi - lo; pred = np.clip(pred, lo - .25 * span, hi + .25 * span)
        out[f"rank{rank}"] = {"alpha": alpha, "B": B, "G": G, "ghat": pred, "v_true_mean": coords(Y[target_rows], alpha, B).mean(0)}
    return out


def est_logcost(blk, est, z):
    if est == "transferred": return blk["log_transferred"]
    rk = blk["rank2"] if est == "verb2" else blk["rank1"]; alpha, B, ghat = rk["alpha"], rk["B"], rk["ghat"]
    sig = blk["sigma"][: B.shape[0]]
    if est == "verb_nn":
        G = rk["G"]; d = ((ghat[:, None, :] - G[None, :, :]) ** 2).sum(-1); return blk["ct_rows"][d.argmin(1)]
    # pessimistic: move along each axis in the direction that raises cost for each model separately
    logc = alpha[None, :] + ghat @ B + z * (np.abs(B) * sig[:, None]).sum(0)[None, :]
    return logc


def decide(cfg, lam, blk):
    est, z, reward = cfg; R = blk["Pr"] if reward == "soft" else blk["Rbal"][None, :]
    return (R - lam * np.exp(est_logcost(blk, est, z))).argmax(1)


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name)
    x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M = len(fn), len(names)
    X = np.hstack([x, np.log1p(D["n_in"])[:, None].astype(np.float32)]); Y = np.log(c + C.EPS)
    methods = ["safe", "soft_prior", "verbosity_selected", "verb1_soft_z0", "verb_nn_soft_z0", "verb1_soft_batch", "cost_oracle", "family_ceiling", "oracle"]
    per = {m: {str(l): {} for l in C.LAMS} for m in methods}; n_by = {}; selected = {}; diag = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        inner = {s: build(D, X, Y, [q for q in S if q != s], np.flatnonzero(fam == s), np.zeros(2)) for s in S}
        sigma = np.array([np.sqrt(np.mean([(inner[s][f"rank{k}"]["ghat"].mean(0)[:k] - inner[s][f"rank{k}"]["v_true_mean"][:k])[k - 1] ** 2 for s in S])) for k in (1, 2)])
        # sigma[k-1] = rms error of the family-mean coordinate on axis k across inner folds; used for both ranks
        sig2 = np.array([sigma[0], sigma[1]])
        for s in S: inner[s]["sigma"] = sig2
        outer = build(D, X, Y, S, te, sig2)
        rt, ct = C.family_tables(r, c, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0)
        diag[fn[g]] = {"gamma_sources_rank1": {fn[s]: float(outer["rank1"]["G"][i, 0]) for i, s in enumerate(S)}, "gamma_target_true": float(outer["rank1"]["v_true_mean"][0]),
                       "gamma_target_hat_mean": float(outer["rank1"]["ghat"].mean(0)[0]), "sigma_inner": sig2.tolist(),
                       "slopes_rank1": {names[j]: float(outer["rank1"]["B"][0, j]) for j in range(M)}}
        selected[fn[g]] = {}
        # batch-mean transductive variant of verb1
        bm = dict(outer); bm["rank1"] = dict(outer["rank1"]); bm["rank1"]["ghat"] = np.repeat(outer["rank1"]["ghat"].mean(0, keepdims=True), len(te), 0)
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te)); score = {}
            for cfg in GRID:
                vals = []
                for s in S:
                    rows = np.flatnonzero(fam == s); Us = r[rows] - lam * c[rows]; vals.append(Us[np.arange(len(rows)), decide(cfg, lam, inner[s])].mean())
                v = np.sort(vals); score[cfg] = (float(v[0]), float(v.mean()))
            best = max(GRID, key=lambda k: score[k]); selected[fn[g]][L] = {"config": list(best), "inner_worst_mean": score[best]}
            ch = {"safe": np.full(len(te), int((rt - lam * ct).mean(0).argmax())), "soft_prior": decide(("transferred", 0.0, "soft"), lam, outer),
                  "verbosity_selected": decide(best, lam, outer), "verb1_soft_z0": decide(("verb1", 0.0, "soft"), lam, outer), "verb_nn_soft_z0": decide(("verb_nn", 0.0, "soft"), lam, outer),
                  "verb1_soft_batch": decide(("verb1", 0.0, "soft"), lam, bm),
                  "cost_oracle": (outer["Pr"] - lam * cbar[None, :]).argmax(1), "family_ceiling": np.full(len(te), int((rbar - lam * cbar).argmax())), "oracle": U.argmax(1)}
            for m in methods: per[m][L][fn[g]] = float(U[ii, ch[m]].mean())
        d = diag[fn[g]]
        print(f"{fn[g]:18s} gamma true {d['gamma_target_true']:+.2f} hat {d['gamma_target_hat_mean']:+.2f} sigma {sig2[0]:.2f} | lam100: safe {per['safe']['100.0'][fn[g]]:.3f} soft {per['soft_prior']['100.0'][fn[g]]:.3f} sel {per['verbosity_selected']['100.0'][fn[g]]:.3f} {selected[fn[g]]['100.0']['config']} verb1 {per['verb1_soft_z0']['100.0'][fn[g]]:.3f} nn {per['verb_nn_soft_z0']['100.0'][fn[g]]:.3f} batch {per['verb1_soft_batch']['100.0'][fn[g]]:.3f} c-oracle {per['cost_oracle']['100.0'][fn[g]]:.3f} ceil {per['family_ceiling']['100.0'][fn[g]]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in methods}
    def wl(a, b, L): return {"wins": int(sum(per[a][L][f] > per[b][L][f] + 1e-9 for f in fn)), "losses": int(sum(per[a][L][f] < per[b][L][f] - 1e-9 for f in fn)), "worst_delta": float(min(per[a][L][f] - per[b][L][f] for f in fn))}
    wtl = {L: {m: wl(m, "soft_prior", L) for m in ("verbosity_selected", "verb1_soft_z0", "verb_nn_soft_z0")} for L in agg["soft_prior"]}
    res = C.provenance(D); res.update({"experiment": EXP, "grid": [list(g_) for g_ in GRID], "per_family": per, "aggregate_qw_macro": agg, "selected": selected, "diagnostics": diag, "vs_soft_prior": wtl})
    md = [f"# {EXP} ({panel_name})", "", "Strict held-out family; nested inner-LOFO selection over cost estimator {transferred, verb1, verb2, verb_nn} x pessimism z {0,1,2} x reward side {soft, balanced}. Query-weighted [macro]. '_batch' is transductive (separate regime).", "",
          "lambda | " + " | ".join(methods) + " | selected vs soft W/L (worst)", "---|" + "---|" * (len(methods) + 1)]
    for lam in C.LAMS:
        L = str(lam); w = wtl[L]["verbosity_selected"]; md.append(f"{lam:g} | " + " | ".join(f"{agg[m][L][0]:.3f} [{agg[m][L][1]:.3f}]" for m in methods) + f" | {w['wins']}/{w['losses']} ({w['worst_delta']:+.3f})")
    for lam in (10.0, 50.0, 100.0, 1000.0):
        L = str(lam); md += ["", f"## per family, lambda = {lam:g} (selected config)", "", "family | " + " | ".join(methods) + " | config", "---|" + "---|" * (len(methods) + 1)]
        for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{per[m][L][fn[g]]:.3f}" for m in methods) + f" | {selected[fn[g]][L]['config']}")
    md += ["", "## verbosity axis: true vs predicted family coordinate (rank 1), inner sigma", "", "family | gamma true | gamma hat (mean over queries) | sigma", "---|---|---|---"]
    for g in range(F): d = diag[fn[g]]; md.append(f"{fn[g]} | {d['gamma_target_true']:+.2f} | {d['gamma_target_hat_mean']:+.2f} | {d['sigma_inner'][0]:.2f}")
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
