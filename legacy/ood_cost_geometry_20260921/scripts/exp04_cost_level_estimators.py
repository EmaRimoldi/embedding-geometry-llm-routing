"""exp04 - estimating the cost LEVEL of an unseen family: price x verbosity, and competing estimators.

exp01 showed that the family MEAN cost vector cbar_g(m) is enough to close the held-out loss at
lambda >= 50 on both panels. This experiment asks how well that vector can be estimated without
target outcomes, and what each estimate buys once plugged into the same router
      argmax_m  P(x) @ rt[:, m]  -  lam * Chat_m(x)          (reward side = the soft family prior).

(0) Structure test: on the source families' log mean-cost table  log ct[s, m]  fit the rank-1
    additive model  log ct[s, m] ~ alpha_m + gamma_s  ("price x verbosity"); report the fraction of
    variance explained, and the same for the full per-query log-cost matrix.
Estimators of cbar_g(m) (all fitted on source rows only):
  transferred    P(x) @ ct                                          (= the held-out study)
  ridge_query    exp(ridge_m(x)) from GTE + log(1+n_in), per query           (13 targets)
  ridge_clip     ridge_query with log prediction clipped to [min_s, max_s] log ct[s, m]
  regime         K-means (K=2,3) of source families on their log-cost profile; a class-balanced
                 GTE logistic query->regime posterior; cost = posterior-weighted regime mean table
  verbosity      gamma(x) = ridge from GTE + log(1+n_in) to the per-query verbosity scalar
                 v(x) = mean_m [log c_m(x) - alpha_m]; cost_m(x) = exp(alpha_m + gamma(x)), clipped
                 to the source range of gamma_s                                      (1 target)
  verbosity_len  same with log(1+n_in) as the only feature
  *_batch        transductive variants: replace the per-query estimate by its mean over the
                 target batch (reported separately, as in the held-out study).
Metrics per held-out family: mean |log10 chat/cbar| over models; Spearman over models; router
utility at every lambda, query-weighted [macro], per family; the exp01 references (soft, cost oracle,
family ceiling, oracle). No target outcome is used anywhere except in the reference rows.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import common as C

EXP = "exp04_cost_level_estimators"; ALPHAS = np.logspace(-1, 5, 13)


def rank1_fit(T):
    """Additive fit of a log table T[s, m] = alpha_m + gamma_s; returns alpha, gamma, R2."""
    mu = T.mean(); a = T.mean(0) - mu; g = T.mean(1) - mu; fit = mu + a[None, :] + g[:, None]
    return a + mu, g, 1 - ((T - fit) ** 2).sum() / ((T - mu) ** 2).sum()


def ridge(Xtr, Ytr, Xte, w):
    sc = StandardScaler().fit(Xtr); m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Xtr), Ytr, sample_weight=w)
    return m.predict(sc.transform(Xte)).reshape(len(Xte), -1)


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name)
    x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M = len(fn), len(names)
    Xg = np.hstack([x, np.log1p(D["n_in"])[:, None].astype(np.float32)]); Xl = np.log1p(D["n_in"])[:, None]
    Y = np.log(c + C.EPS)
    ests = ["transferred", "ridge_query", "ridge_clip", "regime2", "regime3", "verbosity", "verbosity_len", "ridge_clip_batch", "verbosity_batch", "verbosity_len_batch"]
    refs = ["soft", "cost_oracle", "family_ceiling", "oracle"]
    per = {m: {str(l): {} for l in C.LAMS} for m in ests + refs}; err = {m: {} for m in ests}; n_by = {}; structure = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; tr = np.flatnonzero(fam != g); n_by[fn[g]] = len(te)
        w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
        rt, ct = C.family_tables(r, c, fam, S); logct = np.log(ct + C.EPS)
        P = C.fit_posterior(x, fam, S, te, True, 42); Pr = P @ rt
        rbar, cbar = r[te].mean(0), c[te].mean(0)
        # (0) structure
        alpha, gamma_s, r2_tab = rank1_fit(logct)
        v_all = (Y - alpha[None, :]).mean(1)                       # per-query verbosity scalar
        fit_q = alpha[None, :] + v_all[:, None]; r2_query = 1 - ((Y[tr] - fit_q[tr]) ** 2).sum() / ((Y[tr] - Y[tr].mean()) ** 2).sum()
        structure[fn[g]] = {"r2_rank1_source_table": float(r2_tab), "r2_rank1_source_queries": float(r2_query), "gamma_sources": {fn[s]: float(gamma_s[i]) for i, s in enumerate(S)}, "gamma_target_true": float(v_all[te].mean() - (Y[tr] - alpha[None, :]).mean())}
        lo, hi = logct.min(0), logct.max(0); glo, ghi = gamma_s.min(), gamma_s.max()
        # estimators (log cost per query per model)
        est = {"transferred": np.log(P @ ct + C.EPS)}
        rq = ridge(Xg[tr], Y[tr], Xg[te], w); est["ridge_query"] = rq; est["ridge_clip"] = np.clip(rq, lo[None, :], hi[None, :])
        for K in (2, 3):
            if len(S) <= K: est[f"regime{K}"] = est["transferred"]; continue
            km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(logct); lab = km.labels_
            reg_tab = np.stack([logct[lab == k].mean(0) for k in range(K)])
            qlab = lab[np.searchsorted(S, fam[tr])]
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, class_weight="balanced", random_state=42).fit(x[tr], qlab)
            Q = clf.predict_proba(x[te]); est[f"regime{K}"] = np.log(Q @ np.exp(reg_tab) + C.EPS)
            structure[fn[g]][f"regime{K}_clusters"] = {fn[s]: int(lab[i]) for i, s in enumerate(S)}
        mu_v = (Y[tr] - alpha[None, :]).mean()
        gq = ridge(Xg[tr], (v_all - mu_v)[tr, None], Xg[te], w)[:, 0]; est["verbosity"] = alpha[None, :] + mu_v + np.clip(gq, glo, ghi)[:, None]
        gl = ridge(Xl[tr], (v_all - mu_v)[tr, None], Xl[te], w)[:, 0]; est["verbosity_len"] = alpha[None, :] + mu_v + np.clip(gl, glo, ghi)[:, None]
        for k in ("ridge_clip", "verbosity", "verbosity_len"):
            est[k + "_batch"] = np.repeat(np.log(np.exp(est[k]).mean(0, keepdims=True) + C.EPS), len(te), axis=0)
        structure[fn[g]]["gamma_hat"] = {"verbosity": float(np.clip(gq, glo, ghi).mean()), "verbosity_len": float(np.clip(gl, glo, ghi).mean())}
        for k in ests:
            mean_hat = np.exp(est[k]).mean(0)
            err[k][fn[g]] = {"mean_abs_log10_ratio": float(np.mean(np.abs(np.log10((mean_hat + 1e-9) / (cbar + 1e-9))))), "spearman_models": float(spearmanr(mean_hat, cbar).correlation),
                             "log10_ratio_per_model": {names[j]: float(np.log10((mean_hat[j] + 1e-9) / (cbar[j] + 1e-9))) for j in range(M)}}
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te))
            for k in ests: per[k][L][fn[g]] = float(U[ii, (Pr - lam * np.exp(est[k])).argmax(1)].mean())
            per["soft"][L][fn[g]] = per["transferred"][L][fn[g]]
            per["cost_oracle"][L][fn[g]] = float(U[ii, (Pr - lam * cbar[None, :]).argmax(1)].mean())
            per["family_ceiling"][L][fn[g]] = float(U[:, int((rbar - lam * cbar).argmax())].mean()); per["oracle"][L][fn[g]] = float(U.max(1).mean())
        print(f"{fn[g]:18s} rank1 R2 table {r2_tab:.2f} queries {r2_query:.2f} | lam100: soft {per['soft']['100.0'][fn[g]]:.3f} ridge_clip {per['ridge_clip']['100.0'][fn[g]]:.3f} regime2 {per['regime2']['100.0'][fn[g]]:.3f} verbosity {per['verbosity']['100.0'][fn[g]]:.3f} verb_len {per['verbosity_len']['100.0'][fn[g]]:.3f} oracle-cost {per['cost_oracle']['100.0'][fn[g]]:.3f} | err log10 verb {err['verbosity'][fn[g]]['mean_abs_log10_ratio']:.2f} clip {err['ridge_clip'][fn[g]]['mean_abs_log10_ratio']:.2f} transf {err['transferred'][fn[g]]['mean_abs_log10_ratio']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in per}
    res = C.provenance(D); res.update({"experiment": EXP, "estimators": ests, "per_family": per, "aggregate_qw_macro": agg, "cost_level_error": err, "structure": structure})
    cols = ests + refs
    md = [f"# {EXP} ({panel_name})", "", "Router = soft family prior for reward + the named cost estimator; strict held-out family; query-weighted [macro]. '_batch' rows are transductive (target-batch mean of the per-query estimate) and belong to a separate regime.", "",
          "## price x verbosity structure of the source log mean-cost table (rank-1 additive fit), per outer fold", "", "held-out | R2 table | R2 all source queries | true target gamma | gamma_hat verbosity | gamma_hat len", "---|---|---|---|---|---"]
    for g in range(F): s_ = structure[fn[g]]; md.append(f"{fn[g]} | {s_['r2_rank1_source_table']:.3f} | {s_['r2_rank1_source_queries']:.3f} | {s_['gamma_target_true']:+.2f} | {s_['gamma_hat']['verbosity']:+.2f} | {s_['gamma_hat']['verbosity_len']:+.2f}")
    md += ["", "## cost-level error on the held-out family: mean over models of |log10(estimated/true mean cost)|  (Spearman over models)", "", "family | " + " | ".join(ests), "---|" + "---|" * len(ests)]
    for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{err[k][fn[g]]['mean_abs_log10_ratio']:.2f} ({err[k][fn[g]]['spearman_models']:.2f})" for k in ests))
    md += ["", "## router utility", "", "lambda | " + " | ".join(cols), "---|" + "---|" * len(cols)]
    for lam in C.LAMS:
        L = str(lam); md.append(f"{lam:g} | " + " | ".join(f"{agg[m][L][0]:.3f} [{agg[m][L][1]:.3f}]" for m in cols))
    for lam in (10.0, 50.0, 100.0, 1000.0):
        L = str(lam); md += ["", f"## per family, lambda = {lam:g}", "", "family | " + " | ".join(cols), "---|" + "---|" * len(cols)]
        for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{per[m][L][fn[g]]:.3f}" for m in cols))
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
