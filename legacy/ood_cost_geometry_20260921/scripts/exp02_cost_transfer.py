"""exp02 - does the cost of each model on an UNSEEN family transfer from the query text?

Strict outer family holdout. Target y[x, m] = log(cost_m(x) + 1e-7) for every model m (multi-output).
Feature sets: len  = [log(1 + n_in)]            (observed prompt tokens only)
              bow  = hashed bag-of-words 4096-d + len
              gte  = GTE-Qwen2-7B 3584-d
              gtel = GTE + len
Ridge with per-target alpha chosen by efficient leave-one-out on the SOURCE rows (RidgeCV,
alpha_per_target), features standardised on source rows, source rows weighted so that every
source family has equal total weight. No target information enters the fit.
Reported on the held-out family, per model: R^2 of log cost per query; the predicted family MEAN
cost (mean of exp(pred)) against the true mean as a log10 ratio; Spearman over models of predicted
vs true family-mean cost; plus, at every lambda, the utility of the router
     argmax_m  P(x) @ rt[:, m]  -  lam * chat_m(x)                    (soft reward, predicted per-query cost)
     argmax_m  P(x) @ rt[:, m]  -  lam * mean_target(chat_m)          (transductive: target-batch mean of predictions)
against the study's soft prior (transferred cost) and the family ceiling.
PROVES whether the representation reads the cost regime of a family it has never seen, and what a
router gains from that alone.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import common as C

EXP = "exp02_cost_transfer"; ALPHAS = np.logspace(-1, 5, 13)


def feats(D, which):
    ln = np.log1p(D["n_in"])[:, None].astype(np.float32)
    if which == "len": return ln
    if which == "bow": return np.hstack([D["bow"], ln])
    if which == "gte": return D["x"]
    if which == "gtel": return np.hstack([D["x"], ln])
    raise ValueError(which)


def main():
    panel_name = sys.argv[1]; sets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["len", "bow", "gte", "gtel"]
    t0 = time.time(); D = C.load(panel_name, with_bow="bow" in sets)
    x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M = len(fn), len(names)
    Y = np.log(c + C.EPS)
    out = {s: {} for s in sets}; n_by = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); tr = np.flatnonzero(fam != g); sources = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
        rt, ct = C.family_tables(r, c, fam, sources); P = C.fit_posterior(x, fam, sources, te, True, 42); Pr, Pc = P @ rt, P @ ct
        rbar, cbar = r[te].mean(0), c[te].mean(0); true_mean = cbar
        for s in sets:
            X = feats(D, s); sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr, Y[tr], sample_weight=w)
            pred = m.predict(Xte); chat = np.exp(pred) - C.EPS; chat = np.maximum(chat, 0.0)
            ss_res = ((Y[te] - pred) ** 2).sum(0); ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
            r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
            pred_mean = chat.mean(0)
            rec = {"r2_logcost_per_model": {names[j]: float(r2[j]) for j in range(M)}, "r2_mean": float(r2.mean()),
                   "log10_ratio_pred_over_true_family_mean_cost": {names[j]: float(np.log10((pred_mean[j] + 1e-9) / (true_mean[j] + 1e-9))) for j in range(M)},
                   "spearman_models_pred_vs_true_mean_cost": float(spearmanr(pred_mean, true_mean).correlation),
                   "alphas": {names[j]: float(m.alpha_[j]) for j in range(M)}, "utility": {}}
            for lam in C.LAMS:
                U = r[te] - lam * c[te]; ii = np.arange(len(te))
                ch_soft = (Pr - lam * Pc).argmax(1); ch_q = (Pr - lam * chat).argmax(1); ch_b = (Pr - lam * pred_mean[None, :]).argmax(1)
                ch_ceil = int((rbar - lam * cbar).argmax()); ch_co = (Pr - lam * cbar[None, :]).argmax(1)
                rec["utility"][str(lam)] = {"soft_transferred_cost": float(U[ii, ch_soft].mean()), "soft_reward_pred_cost_query": float(U[ii, ch_q].mean()),
                                            "soft_reward_pred_cost_batchmean": float(U[ii, ch_b].mean()), "soft_reward_true_mean_cost": float(U[ii, ch_co].mean()),
                                            "family_ceiling": float(U[:, ch_ceil].mean()), "oracle": float(U.max(1).mean())}
            out[s][fn[g]] = rec
            print(f"{fn[g]:20s} {s:5s} r2 mean {r2.mean():.3f}  spearman {rec['spearman_models_pred_vs_true_mean_cost']:.2f}  u100 soft {rec['utility']['100.0']['soft_transferred_cost']:.3f} pred-q {rec['utility']['100.0']['soft_reward_pred_cost_query']:.3f} pred-b {rec['utility']['100.0']['soft_reward_pred_cost_batchmean']:.3f} true-mean {rec['utility']['100.0']['soft_reward_true_mean_cost']:.3f} ceiling {rec['utility']['100.0']['family_ceiling']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    res = C.provenance(D); res.update({"experiment": EXP, "feature_sets": sets, "per_family": out})
    md = [f"# {EXP} ({panel_name})", "", "Strict held-out family. Ridge (per-target LOO alpha) from text features to log cost of every model, fitted on the other families with equal family weight.", ""]
    for s in sets:
        md += [f"## features: {s}", "", "family | mean R2 log-cost | Spearman(models) pred vs true mean cost | max |log10 ratio| of predicted/true family-mean cost", "---|---|---|---"]
        for g in range(F):
            rec = out[s][fn[g]]; md.append(f"{fn[g]} | {rec['r2_mean']:.3f} | {rec['spearman_models_pred_vs_true_mean_cost']:.2f} | {max(abs(v) for v in rec['log10_ratio_pred_over_true_family_mean_cost'].values()):.2f}")
        md += ["", "lambda | soft (transferred cost) | soft reward + predicted per-query cost | + target-batch mean | soft reward + TRUE mean cost | family ceiling | oracle  (query-weighted [macro])", "---|---|---|---|---|---|---"]
        for lam in C.LAMS:
            L = str(lam); cols = ["soft_transferred_cost", "soft_reward_pred_cost_query", "soft_reward_pred_cost_batchmean", "soft_reward_true_mean_cost", "family_ceiling", "oracle"]
            vals = [C.aggregate({fn[g]: out[s][fn[g]]["utility"][L][k] for g in range(F)}, n_by) for k in cols]
            md.append(f"{lam:g} | " + " | ".join(f"{a:.3f} [{b:.3f}]" for a, b in vals))
        md += ["", "per family at lambda = 100: soft / pred-q / pred-batch / true-mean / ceiling", ""]
        for g in range(F):
            u = out[s][fn[g]]["utility"]["100.0"]; md.append(f"- {fn[g]}: {u['soft_transferred_cost']:.3f} / {u['soft_reward_pred_cost_query']:.3f} / {u['soft_reward_pred_cost_batchmean']:.3f} / {u['soft_reward_true_mean_cost']:.3f} / {u['family_ceiling']:.3f}")
        md.append("")
    C.save(C.RESULTS / EXP / panel_name / "+".join(sets), res, md)


if __name__ == "__main__":
    main()
