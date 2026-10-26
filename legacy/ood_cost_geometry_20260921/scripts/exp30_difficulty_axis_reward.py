"""exp30 - the reward analogue of the cost router: keep the content-borrowed reward row (soft prior, p.R) but fix its
DIFFICULTY LEVEL. Training families' logit-reward table -> rank-1 axis: logit R[f, m] = mu[m] + a[m] * delta[f]
(delta = family difficulty coordinate, a[m] = model capability slope). A ridge from features (+ log n_in) to the training
rows' family coordinate gives delta_hat(q) for a held-out query; delta_prior(q) = posterior-weighted training coordinates.
Corrected reward: expit( logit(p.R) + kappa * a * (delta_hat - delta_prior) ), kappa in {0.5, 1}. Cost side: transferred
(soft) or reader cost (beta = 1). Oracles: delta_true (held-out family's coordinate on the training axis), true reward
row; references soft prior, family ceiling, oracle. Features: gtel and reader. Usage: exp30 <panel>."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import expit, logit
import common as C
from exp05_verbosity_axis_router import axes, coords, fit_scalar
from exp07_calibrated_verbosity_router import build, READER
from exp25_gated_cost_router import reader_logcost

EXP = "exp30_difficulty_axis_reward"; KAPPAS = (0.5, 1.0)


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name); x, r, c, fam, fn = D["x"], D["r"], D["c"], D["fam"], D["fnames"]; F = len(fn)
    ln = np.log1p(D["n_in"])[:, None].astype(np.float32); h = np.load(C.E4 / READER[panel_name], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(r) else h, dtype=np.float32)
    FE = {"gtel": np.hstack([x, ln]), "reader": np.hstack([h, ln])}; Y = np.log(c + C.EPS); LR = logit(np.clip(r, 1e-3, 1 - 1e-3))
    per = {}; n_by = {}; dg = {}
    def rec(m, L, g, v): per.setdefault(m, {}).setdefault(L, {})[fn[g]] = float(v)
    for g in range(F):
        te = np.flatnonzero(fam == g); tr = np.flatnonzero(fam != g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te); ii = np.arange(len(te))
        rt, ct = C.family_tables(r, c, fam, S); Lt = logit(np.clip(rt, 1e-3, 1 - 1e-3)); alpha_r, B_r, G_r = axes(Lt, 1); a_r = B_r[0]
        d_true = float(coords(logit(np.clip(r[te].mean(0), 1e-3, 1 - 1e-3))[None, :], alpha_r, B_r)[0, 0])
        blk = {f: build(D, X, Y, S, te) for f, X in FE.items()}; P = C.fit_posterior(x, fam, S, te, True, 42); Pr = P @ rt; d_prior = P @ G_r[:, 0]
        v = np.array([G_r[S.index(int(s)), 0] for s in fam[tr]]); w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
        d_hat = {}
        for f, X in FE.items():
            dh = fit_scalar(X[tr], v, w)(X[te])[:, 0]; lo, hi = G_r[:, 0].min(), G_r[:, 0].max(); d_hat[f] = np.clip(dh, lo - .25 * (hi - lo), hi + .25 * (hi - lo))
        dg[fn[g]] = {"delta_true": d_true, "delta_prior_mean": float(d_prior.mean()), **{f"delta_hat_{f}": float(d_hat[f].mean()) for f in FE}}
        logPr = logit(np.clip(Pr, 1e-3, 1 - 1e-3)); rbar, cbar = r[te].mean(0), c[te].mean(0)
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]
            costs = {"soft": np.exp(blk["gtel"]["log_transferred"]), "reader": np.exp(reader_logcost(blk["reader"]))}
            ch = {"soft_prior": (Pr - lam * costs["soft"]).argmax(1), "reader_cost": (Pr - lam * costs["reader"]).argmax(1), "reward_delta_true_oracle": (expit(logPr + a_r[None, :] * (d_true - d_prior)[:, None]) - lam * costs["soft"]).argmax(1), "true_reward_row": (rbar[None, :] - lam * costs["soft"]).argmax(1), "family_ceiling": np.full(len(te), int((rbar - lam * cbar).argmax())), "oracle": U.argmax(1)}
            for f in FE:
                for k in KAPPAS:
                    Rc = expit(logPr + k * a_r[None, :] * (d_hat[f] - d_prior)[:, None]); ch[f"reward_{f}_k{k:g}"] = (Rc - lam * costs["soft"]).argmax(1); ch[f"reward_{f}_k{k:g}+reader_cost"] = (Rc - lam * costs["reader"]).argmax(1)
            for m, sel in ch.items(): rec(m, L, g, U[ii, sel].mean())
        print(fn[g], dg[fn[g]], round(time.time() - t0), flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in per}
    res = C.provenance(D); res.update({"experiment": EXP, "per_family": per, "aggregate_qw_macro": agg, "delta_diagnostics": dg})
    md = [f"# {EXP} ({panel_name})", "", "Held-out family; reward = soft prior with its difficulty level corrected along the training rank-1 logit-reward axis; qw [macro].", "", "method | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)]
    for m in per: md.append(f"{m} | " + " | ".join(f"{agg[m][str(l)][0]:.3f} [{agg[m][str(l)][1]:.3f}]" for l in C.LAMS))
    md += ["", "## difficulty coordinate per held-out family: true / prior / read from gtel / read from reader", ""] + [f"- {f}: {v['delta_true']:+.2f} / {v['delta_prior_mean']:+.2f} / {v['delta_hat_gtel']:+.2f} / {v['delta_hat_reader']:+.2f}" for f, v in dg.items()]
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
