"""exp01 - which side of the held-out-family loss is cost, which is reward? (no learning beyond the posterior)

For every held-out family g (strict outer holdout), with source families S = all others:
  tables      rt[s,m], ct[s,m] = mean reward / cost of model m on source family s
  posterior   P[x,s] class-balanced GTE logistic posterior over S (same as the held-out study)
  soft        argmax_m P(x) @ (rt - lam*ct)                     -> the study's soft family prior
  soft+cost oracle    argmax_m P(x) @ rt - lam * cbar_g(m)       -> transferred reward, TRUE target mean cost
  soft+reward oracle  argmax_m rbar_g(m) - lam * P(x) @ ct       -> TRUE target mean reward, transferred cost
  family ceiling      argmax_m rbar_g - lam*cbar_g               -> in-domain family lookup (non deployable)
  safe        family-balanced best single on S
  knn100 / knn100+cost oracle: neighbours' mean reward - lam * (neighbours' mean cost | target mean cost)
  oracle
The two oracle rows use target outcomes only through a family MEAN vector and are diagnostics, not routers.
PROVES which side carries the OOD loss, per family and per lambda, both panels.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

EXP = "exp01_ood_decomposition" + ("_lam" + os.environ["E4_OCG_LAMS"].replace(";", "-").replace(",", "-").replace(" ", "-") if os.environ.get("E4_OCG_LAMS") else "")


def main():
    panel_name = sys.argv[1]
    D = C.load(panel_name); x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]
    F = len(fn); xu = C.unit(x)
    methods = ["safe", "soft", "soft_cost_oracle", "soft_reward_oracle", "family_ceiling", "knn100", "knn100_cost_oracle", "oracle"]
    per = {m: {str(l): {} for l in C.LAMS} for m in methods}; choice_info = {}; n_by = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); sources = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        rt, ct = C.family_tables(r, c, fam, sources)
        P = C.fit_posterior(x, fam, sources, te, balanced=True, seed=42)
        rbar, cbar = r[te].mean(0), c[te].mean(0)
        src_rows = np.flatnonzero(fam != g); nb = src_rows[C.knn_mean(xu[te], xu[src_rows], 100)]
        nr, nc = r[nb].mean(1), c[nb].mean(1)
        Pr, Pc = P @ rt, P @ ct
        choice_info[fn[g]] = {"posterior_mean": {fn[s]: float(P[:, i].mean()) for i, s in enumerate(sources)}}
        for lam in C.LAMS:
            U = r[te] - lam * c[te]; L = str(lam)
            ch = {"safe": np.full(len(te), int((rt - lam * ct).mean(0).argmax())),
                  "soft": (Pr - lam * Pc).argmax(1),
                  "soft_cost_oracle": (Pr - lam * cbar[None, :]).argmax(1),
                  "soft_reward_oracle": (rbar[None, :] - lam * Pc).argmax(1),
                  "family_ceiling": np.full(len(te), int((rbar - lam * cbar).argmax())),
                  "knn100": (nr - lam * nc).argmax(1),
                  "knn100_cost_oracle": (nr - lam * cbar[None, :]).argmax(1),
                  "oracle": U.argmax(1)}
            for m in methods:
                per[m][L][fn[g]] = float(U[np.arange(len(te)), ch[m]].mean())
            if lam == (100.0 if 100.0 in C.LAMS else C.LAMS[-1]):
                choice_info[fn[g]]["lam100_choice"] = {m: names[int(np.bincount(ch[m], minlength=len(names)).argmax())] for m in methods if m != "oracle"}
        print(fn[g], "done", flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in methods}
    res = C.provenance(D); res.update({"experiment": EXP, "per_family": per, "aggregate_qw_macro": agg, "choices": choice_info})
    md = [f"# {EXP} ({panel_name})", "", "Utility on the held-out family; query-weighted [macro over families]. 'soft' = class-balanced GTE posterior over source families x their (reward, cost) tables. Oracle rows substitute the target family's TRUE mean cost / reward vector (diagnostic only).", "",
          "lambda | " + " | ".join(methods), "---|" + "---|" * len(methods)]
    for lam in C.LAMS:
        L = str(lam); md.append(f"{lam:g} | " + " | ".join(f"{agg[m][L][0]:.3f} [{agg[m][L][1]:.3f}]" for m in methods))
    for lam in [l for l in (2.0, 5.0, 10.0, 50.0, 100.0, 1000.0) if l in C.LAMS]:
        L = str(lam); md += ["", f"## per family, lambda = {lam:g}", "", "family | " + " | ".join(methods), "---|" + "---|" * len(methods)]
        for g in range(F):
            md.append(f"{fn[g]} | " + " | ".join(f"{per[m][L][fn[g]]:.3f}" for m in methods))
        md += ["", "share of (ceiling - soft) recovered by the cost oracle / by the reward oracle:"]
        for g in range(F):
            gap = per["family_ceiling"][L][fn[g]] - per["soft"][L][fn[g]]
            if abs(gap) > 1e-6:
                md.append(f"- {fn[g]}: gap {gap:+.3f}; cost oracle {(per['soft_cost_oracle'][L][fn[g]] - per['soft'][L][fn[g]]) / gap:.2f}; reward oracle {(per['soft_reward_oracle'][L][fn[g]] - per['soft'][L][fn[g]]) / gap:.2f}")
    md += ["", f"## lambda = {(100.0 if 100.0 in C.LAMS else C.LAMS[-1]):g} majority choice per family", ""]
    for g in range(F):
        ci = choice_info[fn[g]]; md.append(f"- {fn[g]}: posterior mass " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(ci["posterior_mean"].items(), key=lambda t: -t[1])[:3]) + "; choices " + ", ".join(f"{k}={v}" for k, v in ci["lam100_choice"].items()))
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
