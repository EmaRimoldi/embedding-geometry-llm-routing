"""exp06 - how much do the zero-cost failed calls contaminate LLMRouterBench results?

Artifact: 2,366 (model, query) cells with cost == 0 and reward == 0 (failed calls stored as zeros;
RouterBench has none). Two panels are compared:
  raw      the frozen panel as released
  imputed  artifact cells keep reward 0 but are charged the model's median cost over the
           non-artifact rows of the same family (panel median if the family has none):
           "a failed call is paid for and returns nothing"
For lambda in {1, 10, 50, 100}, both panels:
  in-domain (5 family-stratified 70/30 splits, seeds 0-4): best single (train), (family, topic) cell
  lookup (min 50 rows), oracle, gap and its reward / cost parts; share of queries where each
  picks an artifact cell (evaluated on the raw panel).
  held-out family: safe, soft family prior, cost oracle, family ceiling, oracle; same exploitation
  shares. Lookups/tables are fitted on the panel under test; a router fitted on raw is also
  evaluated on imputed ("raw-fit / imputed-eval") to show what the released tables cost a
  deployable router when failed calls are really paid for.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from rb.panelctx import topic_str
from rb.probes.sweep import split

EXP = "exp06_artifact_contamination"; LAMS = (1.0, 10.0, 50.0, 100.0); MIN_CELL = 50


def impute(r, c, fam):
    z = (c == 0) & (r == 0); ci = c.copy(); F = int(fam.max()) + 1
    for j in range(c.shape[1]):
        pm = np.median(c[~z[:, j], j]) if (~z[:, j]).any() else 0.0
        for g in range(F):
            m = (fam == g) & z[:, j]
            if m.any():
                ok = (fam == g) & ~z[:, j]; ci[m, j] = np.median(c[ok, j]) if ok.any() else pm
    return ci, z


def cell_lookup(U, tr, fam, cell, F, n_cells):
    fb = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)]); cb = np.full(n_cells, -1)
    for k in np.unique(cell[tr]):
        m = cell[tr] == k
        if m.sum() >= MIN_CELL: cb[k] = U[tr][m].mean(0).argmax()
    ch = cb[cell]; ch = np.where(ch < 0, fb[fam], ch); return ch


def main():
    panel_name = sys.argv[1]; D = C.load(panel_name); x, r, c_raw, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]
    F, M, n = len(fn), len(names), len(r); c_imp, Z = impute(r, c_raw, fam)
    _, cell = np.unique(np.array([str(t) for t in topic_str(D["panel"])], dtype=object), return_inverse=True); n_cells = int(cell.max()) + 1
    rows = np.arange(n); res = {"n_artifact_cells": int(Z.sum()), "in_domain": {}, "held_out": {}}
    md = [f"# {EXP} ({panel_name})", "", f"{int(Z.sum())} artifact cells (cost 0, reward 0). 'imputed' charges them the model's family-median cost with reward 0.", ""]
    # ---------------- in-domain
    md += ["## in-domain (5 splits): utility per query; 'exploit' = share of test queries routed to an artifact cell", "", "lambda | panel | best single | cell lookup | oracle | gap = reward + cost | exploit oracle / lookup", "---|---|---|---|---|---|---"]
    for lam in LAMS:
        for tag, c in (("raw", c_raw), ("imputed", c_imp)):
            U = r - lam * c; acc = []
            for seed in (0, 1, 2, 3, 4):
                tr, te = split(n, fam, seed); ml = cell_lookup(U, tr, fam, cell, F, n_cells); bs = int(U[tr].mean(0).argmax()); mo = U.argmax(1)
                g = U[te, mo[te]] - U[te, ml[te]]; rg = r[te, mo[te]] - r[te, ml[te]]; cg = -lam * (c[te, mo[te]] - c[te, ml[te]])
                acc.append([U[te, bs].mean(), U[te, ml[te]].mean(), U[te, mo[te]].mean(), g.mean(), rg.mean(), cg.mean(), Z[te, mo[te]].mean(), Z[te, ml[te]].mean()])
            a = np.mean(acc, 0); res["in_domain"][f"{lam:g}/{tag}"] = dict(zip(["best_single", "cell_lookup", "oracle", "gap", "gap_reward", "gap_cost", "exploit_oracle", "exploit_lookup"], map(float, a)))
            md.append(f"{lam:g} | {tag} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} | {a[3]:.3f} = {a[4]:.3f} + {a[5]:.3f} | {a[6]:.3f} / {a[7]:.3f}")
    # ---------------- held-out
    md += ["", "## held-out family: query-weighted [macro]; 'raw-fit/imputed-eval' = soft prior fitted on raw tables, charged imputed costs", "", "lambda | panel | safe | soft prior | cost oracle | family ceiling | oracle | exploit soft / oracle | soft raw-fit/imputed-eval", "---|---|---|---|---|---|---|---|---"]
    P_by = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; P_by[g] = (te, S, C.fit_posterior(x, fam, S, te, True, 42))
    for lam in LAMS:
        for tag, c in (("raw", c_raw), ("imputed", c_imp)):
            per = {k: {} for k in ("safe", "soft", "cost_oracle", "ceiling", "oracle", "ex_soft", "ex_oracle", "soft_rawfit_impeval")}; n_by = {}
            for g in range(F):
                te, S, P = P_by[g]; n_by[fn[g]] = len(te); U = r[te] - lam * c[te]; ii = np.arange(len(te))
                rt, ct = C.family_tables(r, c, fam, S); rt0, ct0 = C.family_tables(r, c_raw, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0)
                ch_soft = (P @ rt - lam * (P @ ct)).argmax(1); ch_soft_raw = (P @ rt0 - lam * (P @ ct0)).argmax(1); mo = U.argmax(1)
                per["safe"][fn[g]] = float(U[:, int((rt - lam * ct).mean(0).argmax())].mean()); per["soft"][fn[g]] = float(U[ii, ch_soft].mean())
                per["cost_oracle"][fn[g]] = float(U[ii, (P @ rt - lam * cbar[None, :]).argmax(1)].mean()); per["ceiling"][fn[g]] = float(U[:, int((rbar - lam * cbar).argmax())].mean())
                per["oracle"][fn[g]] = float(U.max(1).mean()); per["ex_soft"][fn[g]] = float(Z[te, ch_soft].mean()); per["ex_oracle"][fn[g]] = float(Z[te, mo].mean())
                per["soft_rawfit_impeval"][fn[g]] = float((r[te] - lam * c_imp[te])[ii, ch_soft_raw].mean())
            agg = {k: C.aggregate(v, n_by) for k, v in per.items()}; res["held_out"][f"{lam:g}/{tag}"] = {"aggregate": agg, "per_family": per}
            md.append(f"{lam:g} | {tag} | " + " | ".join(f"{agg[k][0]:.3f} [{agg[k][1]:.3f}]" for k in ("safe", "soft", "cost_oracle", "ceiling", "oracle")) + f" | {agg['ex_soft'][0]:.3f} / {agg['ex_oracle'][0]:.3f} | {agg['soft_rawfit_impeval'][0]:.3f} [{agg['soft_rawfit_impeval'][1]:.3f}]")
    md += ["", "## held-out per family at lambda = 50 and 100, raw vs imputed: soft prior / cost oracle / ceiling / oracle", ""]
    for lam in (50.0, 100.0):
        for g in range(F):
            a = res["held_out"][f"{lam:g}/raw"]["per_family"]; b = res["held_out"][f"{lam:g}/imputed"]["per_family"]
            md.append(f"- λ={lam:g} {fn[g]}: raw {a['soft'][fn[g]]:.3f} / {a['cost_oracle'][fn[g]]:.3f} / {a['ceiling'][fn[g]]:.3f} / {a['oracle'][fn[g]]:.3f} (oracle exploits {a['ex_oracle'][fn[g]]:.2f}) → imputed {b['soft'][fn[g]]:.3f} / {b['cost_oracle'][fn[g]]:.3f} / {b['ceiling'][fn[g]]:.3f} / {b['oracle'][fn[g]]:.3f}")
    out = C.provenance(D); out.update({"experiment": EXP, "results": res}); C.save(C.RESULTS / EXP / panel_name, out, md)


if __name__ == "__main__":
    main()
