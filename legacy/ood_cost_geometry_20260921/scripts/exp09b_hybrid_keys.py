"""exp09b - hybrid retrieval targets: reward from the family posterior, cost from the query's regime.

exp09 showed that no single family-agnostic key (regime, difficulty, topic, products) with TRUE values
matches the family ceiling: pooling REWARD across families by regime loses family-specific strengths.
exp01 showed soft-posterior reward + the TRUE family-mean cost is at the ceiling. Here the cost side is
estimated from the query's TRUE regime coordinate but WITHOUT its family:
  reward side R(x):   soft = P(x) @ rt ;  fam = the held-out family's own mean reward (ceiling side)
  cost side C(x):     regime_bin  = family-balanced mean cost of source queries in the same regime bin (NB bins)
                      regime_knn  = mean cost of the K source queries nearest in the 1-D regime coordinate,
                                    family-balanced by weighting each source family equally among the K
                      axis        = exp(alpha_m + b_m * v(x)) from the rank-1 axis at the query's own coordinate
                      family_true = the held-out family's true mean cost (exp01 cost oracle)
                      transferred = P(x) @ ct (soft prior)
Utility on the held-out family, lambda grid, query-weighted [macro]. All regime coordinates use the
test query's own cost vector (oracle); this bounds what a perfect regime estimator could achieve with
each cost-side construction. PROVES whether "cost from regime, reward from resembling families" is the
right retrieval target, and which cost-side construction is adequate.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp05_verbosity_axis_router import axes, coords

EXP = "exp09b_hybrid_keys" + ("_lam" + os.environ["E4_OCG_LAMS"].replace(";", "-").replace(",", "-").replace(" ", "-") if os.environ.get("E4_OCG_LAMS") else ""); NB = 8; K = 300


def main():
    panel_name = sys.argv[1]; D = C.load(panel_name); r, c, fam, fn, names = D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M, n = len(fn), len(names), len(r)
    Y = np.log(c + C.EPS)
    costs = ["transferred", "regime_bin", "regime_knn", "axis", "family_true"]; rewards = ["soft", "fam"]
    methods = [f"{rw}+{cs}" for rw in rewards for cs in costs] + ["oracle"]
    per = {m: {str(l): {} for l in C.LAMS} for m in methods}; n_by = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); tr = np.flatnonzero(fam != g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        rt_f, ct_f = C.family_tables(r, c, fam, S); alpha, B, G = axes(np.log(ct_f + C.EPS), 1); v = coords(Y, alpha, B)[:, 0]
        P = C.fit_posterior(D["x"], fam, S, te, True, 42)
        R = {"soft": P @ rt_f, "fam": np.repeat(r[te].mean(0, keepdims=True), len(te), 0)}
        # cost constructions (n_te x M)
        Cc = {"transferred": P @ ct_f, "family_true": np.repeat(c[te].mean(0, keepdims=True), len(te), 0), "axis": np.exp(alpha[None, :] + v[te][:, None] * B[0][None, :])}
        qb = np.quantile(v[tr], np.linspace(0, 1, NB + 1)[1:-1]); bt, bq = np.digitize(v[tr], qb), np.digitize(v[te], qb)
        tab = {}
        for b in range(NB):
            m = tr[bt == b]
            if len(m) >= 50: fams = np.unique(fam[m]); tab[b] = np.mean([c[m][fam[m] == f].mean(0) for f in fams], 0)
        glob = np.mean([c[fam == s].mean(0) for s in S], 0); Cc["regime_bin"] = np.stack([tab.get(b, glob) for b in bq])
        order = np.argsort(v[tr]); vs = v[tr][order]; knn = np.zeros((len(te), M))
        for i, vi in enumerate(v[te]):
            j = np.searchsorted(vs, vi); lo, hi = max(0, j - K // 2), min(len(vs), j + K // 2); idx = tr[order[lo:hi]]
            w = np.array([1.0 / (fam[idx] == fam[q]).sum() for q in idx]); knn[i] = (w[:, None] * c[idx]).sum(0) / w.sum()
        Cc["regime_knn"] = knn
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te))
            for rw in rewards:
                for cs in costs: per[f"{rw}+{cs}"][L][fn[g]] = float(U[ii, (R[rw] - lam * Cc[cs]).argmax(1)].mean())
            per["oracle"][L][fn[g]] = float(U.max(1).mean())
        LK = str(100.0 if 100.0 in C.LAMS else C.LAMS[-1]); print(fn[g], f"lam{float(LK):g}:", {m: round(per[m][LK][fn[g]], 3) for m in methods}, flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in methods}
    res = C.provenance(D); res.update({"experiment": EXP, "bins": NB, "k": K, "per_family": per, "aggregate_qw_macro": agg})
    md = [f"# {EXP} ({panel_name})", "", f"Reward side x cost side; regime coordinate of the test query is its TRUE value (oracle). {NB} bins / K={K} neighbours in the 1-D coordinate, family-balanced. Query-weighted [macro].", "", "lambda | " + " | ".join(methods), "---|" + "---|" * len(methods)]
    for lam in C.LAMS:
        L = str(lam); md.append(f"{lam:g} | " + " | ".join(f"{agg[m][L][0]:.3f} [{agg[m][L][1]:.3f}]" for m in methods))
    for lam in [l for l in (2.0, 5.0, 50.0, 100.0) if l in C.LAMS]:
        L = str(lam); md += ["", f"## per family, lambda = {lam:g}", "", "family | " + " | ".join(methods), "---|" + "---|" * len(methods)]
        for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{per[m][L][fn[g]]:.3f}" for m in methods))
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
