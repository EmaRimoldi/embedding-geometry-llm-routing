"""exp15 - paired significance for the relabelled cell lookup on LLMRouterBench (in-domain, 5 splits).

Comparisons at lambda in {0.5, 2, 5, 10} (rescaled LRB grid): cell lookup with LLM labels vs kNN-100; audited
labels vs kNN-100; LLM labels vs released labels; family lookup vs kNN-100. Same splits, lookups and kNN as exp14.
Significance as in the paper: (a) per query - paired difference of per-query utilities, 1,000-resample bootstrap
over test queries, 95% CI, per split; (b) equal weight over families - per-family mean differences, 1,000-resample
bootstrap over the 10 families, 95% CI, per split. Reported: mean point estimate over splits, mean CI bounds, and
the number of splits whose CI excludes 0 (with sign). Writes results/exp15_lrb_lookup_significance/.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from rb.probes.sweep import split
from exp14_lrb_topic_labels import labels, lookup, PANEL, MIN_CELL, SEEDS

EXP = "exp15_lrb_lookup_significance"; LAMS = (0.5, 2.0, 5.0, 10.0); B = 1000
PAIRS = [("lookup:llm", "knn100"), ("lookup:audited", "knn100"), ("lookup:llm", "lookup:released"), ("lookup:family", "knn100"), ("lookup:released", "knn100")]


def main():
    D = C.load(PANEL); r, c, fam, fn = D["r"], D["c"], D["fam"], D["fnames"]; F, n = len(fn), len(r); xu = C.unit(D["x"])
    L = labels(D); cells = {k: np.unique(v, return_inverse=True)[1] for k, v in L.items()}; rng = np.random.default_rng(0)
    out = {f"{a} - {b}": {f"{lam:g}": {"pq_point": [], "pq_ci": [], "eq_point": [], "eq_ci": []} for lam in LAMS} for a, b in PAIRS}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); nb = tr[C.knn_mean(xu[te], xu[tr], 100)]; ii = np.arange(len(te)); fte = fam[te]
        for lam in LAMS:
            U = r - lam * c; Ute = U[te]; u = {"knn100": Ute[ii, U[nb].mean(1).argmax(1)]}
            for k, cl in cells.items(): u[f"lookup:{k}"] = Ute[ii, lookup(U, tr, te, fam, cl, F)[0]]
            for a, b in PAIRS:
                d = u[a] - u[b]; key = f"{a} - {b}"; rec = out[key][f"{lam:g}"]
                idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1); rec["pq_point"].append(float(d.mean())); rec["pq_ci"].append([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
                pf = np.array([d[fte == g].mean() for g in range(F)]); fidx = rng.integers(0, F, (B, F)); fbs = pf[fidx].mean(1)
                rec["eq_point"].append(float(pf.mean())); rec["eq_ci"].append([float(np.percentile(fbs, 2.5)), float(np.percentile(fbs, 97.5))])
        print("seed", seed, flush=True)
    res = C.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMS), "bootstrap": B, "results": out})
    md = [f"# {EXP} ({PANEL})", "", "Paired differences (first minus second), in-domain 5 splits. Per query: bootstrap over test queries; equal weight: bootstrap over families. Entries: mean point estimate over splits [mean 95% CI], and n of 5 splits whose CI excludes 0 (+ or −).", ""]
    for key in out:
        md += [f"## {key}", "", "λ | per query | splits sig. | equal weight | splits sig.", "---|---|---|---|---"]
        for lam in LAMS:
            rec = out[key][f"{lam:g}"]; ci = np.mean(rec["pq_ci"], 0); eci = np.mean(rec["eq_ci"], 0)
            sp = sum(1 for lo, hi in rec["pq_ci"] if lo > 0) ; sn = sum(1 for lo, hi in rec["pq_ci"] if hi < 0); ep = sum(1 for lo, hi in rec["eq_ci"] if lo > 0); en = sum(1 for lo, hi in rec["eq_ci"] if hi < 0)
            md.append(f"{lam:g} | {np.mean(rec['pq_point']):+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {sp}+ / {sn}− | {np.mean(rec['eq_point']):+.4f} [{eci[0]:+.4f}, {eci[1]:+.4f}] | {ep}+ / {en}−")
        md.append("")
    C.save(C.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
