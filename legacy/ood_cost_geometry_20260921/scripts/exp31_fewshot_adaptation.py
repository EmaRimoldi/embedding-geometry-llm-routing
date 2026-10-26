"""exp31 - how many labelled queries from a NEW family close the held-out gap? For each held-out family, K labelled rows
(K in 0, 5, 10, 20, 50, 100, 200; 20 random draws) are observed with all models' rewards and costs. Family row estimate =
shrinkage mix: (K * observed_mean + n0 * prior) / (K + n0), prior = soft prior's posterior-weighted training tables
(reward) and, for cost, the posterior-weighted training cost (variant 'soft') or the reader cost (variant 'reader', RB and
LRB only). The remaining rows of the family are routed with argmax of the estimated row (family-level decision, like the
in-domain family lookup). Reports pooled per query [equal weight] utility vs K against soft prior, family ceiling, oracle.
n0 in {10, 50}. Usage: exp31 <panel>  (panel = sprout uses sprout_common, cost variant soft only)."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
panel = sys.argv[1]; os.environ["E4_PANEL"] = panel if panel != "sprout" else "routerbench-0shot"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = f"exp31_fewshot_adaptation_{'sprout' if panel == 'sprout' else 'rb' if panel.startswith('router') else 'lrb'}"
KS = (0, 5, 10, 20, 50, 100, 200); DRAWS = 20; N0 = (10, 50)
LAMS = {"sprout": (0.0, 0.5, 5.0, 25.0, 50.0, 250.0, 500.0, 1000.0), "routerbench-0shot": (0.0, 1.0, 10.0, 50.0, 100.0, 500.0), "llmrouterbench-performance-cost": (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)}[panel]


def load():
    if panel == "sprout":
        D = S.load(); return D["r"], D["c"], D["fam"], D["fnames"], D["x"], None
    import common as Cm; from exp07_calibrated_verbosity_router import build, READER
    D = Cm.load(panel); ln = np.log1p(D["n_in"])[:, None].astype(np.float32); h = np.load(Cm.E4 / READER[panel], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(D["r"]) else h, dtype=np.float32)
    return D["r"], D["c"], D["fam"], D["fnames"], D["x"], (D, np.hstack([h, ln]), np.log(D["c"] + Cm.EPS), build)


def main():
    t0 = time.time(); r, c, fam, fn, x, reader = load(); F, n = len(fn), len(r); xu = S.unit(x); rng = np.random.default_rng(0)
    variants = ["soft"] + (["reader"] if reader else []); per = {}; nte = []
    def rec(m, L, v): per.setdefault(m, {}).setdefault(L, []).append(float(v))
    for g in range(F):
        te = np.where(fam == g)[0]; tr = np.where(fam != g)[0]; src = [s for s in range(F) if s != g]; nte.append(len(te))
        sc = StandardScaler().fit(xu[tr]); clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(sc.transform(xu[tr]), fam[tr]); post = clf.predict_proba(sc.transform(xu[te]))[:, [list(clf.classes_).index(s) for s in src]]
        R = np.stack([r[fam == s].mean(0) for s in src]); Cc = np.stack([c[fam == s].mean(0) for s in src]); Pr, Pc = post @ R, post @ Cc; rt, ct = r[te].mean(0), c[te].mean(0)
        cost_prior = {"soft": Pc}
        if reader:
            D, X, Y, build = reader; from exp25_gated_cost_router import reader_logcost; blk = build(D, X, Y, src, te); cost_prior["reader"] = np.exp(reader_logcost(blk))
        for lam in LAMS:
            L = f"{lam:g}"; U = r[te] - lam * c[te]; ii = np.arange(len(te))
            rec("soft_prior", L, U[ii, (Pr - lam * Pc).argmax(1)].mean()); rec("family_ceiling", L, U[ii, int((rt - lam * ct).argmax())].mean()); rec("oracle", L, U.max(1).mean())
            for v in variants:
                for n0 in N0:
                    for K in KS:
                        if K >= len(te) // 2: continue   # small families (MT-Bench 80 rows, AIME 60): K must leave a test half
                        vals = []
                        for d in range(DRAWS if K > 0 else 1):
                            obs = rng.choice(len(te), K, replace=False) if K > 0 else np.array([], dtype=int); rest = np.setdiff1d(np.arange(len(te)), obs)
                            ro = r[te][obs].mean(0) if K > 0 else 0; co = c[te][obs].mean(0) if K > 0 else 0
                            # family-level row estimate: shrink the observed means toward the query-level priors (averaged over the family's rows)
                            Rrow = (K * ro + n0 * Pr[rest].mean(0)) / (K + n0); Crow = (K * co + n0 * cost_prior[v][rest].mean(0)) / (K + n0)
                            m = int((Rrow - lam * Crow).argmax()); vals.append(float(U[rest, m].mean()))
                        rec(f"fewshot_{v}_n0{n0}_K{K}", L, np.mean(vals))
        print(fn[g], round(time.time() - t0), flush=True)
    # methods missing on small families: pool over the families where they were run (their own weights)
    def agg(m, L):
        v = per[m][L]; fams = [g for g in range(F) if len(per[m][L]) > g] if False else None
        return v
    res = {"experiment": EXP, "panel": panel, "lambdas": list(LAMS), "K": list(KS), "n0": list(N0), "draws": DRAWS, "families": fn, "results": {}}
    fam_of = {}  # method -> list of family indices in order of recording
    for m in per:
        for L in per[m]:
            v = per[m][L]; k = len(v); gs = [g for g in range(F)] if k == F else [g for g in range(F) if KS and any(f"_K{K}" in m and K < nte[g] // 2 for K in KS)] if "fewshot" in m else list(range(F))
            gs = gs[:k]; w = np.array([nte[g] for g in gs], dtype=float); w /= w.sum(); res["results"].setdefault(m, {})[L] = [float(np.dot(w, v)), float(np.mean(v)), dict(zip([fn[g] for g in gs], v)), len(gs)]
    md = [f"# {EXP} ({panel})", "", "Held-out family; K labelled queries of the new family; row estimate shrunk to the prior (n0 pseudo-counts); remaining rows routed at the family level. Pooled per query [equal weight].", ""]
    for v in variants:
        for n0 in N0:
            md += [f"## cost prior = {v}, n0 = {n0}", "", "K | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
            md.append("soft prior (query-level) | " + " | ".join(f"{res['results']['soft_prior'][f'{l:g}'][0]:.3f} [{res['results']['soft_prior'][f'{l:g}'][1]:.3f}]" for l in LAMS))
            for K in KS: md.append(f"{K} | " + " | ".join(f"{res['results'][f'fewshot_{v}_n0{n0}_K{K}'][f'{l:g}'][0]:.3f} [{res['results'][f'fewshot_{v}_n0{n0}_K{K}'][f'{l:g}'][1]:.3f}] ({res['results'][f'fewshot_{v}_n0{n0}_K{K}'][f'{l:g}'][3]} fam)" for l in LAMS))
            md.append("family ceiling | " + " | ".join(f"{res['results']['family_ceiling'][f'{l:g}'][0]:.3f} [{res['results']['family_ceiling'][f'{l:g}'][1]:.3f}]" for l in LAMS)); md.append("")
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
