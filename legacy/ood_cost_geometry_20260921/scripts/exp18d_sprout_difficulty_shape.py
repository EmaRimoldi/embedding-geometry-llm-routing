"""exp18d - is SPROUT's strong difficulty locality a difficulty axis or near-duplicate questions?
(a) k=10 locality (fraction-of-models-solving proxy, inside the cell, split 0) with neighbours restricted to cosine
    <= 0.95 and <= 0.90 (near-duplicates excluded), pooled per query and per family;
(b) quartile test (paper's Sec. 3 figure): inside each cell, a query vs a random peer from its own solve-fraction
    quartile and one from the opposite quartile: share of comparisons where the same-quartile peer is closer (chance 0.5);
    per family and pooled; (c) held-out variance of the solve fraction explained beyond the cell mean by two K-means
    clusters fitted inside each cell (embedding only). Writes results/exp18d_sprout_difficulty_shape/."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import sprout_common as S

EXP = "exp18d_sprout_difficulty_shape"; import os; EXP = EXP + ("_v2" if os.environ.get("SPROUT_TOPICS") == "v2" else ""); MIN_CELL = 50; K = 10


def main():
    t0 = time.time(); D = S.load(); r, fam, fn = D["r"], D["fam"], D["fnames"]; F, n = len(fn), D["n"]; xu = S.unit(D["x"]); s = (r >= 0.5).mean(1)
    _, cell = np.unique(D["cell_str"], return_inverse=True); tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=fam)
    cnt = np.bincount(cell[tr], minlength=cell.max() + 1); key_te = np.where(cnt[cell[te]] >= MIN_CELL, cell[te], -1 - fam[te]); key_tr = np.where(cnt[cell[tr]] >= MIN_CELL, cell[tr], -1 - fam[tr])
    res = S.provenance(D); res["experiment"] = EXP; md = [f"# {EXP} (SPROUT, split 0)", ""]
    thr = {"all neighbours": 1.01, "cosine <= 0.95": 0.95, "cosine <= 0.90": 0.90}; acc = {t: {"nn": np.zeros(n), "all": np.zeros(n), "ok": np.zeros(n, bool)} for t in thr}
    quart = {"same_closer": np.zeros(F), "n": np.zeros(F)}; var = {"ss_cell": 0.0, "ss_2cl": 0.0, "ss_tot": 0.0}; rng = np.random.default_rng(0)
    for kk in np.unique(key_te):
        qi = te[key_te == kk]; ti = tr[key_tr == kk]
        if len(ti) < 2 * K + 2: continue
        Sm = xu[qi] @ xu[ti].T
        for t, th in thr.items():
            Sx = np.where(Sm <= th, Sm, -np.inf); enough = (Sx > -np.inf).sum(1) >= K
            p = np.argpartition(-Sx, K - 1, axis=1)[:, :K]; nnr = ti[p]
            d_nn = np.abs(s[qi][:, None] - s[nnr]).mean(1); d_all = np.abs(s[qi][:, None] - s[ti][None, :]).mean(1)
            acc[t]["nn"][qi[enough]] = d_nn[enough]; acc[t]["all"][qi[enough]] = d_all[enough]; acc[t]["ok"][qi[enough]] = True
        # quartile test among training rows of the cell
        st = s[ti]; q1, q3 = np.percentile(st, 25), np.percentile(st, 75); lo, hi = np.where(st <= q1)[0], np.where(st >= q3)[0]
        if len(lo) >= 5 and len(hi) >= 5 and q1 < q3:
            for grp, other in ((lo, hi), (hi, lo)):
                a = rng.choice(grp, min(len(grp), 400), replace=False); same = rng.choice(grp, len(a)); opp = rng.choice(other, len(a)); same = np.where(same == a, rng.choice(grp, len(a)), same)
                cs = np.einsum("ij,ij->i", xu[ti[a]], xu[ti[same]]); co = np.einsum("ij,ij->i", xu[ti[a]], xu[ti[opp]]); g = fam[ti[a][0]]
                quart["same_closer"][g] += (cs > co).sum(); quart["n"][g] += len(a)
        # two clusters inside the cell
        km = KMeans(2, n_init=3, random_state=0).fit(xu[ti]); ct = km.predict(xu[qi]); m_cell = s[ti].mean(); m_cl = np.array([s[ti][km.labels_ == j].mean() for j in range(2)])
        var["ss_tot"] += ((s[qi] - m_cell) ** 2).sum(); var["ss_2cl"] += ((s[qi] - m_cl[ct]) ** 2).sum()
    md += ["## (a) k=10 locality, fraction of models solving, with near-duplicate neighbours excluded (% reduction vs all rows of the cell)", "", "neighbour set | per query | equal weight | " + " | ".join(fn), "---|---|---|" + "---|" * F]; res["locality"] = {}
    for t, a in acc.items():
        ok = a["ok"]; red = 100 * (1 - a["nn"][ok].sum() / a["all"][ok].sum()); pf = {fn[g]: float(100 * (1 - a["nn"][ok & (fam == g)].sum() / a["all"][ok & (fam == g)].sum())) for g in range(F)}
        res["locality"][t] = {"per_query": float(red), "equal_weight": float(np.mean(list(pf.values()))), "per_family": pf, "queries": int(ok.sum())}; md.append(f"{t} | {red:.1f} | {np.mean(list(pf.values())):.1f} | " + " | ".join(f"{pf[f]:.1f}" for f in fn))
    pq = quart["same_closer"].sum() / quart["n"].sum(); pf = {fn[g]: float(quart["same_closer"][g] / quart["n"][g]) for g in range(F) if quart["n"][g] > 0}
    res["quartile_test"] = {"pooled": float(pq), "per_family": pf, "comparisons": int(quart["n"].sum())}; md += ["", f"## (b) quartile test: share of comparisons where the same-quartile peer is closer (chance 0.5): pooled {pq:.3f}; " + ", ".join(f"{k} {v:.3f}" for k, v in pf.items()) + f" (n={int(quart['n'].sum())}); RouterBench 0.511, LLMRouterBench 0.515"]
    ve = 1 - var["ss_2cl"] / var["ss_tot"]; res["two_cluster_extra_variance"] = float(ve); md += ["", f"## (c) held-out solve-fraction variance explained beyond the cell mean by two embedding clusters per cell: {100*ve:.1f}% (RouterBench 0.5%, LLMRouterBench 0.8%)"]
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
