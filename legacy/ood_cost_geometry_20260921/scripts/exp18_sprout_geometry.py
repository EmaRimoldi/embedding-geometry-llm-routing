"""exp18 - Section 3 geometry measurements on SPROUT (GTE-Qwen2-7B, six conceptual families).
Claim 1: K-means with K = 6 on unit vectors (5 seeds): purity, NMI; balanced-subsample NMI; hashed bag-of-words K-means
purity; same-family share of the k nearest neighbours (k = 10, 100, 1000; leave-one-out) with chance.
Claim 2: within each family that carries topic labels, K-means with K = number of topics: NMI for the embedder and for
bag-of-words, chance = NMI against permuted labels.
Difficulty locality (Table difficulty-local): five family-stratified 70/30 splits; for each test query the k = 10 nearest
training rows inside its (family, topic) cell (cells with >= 50 training rows; family otherwise); percentage reduction of
the mean absolute outcome difference relative to all training rows of the cell, for three proxies (fraction of models
solving, mean reward, model-specific reward profile); per query and equal weight over families; bootstrap CI over queries.
Writes results/exp18_sprout_geometry/."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import sprout_common as S

EXP = "exp18_sprout_geometry"; import os; EXP = EXP + ("_v2" if os.environ.get("SPROUT_TOPICS") == "v2" else ""); SEEDS = (0, 1, 2, 3, 4); MIN_CELL = 50; K_LOCAL = 10


def purity(lab, fam):
    tot = 0
    for k in np.unique(lab):
        m = lab == k; tot += np.bincount(fam[m]).max()
    return tot / len(lab)


def main():
    t0 = time.time(); D = S.load(); r, c, fam, fn, x = D["r"], D["c"], D["fam"], D["fnames"], D["x"]; F, n = len(fn), D["n"]; xu = S.unit(x); del x
    print("loaded", n, "rows", F, "families", time.time() - t0, flush=True)
    res = S.provenance(D); res["experiment"] = EXP; md = [f"# {EXP} (SPROUT, GTE-Qwen2-7B)", "", f"Rows {n} (of {D['n_total']}; {D['n_total']-D['n_kept']} above the GTE context, {D['n_kept']-n} exact duplicate prompts removed). Families: " + ", ".join(f"{f} {int((fam==g).sum())}" for g, f in enumerate(fn)) + f". Topic sources: {res['topic_sources']}; cells {res['n_cells']}.", ""]
    # ---- Claim 1: K-means K=F ----
    pur, nm = [], []
    for seed in SEEDS:
        lab = KMeans(F, n_init=1, random_state=seed).fit_predict(xu); pur.append(purity(lab, fam)); nm.append(nmi(fam, lab))
    m = min(np.bincount(fam)); rng = np.random.default_rng(0); nmb = []
    for seed in SEEDS:
        sub = np.concatenate([rng.choice(np.where(fam == g)[0], m, replace=False) for g in range(F)]); lab = KMeans(F, n_init=1, random_state=seed).fit_predict(xu[sub]); nmb.append(nmi(fam[sub], lab))
    print("kmeans done", time.time() - t0, flush=True)
    hv = HashingVectorizer(n_features=2 ** 18, alternate_sign=False, norm="l2"); B = hv.transform(list(D["prompts"])); purb, nmbow = [], []
    for seed in SEEDS:
        lab = KMeans(F, n_init=1, random_state=seed).fit_predict(B); purb.append(purity(lab, fam)); nmbow.append(nmi(fam, lab))
    print("bow kmeans done", time.time() - t0, flush=True)
    nb = S.knn_idx(xu, xu, 1000, exclude_self=True); same = fam[nb] == fam[:, None]
    share = {k: float(same[:, :k].mean()) for k in (10, 100, 1000)}; share_fam = {k: {fn[g]: float(same[fam == g, :k].mean()) for g in range(F)} for k in (10, 100, 1000)}
    p = np.bincount(fam) / n; chance = float(sum(p[g] * (np.bincount(fam)[g] - 1) / (n - 1) for g in range(F)))
    res["claim1"] = {"kmeans_purity": pur, "kmeans_nmi": nm, "balanced_subsample_nmi": nmb, "balanced_per_family": int(m), "bow_kmeans_purity": purb, "bow_kmeans_nmi": nmbow, "same_family_share": share, "same_family_share_per_family": share_fam, "chance": chance}
    md += ["## Claim 1 (family first)", "", f"K-means K={F} on unit GTE, 5 seeds: purity {min(pur):.3f}-{max(pur):.3f}, NMI {min(nm):.3f}-{max(nm):.3f}; balanced subsample ({m} per family) NMI {min(nmb):.3f}-{max(nmb):.3f}. Hashed bag-of-words: purity {min(purb):.3f}-{max(purb):.3f}, NMI {min(nmbow):.3f}-{max(nmbow):.3f}.",
           f"Same-family share of the k nearest (leave-one-out): k=10 {share[10]:.3f}, k=100 {share[100]:.3f}, k=1000 {share[1000]:.3f}; chance {chance:.3f}.", "", "family | k=10 | k=100 | k=1000", "---|---|---|---"] + [f"{f} | {share_fam[10][f]:.3f} | {share_fam[100][f]:.3f} | {share_fam[1000][f]:.3f}" for f in fn] + [""]
    print("knn done", time.time() - t0, flush=True)
    # ---- Claim 2: topic recovery inside families ----
    res["claim2"] = {}; md += ["## Claim 2 (topic within a family): K-means with K = number of topics; NMI embedder / bag-of-words / chance (permuted labels)", "", "family | label set | rows | topics | NMI GTE | NMI BoW | chance", "---|---|---|---|---|---|---"]
    sets = [("mmlupro", "authoritative category", (fam == fn.index("mmlupro")) & (D["topic_src"] == "authoritative")), ("math", "authoritative type", (fam == fn.index("math")) & (D["topic_src"] == "authoritative")),
            ("openhermes", "Qwen subject (accepted)", (fam == fn.index("openhermes")) & (np.char.startswith(D["topic_src"].astype(str), "qwen"))), ("gpqa", "Qwen subject (accepted)", (fam == fn.index("gpqa")) & (np.char.startswith(D["topic_src"].astype(str), "qwen"))),
            ("ragbench", "Qwen subject (accepted)", (fam == fn.index("ragbench")) & (np.char.startswith(D["topic_src"].astype(str), "qwen"))), ("ragbench", "released sub-dataset", fam == fn.index("ragbench")), ("musr", "released task", fam == fn.index("musr"))]
    for f, lab_name, mask in sets:
        rows_ = np.where(mask)[0]
        if len(rows_) < 100: continue
        lab = D["dataset"][rows_] if lab_name == "released sub-dataset" else D["topic"][rows_]; _, t = np.unique(lab, return_inverse=True); K = int(t.max()) + 1
        if K < 2 or len(rows_) < 100: continue
        g_e, g_b, ch = [], [], []
        for seed in SEEDS[:3]:
            le = KMeans(K, n_init=1, random_state=seed).fit_predict(xu[rows_]); lb = KMeans(K, n_init=1, random_state=seed).fit_predict(B[rows_]); g_e.append(nmi(t, le)); g_b.append(nmi(t, lb)); ch.append(nmi(np.random.default_rng(seed).permutation(t), le))
        res["claim2"][f"{f}|{lab_name}"] = {"rows": int(len(rows_)), "topics": K, "nmi_gte": g_e, "nmi_bow": g_b, "chance": ch}
        md.append(f"{f} | {lab_name} | {len(rows_)} | {K} | {min(g_e):.3f}-{max(g_e):.3f} | {min(g_b):.3f}-{max(g_b):.3f} | {np.mean(ch):.3f}"); print("claim2", f, lab_name, time.time() - t0, flush=True)
    md.append("")
    # ---- Difficulty locality ----
    solved = (r >= 0.5).astype(float); prox = {"fraction of models solving": solved.mean(1), "mean model reward": r.mean(1)}
    _, cell = np.unique(D["cell_str"], return_inverse=True)
    acc = {p_: {"nn": np.zeros(n), "all": np.zeros(n), "cnt": np.zeros(n)} for p_ in list(prox) + ["model-specific reward profile"]}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=fam); trs = set(tr.tolist())
        cnt = np.bincount(cell[tr], minlength=cell.max() + 1)
        key_te = np.where(cnt[cell[te]] >= MIN_CELL, cell[te], -1 - fam[te]); key_tr = np.where(cnt[cell[tr]] >= MIN_CELL, cell[tr], -1 - fam[tr])
        for kk in np.unique(key_te):
            qi = te[key_te == kk]; ti = tr[key_tr == kk]
            if len(ti) < K_LOCAL + 1: continue
            for s in range(0, len(qi), 512):
                q = qi[s:s + 512]; Sm = xu[q] @ xu[ti].T; p = np.argpartition(-Sm, K_LOCAL - 1, axis=1)[:, :K_LOCAL]; nnr = ti[p]
                for p_, v in prox.items():
                    d_all = np.abs(v[q][:, None] - v[ti][None, :]).mean(1); d_nn = np.abs(v[q][:, None] - v[nnr]).mean(1)
                    acc[p_]["nn"][q] += d_nn; acc[p_]["all"][q] += d_all; acc[p_]["cnt"][q] += 1
                d_all = np.abs(r[q][:, None, :] - r[ti][None, :, :]).mean((1, 2)); d_nn = np.abs(r[q][:, None, :] - r[nnr]).mean((1, 2))
                acc["model-specific reward profile"]["nn"][q] += d_nn; acc["model-specific reward profile"]["all"][q] += d_all; acc["model-specific reward profile"]["cnt"][q] += 1
        print("locality seed", seed, time.time() - t0, flush=True)
    res["difficulty_locality"] = {}; md += [f"## Difficulty locality at k={K_LOCAL} inside the (family, topic) cell (5 splits; % reduction of mean |outcome difference| vs all training rows of the cell; 95% bootstrap CI over unique held-out queries)", "", "proxy | per query | equal weight over families", "---|---|---"]
    rng = np.random.default_rng(0)
    for p_, a in acc.items():
        ok = a["cnt"] > 0; nn_ = a["nn"][ok] / a["cnt"][ok]; al = a["all"][ok] / a["cnt"][ok]; fo = fam[ok]
        def red(idx): 
            return 100 * (1 - nn_[idx].sum() / al[idx].sum())
        def red_eq(idx):
            return 100 * np.mean([1 - nn_[idx][fo[idx] == g].sum() / al[idx][fo[idx] == g].sum() for g in range(F) if (fo[idx] == g).any()])
        bs_q = [red(rng.integers(0, ok.sum(), ok.sum())) for _ in range(300)]; bs_e = [red_eq(rng.integers(0, ok.sum(), ok.sum())) for _ in range(300)]
        per_fam = {fn[g]: float(100 * (1 - nn_[fo == g].sum() / al[fo == g].sum())) for g in range(F)}
        res["difficulty_locality"][p_] = {"per_query": [red(np.arange(ok.sum())), np.percentile(bs_q, 2.5), np.percentile(bs_q, 97.5)], "equal_weight": [red_eq(np.arange(ok.sum())), np.percentile(bs_e, 2.5), np.percentile(bs_e, 97.5)], "per_family": per_fam, "queries": int(ok.sum())}
        v = res["difficulty_locality"][p_]; md.append(f"{p_} | {v['per_query'][0]:.1f} [{v['per_query'][1]:.1f}, {v['per_query'][2]:.1f}] | {v['equal_weight'][0]:.1f} [{v['equal_weight'][1]:.1f}, {v['equal_weight'][2]:.1f}]")
    md += ["", "per family (fraction of models solving): " + ", ".join(f"{k} {v:.1f}" for k, v in res["difficulty_locality"]["fraction of models solving"]["per_family"].items()), ""]
    # solve-count read-out: variance of the solve fraction explained by the cell mean (train->test)
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
