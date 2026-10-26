"""exp27 - the paper's Section 3 / Section 4 / held-out measurements with the SAME classifiers on another representation.
rep = 'gte' (emitted GTE-Qwen2-7B vector) or 'hidden/<model>/<file>' e.g. hidden/qwen2.5-7b-instruct/L21_last,
hidden/gte-qwen2-7b/L21_last, hidden/qwen2.5-coder-1.5b-instruct/L28_last, hidden/qwen3-embedding-0.6b/L28_last.
Section 3: K-means K=F purity/NMI (3 seeds), same-family share of k=10/100 neighbours (chance), same-topic share of the
10 nearest inside the topic family (RB: MMLU subjects; LRB: MMLU-Pro categories) with chance and a linear probe,
difficulty locality k=10 inside the (family, topic) cell (5 splits, solve-fraction proxy).
Section 4 (5 stratified splits): kNN-100, K-means-64 utility router, predicted-cell lookup (accuracy), family lookup,
cell lookup (label-based, representation-free, for reference), best single, oracle.
Held-out family: soft prior (posterior from this rep), classify-then-family-lookup, kNN-100, training best single, ceiling.
All utilities at C.LAMS; per query [equal weight]. Usage: exp27 <panel> <rep>."""
from __future__ import annotations
import re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import common as C
from rb.panelctx import topic_str
from sprout_common import knn_idx, unit
from exp19_sprout_routing import lookup

EXP = "exp27_representation_swap"; MIN_CELL = 50; SEEDS = (0, 1, 2, 3, 4)


def load_rep(D, panel_name, rep):
    if rep == "gte": return D["x"]
    h = np.load(C.E4 / "features" / panel_name / "prompt" / (rep + ".npy"), mmap_mode="r")
    return np.asarray(h[D["panel"].keep] if h.shape[0] != len(D["r"]) else h, dtype=np.float32)


def purity(lab, fam):
    return sum(np.bincount(fam[lab == k]).max() for k in np.unique(lab)) / len(lab)


def main():
    panel_name, rep = sys.argv[1], sys.argv[2]; t0 = time.time(); D = C.load(panel_name); r, c, fam, fn, names = D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, n, M = len(fn), len(r), len(names)
    xu = unit(load_rep(D, panel_name, rep)); cell_str = np.asarray([str(t) for t in topic_str(D["panel"])], dtype=object); _, cell = np.unique(cell_str, return_inverse=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", rep).strip("_"); res = C.provenance(D); res.update({"experiment": EXP, "representation": rep, "dim": int(xu.shape[1])}); md = [f"# {EXP} ({panel_name}, rep = {rep}, dim {xu.shape[1]})", ""]
    # ---- Section 3 ----
    pur, nm = [], []
    for s in range(3): lab = KMeans(F, n_init=1, random_state=s).fit_predict(xu); pur.append(purity(lab, fam)); nm.append(nmi(fam, lab))
    nb = knn_idx(xu, xu, 100, exclude_self=True); same = fam[nb] == fam[:, None]; sh10, sh100 = float(same[:, :10].mean()), float(same.mean()); p = np.bincount(fam) / n; chance = float(sum(p[g] * (np.bincount(fam)[g] - 1) / (n - 1) for g in range(F)))
    tf = [g for g, f in enumerate(fn) if "mmlu" in f.lower()]; topic = {}
    if tf:
        rows_ = np.where(fam == tf[0])[0]; _, t = np.unique(cell_str[rows_], return_inverse=True); K = int(t.max()) + 1
        if K > 1:
            nbt = knn_idx(xu[rows_], xu[rows_], 10, exclude_self=True); share = float((t[nbt] == t[:, None]).mean()); pt = np.bincount(t) / len(t)
            tr, te = train_test_split(np.arange(len(rows_)), test_size=0.3, random_state=0, stratify=t if np.bincount(t).min() >= 2 else None); sc = StandardScaler().fit(xu[rows_][tr]); acc = float((LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(xu[rows_][tr]), t[tr]).predict(sc.transform(xu[rows_][te])) == t[te]).mean())
            topic = {"family": fn[tf[0]], "topics": K, "same_topic_share_k10": share, "chance": float((pt ** 2).sum()), "probe_acc": acc, "majority": float(pt.max())}
    s_ = (r >= 0.5).mean(1); acc_nn = np.zeros(n); acc_all = np.zeros(n); cnt = np.zeros(n)
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=fam); ctr = np.bincount(cell[tr], minlength=cell.max() + 1)
        kte = np.where(ctr[cell[te]] >= MIN_CELL, cell[te], -1 - fam[te]); ktr = np.where(ctr[cell[tr]] >= MIN_CELL, cell[tr], -1 - fam[tr])
        for kk in np.unique(kte):
            qi = te[kte == kk]; ti = tr[ktr == kk]
            if len(ti) < 11: continue
            for s0 in range(0, len(qi), 1024):
                q = qi[s0:s0 + 1024]; Sm = xu[q] @ xu[ti].T; pp = np.argpartition(-Sm, 9, axis=1)[:, :10]; nnr = ti[pp]
                acc_nn[q] += np.abs(s_[q][:, None] - s_[nnr]).mean(1); acc_all[q] += np.abs(s_[q][:, None] - s_[ti][None, :]).mean(1); cnt[q] += 1
    ok = cnt > 0; loc = float(100 * (1 - (acc_nn[ok] / cnt[ok]).sum() / (acc_all[ok] / cnt[ok]).sum())); loc_eq = float(np.mean([100 * (1 - (acc_nn[ok & (fam == g)] / cnt[ok & (fam == g)]).sum() / (acc_all[ok & (fam == g)] / cnt[ok & (fam == g)]).sum()) for g in range(F) if (ok & (fam == g)).any()]))
    res["section3"] = {"kmeans_purity": pur, "kmeans_nmi": nm, "same_family_k10": sh10, "same_family_k100": sh100, "chance": chance, "topic": topic, "difficulty_locality_k10_pct": loc, "difficulty_locality_k10_pct_equal_weight": loc_eq}
    md += ["## Section 3", "", f"K-means K={F}: purity {min(pur):.3f}-{max(pur):.3f}, NMI {min(nm):.3f}-{max(nm):.3f}; same-family share k=10 {sh10:.3f}, k=100 {sh100:.3f} (chance {chance:.3f})"] + ([f"topic inside {topic['family']} ({topic['topics']} labels): same-topic share k=10 {topic['same_topic_share_k10']:.3f} (chance {topic['chance']:.3f}), probe {topic['probe_acc']:.3f} (majority {topic['majority']:.3f})"] if topic else []) + [f"difficulty locality k=10 inside cells: {loc:.1f}% per query, {loc_eq:.1f}% equal weight", ""]
    print("section 3 done", round(time.time() - t0), flush=True)
    # ---- Section 4 ----
    S4 = ("best_single", "family_lookup", "cell_lookup", "predicted_cell_lookup", "knn100", "kmeans64_top1", "oracle"); per = {m: {f"{l:g}": {"qw": [], "eq": []} for l in C.LAMS} for m in S4}; pacc = []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=fam); ii = np.arange(len(te)); nb = tr[knn_idx(xu[te], xu[tr], 100)]
        sc = StandardScaler().fit(xu[tr]); pc = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(xu[tr]), cell[tr]).predict(sc.transform(xu[te])); pacc.append(float((pc == cell[te]).mean())); cp = cell.copy(); cp[te] = pc
        km = KMeans(64, n_init=1, random_state=seed).fit(xu[tr]); ktr, kte = km.labels_, km.predict(xu[te])
        for lam in C.LAMS:
            U = r - lam * c; Ute = U[te]; L = f"{lam:g}"; fb = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)])
            cm = np.array([U[tr][ktr == k].mean(0).argmax() if (ktr == k).any() else int(U[tr].mean(0).argmax()) for k in range(64)])
            ch = {"best_single": np.full(len(te), int(U[tr].mean(0).argmax())), "family_lookup": fb[fam[te]], "cell_lookup": lookup(U, tr, te, fam, cell, F), "predicted_cell_lookup": lookup(U, tr, te, fam, cp, F), "knn100": U[nb].mean(1).argmax(1), "kmeans64_top1": cm[kte], "oracle": Ute.argmax(1)}
            for m in S4: u = Ute[ii, ch[m]]; per[m][L]["qw"].append(float(u.mean())); per[m][L]["eq"].append(float(np.mean([u[fam[te] == g].mean() for g in range(F)])))
        print("section 4 seed", seed, round(time.time() - t0), flush=True)
    res["section4"] = {m: {L: [float(np.mean(v["qw"])), float(np.mean(v["eq"]))] for L, v in d.items()} for m, d in per.items()}; res["predicted_cell_accuracy"] = pacc
    md += [f"## Section 4 (5 splits; predicted-cell accuracy {np.mean(pacc):.3f}); per query [equal weight]", "", "router | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)] + [f"{m} | " + " | ".join(f"{res['section4'][m][f'{l:g}'][0]:.3f} [{res['section4'][m][f'{l:g}'][1]:.3f}]" for l in C.LAMS) for m in S4] + [""]
    # ---- held-out family ----
    H = ("best_single_train", "knn100", "classify_family_lookup", "soft_prior", "family_ceiling", "oracle"); ho = {m: {f"{l:g}": [] for l in C.LAMS} for m in H}; nte = []
    for g in range(F):
        te = np.where(fam == g)[0]; tr = np.where(fam != g)[0]; ii = np.arange(len(te)); src = [s for s in range(F) if s != g]; nte.append(len(te)); nb = tr[knn_idx(xu[te], xu[tr], 100)]
        sc = StandardScaler().fit(xu[tr]); clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(sc.transform(xu[tr]), fam[tr]); post = clf.predict_proba(sc.transform(xu[te]))[:, [list(clf.classes_).index(s) for s in src]]
        R = np.stack([r[fam == s].mean(0) for s in src]); Cc = np.stack([c[fam == s].mean(0) for s in src]); rt, ct = r[te].mean(0), c[te].mean(0)
        for lam in C.LAMS:
            U = r - lam * c; Ute = U[te]; L = f"{lam:g}"; T = R - lam * Cc
            ch = {"best_single_train": np.full(len(te), int(U[tr].mean(0).argmax())), "knn100": U[nb].mean(1).argmax(1), "classify_family_lookup": T[post.argmax(1)].argmax(1), "soft_prior": (post @ T).argmax(1), "family_ceiling": np.full(len(te), int((rt - lam * ct).argmax())), "oracle": Ute.argmax(1)}
            for m in H: ho[m][L].append(float(Ute[ii, ch[m]].mean()))
    w = np.array(nte) / sum(nte); res["heldout"] = {m: {L: [float(np.dot(w, v)), float(np.mean(v)), dict(zip(fn, v))] for L, v in d.items()} for m, d in ho.items()}
    md += ["## Held-out family; pooled per query [equal weight]", "", "method | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)] + [f"{m} | " + " | ".join(f"{res['heldout'][m][f'{l:g}'][0]:.3f} [{res['heldout'][m][f'{l:g}'][1]:.3f}]" for l in C.LAMS) for m in H]
    C.save(C.RESULTS / EXP / panel_name / tag, res, md)


if __name__ == "__main__":
    main()
