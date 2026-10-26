"""exp22 - held-out family routing on SPROUT. Headline: six-family holdout (train on five conceptual families, test on
the sixth). Supplementary: source holdout over the 20 released raw identifiers (train on the other 19; a held-out
RAGBench sub-dataset still has eleven RAGBench siblings in training, so this is not family shift). Methods: best single
model of the training rows, kNN-100 over training rows, K-means-64 utility router, classify-then-family-lookup (linear
family classifier on GTE, standardised, C=1; the held-out query is routed with the predicted training family's best
model), soft family prior (class-balanced posterior x family utility tables), oracle. Utility = reward - lambda*cost at
lambda 0, 0.5, 5, 25, 50, 250; pooled per query over held-out rows and equal weight over held-out units.
Writes results/exp22_sprout_holdout/."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = "exp22_sprout_holdout"; LAMS = (0.0, 0.5, 5.0, 25.0, 50.0, 250.0, 500.0, 1000.0); METHODS = ("best_single_train", "knn100", "kmeans64_top1", "classify_family_lookup", "soft_prior", "oracle")


def holdout(D, unit, name, t0):
    r, c, x = D["r"], D["c"], D["x"]; xu = S.unit(x); F = int(unit.max()) + 1; n = len(r); U_all = {lam: r - lam * c for lam in LAMS}
    per = {m: {f"{l:g}": {"pool_u": 0.0, "pool_n": 0, "units": []} for l in LAMS} for m in METHODS}
    for g in range(F):
        te = np.where(unit == g)[0]; tr = np.where(unit != g)[0]; ii = np.arange(len(te)); ut = unit[tr]; src = np.unique(ut)
        nb = tr[S.knn_idx(xu[te], xu[tr], 100)]; km = KMeans(64, n_init=1, random_state=0).fit(xu[tr]); ktr, kte = km.labels_, km.predict(xu[te])
        sc = StandardScaler().fit(xu[tr]); clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(sc.transform(xu[tr]), ut); post = clf.predict_proba(sc.transform(xu[te])); pred = clf.classes_[post.argmax(1)]
        for lam in LAMS:
            U = U_all[lam]; L = f"{lam:g}"; tab = np.stack([U[tr][ut == s].mean(0) for s in src]); ch = {"best_single_train": np.full(len(te), int(U[tr].mean(0).argmax())), "oracle": U[te].argmax(1), "knn100": U[nb].mean(1).argmax(1)}
            cm = np.array([U[tr][ktr == k].mean(0).argmax() if (ktr == k).any() else int(U[tr].mean(0).argmax()) for k in range(64)]); ch["kmeans64_top1"] = cm[kte]
            fb = {s: int(tab[i].argmax()) for i, s in enumerate(src)}; ch["classify_family_lookup"] = np.array([fb[p] for p in pred]); ch["soft_prior"] = (post @ tab).argmax(1)
            for m in METHODS:
                u = U[te][ii, ch[m]]; per[m][L]["pool_u"] += float(u.sum()); per[m][L]["pool_n"] += len(te); per[m][L]["units"].append(float(u.mean()))
        print(name, "fold", g, D.get("unit_names", {}).get(g, g), round(time.time() - t0), flush=True)
    return {m: {L: {"pooled_per_query": v["pool_u"] / v["pool_n"], "equal_weight": float(np.mean(v["units"])), "per_unit": v["units"]} for L, v in d.items()} for m, d in per.items()}


def main():
    t0 = time.time(); D = S.load(); res = S.provenance(D); res["experiment"] = EXP; res["lambdas"] = list(LAMS)
    res["six_family"] = holdout(D, D["fam"], "six-family", t0); fn = D["fnames"]
    ds, raw = np.unique(D["dataset"], return_inverse=True); D["unit_names"] = {i: d for i, d in enumerate(ds)}; res["raw20_source_holdout"] = holdout(D, raw, "raw20", t0); res["raw20_units"] = list(ds)
    md = [f"# {EXP} (SPROUT)", ""]
    for key, title, units in (("six_family", "Six-family holdout (headline)", fn), ("raw20_source_holdout", "Source holdout over the 20 released raw identifiers (supplementary; not family shift for RAGBench / MATH / MuSR sub-sources)", list(ds))):
        md += [f"## {title}: pooled per query [equal weight over held-out units]", "", "method | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
        for m in METHODS: md.append(f"{m} | " + " | ".join(f"{res[key][m][f'{l:g}']['pooled_per_query']:.3f} [{res[key][m][f'{l:g}']['equal_weight']:.3f}]" for l in LAMS))
        md += ["", f"per held-out unit at λ=5 (classify_family_lookup / knn100 / best_single_train / oracle): " + "; ".join(f"{units[i]}: {res[key]['classify_family_lookup']['5']['per_unit'][i]:.3f} / {res[key]['knn100']['5']['per_unit'][i]:.3f} / {res[key]['best_single_train']['5']['per_unit'][i]:.3f} / {res[key]['oracle']['5']['per_unit'][i]:.3f}" for i in range(len(units))), ""]
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
