"""exp32: leave-one-family-out generalization (ProbeDirichlet / kNN-paper style). For each family f: train on the other six, test on f.
Methods on the held-out family: kNN k in {5,10,40,100}; K-means cluster routers K in {7,64,256} (PLAN recipe, fitted on training rows, argmax of cluster mean utility);
classify-then-lookup with the family classifier and with the cell classifier (both trained on the seen families only); best single model (on train); zero router (argmax of
the training mean); oracle; and, as the unattainable reference, the lookup with the held-out family's TRUE label fitted on its own rows (in-domain ceiling of family knowledge).
Representations: templated prompt and deformatted core.  Utility per query on the held-out family at the 15-lambda grid; summaries per held-out family, pooled per query
over the 7 folds, and equal-weight over families.  Writes results/generalization/exp32_lofo/."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results" / "deformatting"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
OUT = RESULTS_ROOT / "generalization" / "exp32_lofo"; KS = (5, 10, 40, 100); KM = (N_FAM, 64, 256); MIN_CELL = 50; REPORT = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0)


def unit(a): return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    models = [str(m).split("/")[-1] for m in panel.models]; r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    cell_str = topic_str(panel); _, cell = np.unique(cell_str, return_inverse=True)
    cell_fam = np.array([fam[cell == k][0] for k in range(cell.max() + 1)])
    X = {"prompt": load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32), "core": core_rep(panel)}
    METHODS = [f"knn{k}" for k in KS] + [f"kmeans{K}" for K in KM] + ["classify_family_lookup", "classify_cell_lookup", "best_single_train", "zero_router", "oracle", "true_family_lookup_in_domain"]
    res = {rep: {names[g]: {} for g in range(F)} for rep in X}; diag = {rep: {} for rep in X}
    for rep, x in X.items():
        u = unit(x)
        for g in range(F):
            tr = np.flatnonzero(fam != g); te = np.flatnonzero(fam == g); idx = np.arange(len(te))
            sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
            pf = LogisticRegression(C=1.0, max_iter=3000).fit(xtr, fam[tr]).predict(xte); pc = LogisticRegression(C=1.0, max_iter=3000).fit(xtr, cell[tr]).predict(xte)
            nb = tr[NearestNeighbors(n_neighbors=max(KS), metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
            kms = {K: KMeans(n_clusters=K, n_init=3, random_state=0).fit(u[tr]) for K in KM}
            diag[rep][names[g]] = {"classifier_maps_heldout_to": {names[k]: float((pf == k).mean()) for k in np.unique(pf)}, "nn10_family_composition": {names[k]: float((fam[nb[:, :10]] == k).mean()) for k in np.unique(fam[nb[:, :10]])},
                                   "nn1_cosine_mean": float((u[te] * u[nb[:, 0]]).sum(1).mean())}
            for lam in LAMBDAS:
                U = r - lam * c; Utr, Ute = U[tr], U[te]; ftab = {k: Utr[fam[tr] == k].mean(0).argmax() for k in np.unique(fam[tr])}
                ctab = {k: (Utr[cell[tr] == k].mean(0).argmax() if (cell[tr] == k).sum() >= MIN_CELL else ftab[cell_fam[k]]) for k in np.unique(cell[tr])}
                out = {}
                for k in KS: out[f"knn{k}"] = Ute[idx, Utr[np.searchsorted(tr, nb[:, :k])].mean(1).argmax(1)] if False else Ute[idx, U[nb[:, :k]].mean(1).argmax(1)]
                for K, km in kms.items():
                    lab_tr = km.labels_; lab_te = km.predict(u[te]); tab = np.array([Utr[lab_tr == q].mean(0).argmax() if (lab_tr == q).any() else Utr.mean(0).argmax() for q in range(K)]); out[f"kmeans{K}"] = Ute[idx, tab[lab_te]]
                out["classify_family_lookup"] = Ute[idx, np.array([ftab[k] for k in pf])]; out["classify_cell_lookup"] = Ute[idx, np.array([ctab[k] for k in pc])]
                out["best_single_train"] = Ute[:, Utr.mean(0).argmax()]; out["zero_router"] = out["best_single_train"]; out["oracle"] = Ute.max(1)
                out["true_family_lookup_in_domain"] = Ute[:, Ute.mean(0).argmax()]      # the held-out family's own best model (needs its labels and outcomes: ceiling only)
                res[rep][names[g]][f"{lam:g}"] = {m: float(v.mean()) for m, v in out.items()}
            print(f"{rep} held-out {names[g]} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); sizes = {names[g]: int((fam == g).sum()) for g in range(F)}
    summ = {rep: {L: {m: {"pooled_per_query": float(sum(res[rep][f][L][m] * sizes[f] for f in names) / n), "equal_weight": float(np.mean([res[rep][f][L][m] for f in names]))} for m in METHODS} for L in [f"{l:g}" for l in LAMBDAS]} for rep in X}
    (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp32_lofo", "methods": METHODS, "per_family": res, "summary": summ, "diagnostics": diag}, indent=1))
    md = ["# exp32 — leave-one-family-out generalization", "", "Train on six families, test on the seventh; utility per query on the held-out family. 'true_family_lookup_in_domain' uses the held-out family's own labels and outcomes and is a ceiling, not a method. zero_router = best_single_train here (argmax of the training mean utility).", ""]
    for rep in X:
        for agg in ("pooled_per_query", "equal_weight"):
            md += [f"## {rep} — {agg} over the 7 held-out families", "", "λ | " + " | ".join(METHODS), "---|" + "---|" * len(METHODS)]
            md += [f"{l:g} | " + " | ".join(f"{summ[rep][f'{l:g}'][m][agg]:.4f}" for m in METHODS) for l in REPORT] + [""]
        md += [f"## {rep} — per held-out family (λ = 0 / 100 / 1000): kNN-100 | K-means-64 | classify-cell lookup | best single (train) | oracle | in-domain ceiling", "", "family | " + " | ".join(f"λ={l:g}" for l in (0.0, 100.0, 1000.0)), "---|---|---|---"]
        for f in names:
            md.append(f"{f} | " + " | ".join(" / ".join(f"{res[rep][f][f'{l:g}'][m]:.3f}" for m in ("knn100", "kmeans64", "classify_cell_lookup", "best_single_train", "oracle", "true_family_lookup_in_domain")) for l in (0.0, 100.0, 1000.0)))
        md += ["", f"## {rep} — where the held-out family goes", "", "held-out family | classifier maps it to (share) | 10-NN family composition | mean cosine to nearest training row", "---|---|---|---"]
        for f in names:
            d = diag[rep][f]; md.append(f"{f} | " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(d["classifier_maps_heldout_to"].items(), key=lambda kv: -kv[1])[:3]) + " | " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(d["nn10_family_composition"].items(), key=lambda kv: -kv[1])[:3]) + f" | {d['nn1_cosine_mean']:.3f}")
        md.append("")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
