"""exp28: which model does each router choose, as a function of lambda? Distribution of chosen models on test rows (mean over the 5 splits)
for the family lookup, the (family, topic) cell lookup (true labels), the predicted-cell lookup (exp20 classifier), kNN-10, kNN-40, kNN-100 and the oracle;
plus, per family, the model the cell lookup / kNN-100 send most queries to at each lambda.  Writes results/outcome_locality/exp28_choice_by_lambda/."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
OUT = RESULTS_ROOT / "outcome_locality" / "exp28_choice_by_lambda"; KS = (10, 40, 100); MIN_CELL = 50


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    models = [str(m).split("/")[-1] for m in panel.models]; M = len(models)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    cell_str = topic_str(panel); _, cell = np.unique(cell_str, return_inverse=True)
    cell_fam = np.array([fam[cell == k][0] for k in range(cell.max() + 1)])
    METHODS = ["family_lookup", "cell_lookup", "predicted_cell_lookup", "knn10", "knn40", "knn100", "oracle"]
    share = {m: {f"{lam:g}": np.zeros(M) for lam in LAMBDAS} for m in METHODS}; perfam = {m: {f"{lam:g}": np.zeros((F, M)) for lam in LAMBDAS} for m in ("cell_lookup", "knn100")}
    agree = {f"{lam:g}": {"knn100_vs_cell": [], "knn100_vs_family": [], "pred_vs_cell": [], "cell_vs_family": []} for lam in LAMBDAS}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte = fam[te]; idx = np.arange(len(te))
        sc = StandardScaler().fit(x[tr]); pc = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x[tr]), cell[tr]).predict(sc.transform(x[te]))
        nb = tr[NearestNeighbors(n_neighbors=max(KS), metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
        for lam in LAMBDAS:
            L = f"{lam:g}"; Ul = r - lam * c; Utr, Ute = Ul[tr], Ul[te]
            ftab = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)]); ctab = np.array([Utr[cell[tr] == k].mean(0).argmax() if (cell[tr] == k).sum() >= MIN_CELL else ftab[cell_fam[k]] for k in range(cell.max() + 1)])
            ch = {"family_lookup": ftab[fte], "cell_lookup": ctab[cell[te]], "predicted_cell_lookup": ctab[pc], "oracle": Ute.argmax(1)}
            for k in KS: ch[f"knn{k}"] = Ul[nb[:, :k]].mean(1).argmax(1)
            for m in METHODS: share[m][L] += np.bincount(ch[m], minlength=M) / len(te) / len(SEEDS)
            for m in ("cell_lookup", "knn100"):
                for g in range(F): perfam[m][L][g] += np.bincount(ch[m][fte == g], minlength=M) / max((fte == g).sum(), 1) / len(SEEDS)
            agree[L]["knn100_vs_cell"].append(float(np.mean(ch["knn100"] == ch["cell_lookup"]))); agree[L]["knn100_vs_family"].append(float(np.mean(ch["knn100"] == ch["family_lookup"])))
            agree[L]["pred_vs_cell"].append(float(np.mean(ch["predicted_cell_lookup"] == ch["cell_lookup"]))); agree[L]["cell_vs_family"].append(float(np.mean(ch["cell_lookup"] == ch["family_lookup"])))
        print(f"seed {seed} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({**panel.provenance(), "experiment": "exp28_choice_by_lambda", "models": models, "share": {m: {L: v.tolist() for L, v in d.items()} for m, d in share.items()},
               "per_family": {m: {L: v.tolist() for L, v in d.items()} for m, d in perfam.items()}, "families": names, "agreement": {L: {k: float(np.mean(v)) for k, v in d.items()} for L, d in agree.items()}}, open(OUT / "result.json", "w"), indent=1)
    md = ["# exp28 — which model is chosen, by λ (share of test queries; mean over 5 splits)", ""]
    for m in METHODS:
        md += [f"## {m}", "", "λ | " + " | ".join(models), "---|" + "---|" * M] + [f"{L} | " + " | ".join(f"{v:.2f}" for v in share[m][L]) for L in share[m]] + [""]
    md += ["## Agreement of chosen model (share of test queries)", "", "λ | kNN-100 = cell lookup | kNN-100 = family lookup | predicted-cell = cell lookup | cell lookup = family lookup", "---|---|---|---|---"]
    md += [f"{L} | {d['knn100_vs_cell']:.3f} | {d['knn100_vs_family']:.3f} | {d['pred_vs_cell']:.3f} | {d['cell_vs_family']:.3f}" for L, d in {L: {k: float(np.mean(v)) for k, v in dd.items()} for L, dd in agree.items()}.items()]
    for m in ("cell_lookup", "knn100"):
        md += ["", f"## {m}: most-chosen model per family (share of that family's test rows)", "", "λ | " + " | ".join(names), "---|" + "---|" * F]
        for L in perfam[m]: md.append(f"{L} | " + " | ".join(f"{models[int(perfam[m][L][g].argmax())]} {perfam[m][L][g].max():.2f}" for g in range(F)))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
