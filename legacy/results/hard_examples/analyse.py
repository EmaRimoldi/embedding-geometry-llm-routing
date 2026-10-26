"""exp22: where do the trained pipelines (K-means, kNN, cell classifier, ridge difficulty) put the hard-example probes?
Every model is fit on the full filtered panel (the probes are external). See experiments/exp22_hard_examples.yaml."""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.probes.constants import KMEANS_KS, RIDGE_ALPHAS  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = OUT_OVERRIDE or HERE / "results"; LAMS = (0.0, 50.0, 100.0, 1000.0); MIN_CELL = 50


def unit(a): return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def main() -> None:
    items = json.loads((HERE / "collection.json").read_text())["items"]
    panel = load_panel("routerbench-0shot"); n = panel.n
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = unit(x)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    n_solved = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64); models = [str(m).split("/")[-1] for m in panel.models]
    fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str); rec = rec[panel.keep] if panel.keep is not None else rec
    src = np.array(["wikihow" if re.search(r"\[(header|title|step|substeps)\]", str(t)) else "activitynet" for t in panel.prompts])
    cell_str = np.where(ds == "mmlu", np.array([s_.split(".")[0] for s_ in rec]), np.where(ds == "hellaswag", "hellaswag/" + src, ds))
    cell_names, cell = np.unique(cell_str, return_inverse=True); cell_names = [str(s) for s in cell_names]
    # --- fit everything on the full panel
    print("fitting", flush=True)
    km = {K: KMeans(n_clusters=K, n_init=3, random_state=0).fit(u) for K in KMEANS_KS}
    sc = StandardScaler().fit(x); xs = sc.transform(x)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(xs, cell); ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(xs, n_solved)
    nn = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute", n_jobs=8).fit(u)
    fam_cent = unit(np.vstack([x[fam == g].mean(0) for g in range(F)]))
    U = {lam: r - lam * c for lam in LAMS}
    fam_lookup = {lam: [int(U[lam][fam == g].mean(0).argmax()) for g in range(F)] for lam in LAMS}
    cell_lookup = {lam: [int(U[lam][cell == k].mean(0).argmax()) if (cell == k).sum() >= MIN_CELL else fam_lookup[lam][fam[cell == k][0]] for k in range(len(cell_names))] for lam in LAMS}
    # reference: how the panel's own GSM8K and MMLU-math rows are treated (for comparison)
    def comp(idx): cnt = collections.Counter(ds[idx].tolist()); return {k: v / len(idx) for k, v in cnt.most_common()}
    out = {"items": [], "reference": {"gsm8k_mean_n_solved": float(n_solved[ds == "grade-school-math"].mean()),
                                     "mmlu_math_mean_n_solved": float(n_solved[np.isin(cell_str, ["mmlu-elementary-mathematics", "mmlu-high-school-mathematics", "mmlu-college-mathematics", "mmlu-abstract-algebra", "mmlu-high-school-statistics"])].mean()),
                                     "panel_mean_n_solved": float(n_solved.mean()), "family_lookup_model": {lam: {names[g]: models[m] for g, m in enumerate(fam_lookup[lam])} for lam in LAMS}}}
    md = ["# exp22 — hard examples through the trained pipelines", "", "All models fit on the full filtered panel (34,661 rows). For every probe and text variant: K-means cluster (K = 7 / 64 / 256) with the cluster's majority family and share; 10 and 100 nearest neighbours (family shares, top cells); the neighbours' mean n_solved (what kNN believes the difficulty is); the cell classifier's prediction; ridge's predicted n_solved; the model kNN-100 and the predicted-cell lookup route to at λ ∈ {0, 50, 100, 1000}.", "",
          f"Reference: panel mean n_solved {out['reference']['panel_mean_n_solved']:.2f}; GSM8K mean {out['reference']['gsm8k_mean_n_solved']:.2f}; MMLU-math mean {out['reference']['mmlu_math_mean_n_solved']:.2f}. 'Right' for the open and IMO probes would be: a low predicted n_solved (no 2023 model solves them) and a route to the strongest model at every λ; the easy controls are the opposite case.", ""]
    for variant in ("raw", "gsm_template"):
        E = np.load(HERE / "embeddings" / f"{REP_NAME}__{variant}.npy").astype(np.float32); Eu = unit(E); Es = sc.transform(E)
        d, nb = nn.kneighbors(Eu); pred_cell = clf.predict(Es); proba = clf.predict_proba(Es).max(1); pred_ns = ridge.predict(Es)
        md += [f"## variant: {variant}", "", "id | category | K=7 cluster (majority family, share) | K=64 (majority cell, share) | K=256 (majority cell, share) | 10-NN families | 100-NN families | top cells among 100-NN | 100-NN mean n_solved | ridge n_solved | predicted cell (p) | nearest family centroid (cos) | kNN-100 route λ=0/50/100/1000 | predicted-cell lookup route λ=0/50/100/1000", "---|---|---|---|---|---|---|---|---|---|---|---|---|---"]
        for i, it in enumerate(items):
            row = {"id": it["id"], "category": it["category"], "variant": variant}
            kmc = {}
            for K, m in km.items():
                lab = int(m.predict(Eu[i:i + 1])[0]); members = np.flatnonzero(m.labels_ == lab)
                if K == 7: cnt = collections.Counter(ds[members].tolist()).most_common(1)[0]; kmc[K] = (cnt[0], cnt[1] / len(members), len(members))
                else: cnt = collections.Counter(cell_str[members].tolist()).most_common(1)[0]; kmc[K] = (cnt[0], cnt[1] / len(members), len(members))
            row["kmeans"] = {str(K): {"majority": v[0], "share": v[1], "size": v[2]} for K, v in kmc.items()}
            f10, f100 = comp(nb[i, :10]), comp(nb[i]); topc = collections.Counter(cell_str[nb[i]].tolist()).most_common(3)
            row.update({"nn10_families": f10, "nn100_families": f100, "nn100_top_cells": topc, "nn100_mean_n_solved": float(n_solved[nb[i]].mean()), "nn10_mean_n_solved": float(n_solved[nb[i, :10]].mean()),
                        "nn1_cosine": float(1 - d[i, 0]), "ridge_n_solved": float(pred_ns[i]), "pred_cell": cell_names[pred_cell[i]], "pred_cell_p": float(proba[i]),
                        "nearest_family_centroid": names[int((fam_cent @ Eu[i]).argmax())], "nearest_family_centroid_cos": float((fam_cent @ Eu[i]).max())})
            row["knn100_route"] = {str(lam): models[int(U[lam][nb[i]].mean(0).argmax())] for lam in LAMS}
            row["cell_lookup_route"] = {str(lam): models[cell_lookup[lam][pred_cell[i]]] for lam in LAMS}
            out["items"].append(row)
            fs = lambda dct: ", ".join(f"{k} {v:.1f}" for k, v in list(dct.items())[:3])
            md.append(f"{it['id']} | {it['category']} | {kmc[7][0]} ({kmc[7][1]:.2f}) | {kmc[64][0]} ({kmc[64][1]:.2f}) | {kmc[256][0]} ({kmc[256][1]:.2f}) | {fs(f10)} | {fs(f100)} | " + ", ".join(f"{k} {v}" for k, v in topc) +
                      f" | {row['nn100_mean_n_solved']:.1f} | {row['ridge_n_solved']:.1f} | {row['pred_cell']} ({row['pred_cell_p']:.2f}) | {row['nearest_family_centroid']} ({row['nearest_family_centroid_cos']:.2f}) | " +
                      "/".join(row["knn100_route"][str(l)] for l in LAMS) + " | " + "/".join(row["cell_lookup_route"][str(l)] for l in LAMS))
        md.append("")
    OUT.mkdir(exist_ok=True); (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp22_hard_examples", **out}, indent=1, default=str))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
