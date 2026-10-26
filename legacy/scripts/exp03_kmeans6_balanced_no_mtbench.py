"""exp03: K-means K = 6 on a family-balanced subsample with MT-Bench excluded (400 per family). Outcome-free.

See experiments/exp03_kmeans6_balanced_no_mtbench.yaml for the approved settings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp03_kmeans6_balanced_no_mtbench"
EXCLUDE = ("mtbench",)
PER_FAMILY = 400
K = 6
SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    panel = load_panel("routerbench-0shot")
    spec = find_rep("routerbench-0shot", "computed/gte-qwen2-7b@prompt")
    x = load_rep(spec, panel.n, panel.keep)
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    fam_all = panel.dataset_index()
    names_all = panel.datasets
    keep_fam = [j for j, n in enumerate(names_all) if n not in EXCLUDE]
    names = [names_all[j] for j in keep_fam]
    remap = {j: i for i, j in enumerate(keep_fam)}
    in_scope = np.isin(fam_all, keep_fam)
    fam = np.array([remap.get(j, -1) for j in fam_all])
    out_dir = ANALYSIS_ROOT / "results" / "geometry_tests" / EXP
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    per_seed = {}
    subsets = {}
    for seed in SUBSAMPLE_SEEDS:
        rng = np.random.default_rng(seed)
        rows = np.concatenate([rng.choice(np.flatnonzero((fam == j) & in_scope), size=min(PER_FAMILY, int(((fam == j) & in_scope).sum())), replace=False)
                               for j in range(len(names))])
        rows.sort()
        subsets[seed] = rows
        lab_sub = KMeans(n_clusters=K, n_init=3, random_state=0).fit_predict(u[rows])
        lab = np.full(panel.n, -1); lab[rows] = lab_sub
        labels[seed] = lab
        f_sub = fam[rows]
        cont = np.zeros((K, len(names)), dtype=int)
        for c in range(K):
            cont[c] = np.bincount(f_sub[lab_sub == c], minlength=len(names))
        per_seed[seed] = {
            "purity": float(cont.max(1).sum() / cont.sum()),
            "nmi": float(normalized_mutual_info_score(f_sub, lab_sub)),
            "ari": float(adjusted_rand_score(f_sub, lab_sub)),
            "n_rows": int(len(rows)),
            "cluster_sizes": cont.sum(1).tolist(),
            "majority_family_per_cluster": [names[int(j)] for j in cont.argmax(1)],
            "contingency": cont.tolist(),
        }
        print(f"seed {seed}: purity={per_seed[seed]['purity']:.4f} nmi={per_seed[seed]['nmi']:.4f} ari={per_seed[seed]['ari']:.4f}", flush=True)

    SEEDS = SUBSAMPLE_SEEDS
    ref = per_seed[SEEDS[0]]
    cont = np.asarray(ref["contingency"])
    # Per family: share of its queries inside the cluster where it is the majority (if any), and how many clusters it spreads over.
    per_family = {}
    for j, name in enumerate(names):
        col = cont[:, j]
        own = [c for c in range(K) if cont[c].argmax() == j]
        per_family[name] = {
            "n": int(col.sum()),
            "clusters_where_majority": own,
            "share_in_own_clusters": float(col[own].sum() / col.sum()) if own else 0.0,
            "largest_single_cluster_share": float(col.max() / col.sum()),
            "n_clusters_holding_ge_5pct": int((col >= 0.05 * col.sum()).sum()),
        }
    # Stability across subsamples: ARI on the rows shared by seed 0 and seed s (different draws share few rows for big families).
    stability = {}
    for s in SEEDS[1:]:
        shared = np.intersect1d(subsets[SEEDS[0]], subsets[s])
        stability[f"ari_seed0_vs_seed{s}_on_{len(shared)}_shared_rows"] = float(adjusted_rand_score(labels[SEEDS[0]][shared], labels[s][shared])) if len(shared) > 10 else float("nan")

    result = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "representation_sha256_file": str(spec.path.with_suffix(".npy.json")),
              "k": K, "n_init": 3, "per_family": PER_FAMILY, "excluded_families": list(EXCLUDE), "subsample_seeds": list(SEEDS), "kmeans_seed": 0, "family_order": names,
              "per_seed": per_seed, "per_family_seed0": per_family, "stability_across_seeds": stability}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))

    # Human-readable table for the reference seed.
    lines = ["cluster | size | " + " | ".join(names), "---|---|" + "---|" * len(names)]
    for c in range(K):
        lines.append(f"{c} | {cont[c].sum()} | " + " | ".join(str(v) for v in cont[c]))
    md = [f"# {EXP}", "", f"Panel: {result['panel']} filtered, {result['rows']} rows. Representation: {spec.name}. K = {K}, unit vectors, n_init = 3. Balanced subsample: {PER_FAMILY} per family, MT-Bench excluded, {ref['n_rows']} rows per draw, 5 draws.", "",
          f"## Agreement with the {len(names)} families", "", "seed | purity | NMI | ARI", "---|---|---|---"]
    md += [f"{s} | {per_seed[s]['purity']:.4f} | {per_seed[s]['nmi']:.4f} | {per_seed[s]['ari']:.4f}" for s in SEEDS]
    md += ["", "Stability across subsample draws (ARI on shared rows): " + ", ".join(f"{k.split(chr(95)+chr(111)+chr(110))[0]}: {v:.3f}" for k, v in stability.items()), "",
           "## Contingency table, seed 0 (rows = clusters, columns = families)", ""] + lines
    md += ["", "## Per family (seed 0)", "", "family | n | clusters where it is the majority | share of family in those clusters | largest single-cluster share | clusters holding >= 5% of it", "---|---|---|---|---|---"]
    md += [f"{n} | {v['n']} | {v['clusters_where_majority']} | {v['share_in_own_clusters']:.3f} | {v['largest_single_cluster_share']:.3f} | {v['n_clusters_holding_ge_5pct']}" for n, v in per_family.items()]
    (out_dir / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
