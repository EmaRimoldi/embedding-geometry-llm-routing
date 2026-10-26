"""exp01: K-means (K = 7) on the GTE embedding vs the 7 task families. Outcome-free.

See experiments/exp01_kmeans7_vs_families.yaml for the approved settings.
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
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp01_kmeans7_vs_families"
K = N_FAM
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    panel = load_panel(PANEL)
    spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep)
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    fam = panel.dataset_index()
    names = panel.datasets
    out_dir = RESULTS_ROOT / "geometry_tests" / EXP
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    per_seed = {}
    for seed in SEEDS:
        lab = KMeans(n_clusters=K, n_init=3, random_state=seed).fit_predict(u)
        labels[seed] = lab
        cont = np.zeros((K, len(names)), dtype=int)
        for c in range(K):
            cont[c] = np.bincount(fam[lab == c], minlength=len(names))
        per_seed[seed] = {
            "purity": float(cont.max(1).sum() / cont.sum()),
            "nmi": float(normalized_mutual_info_score(fam, lab)),
            "ari": float(adjusted_rand_score(fam, lab)),
            "cluster_sizes": cont.sum(1).tolist(),
            "majority_family_per_cluster": [names[int(j)] for j in cont.argmax(1)],
            "contingency": cont.tolist(),
        }
        print(f"seed {seed}: purity={per_seed[seed]['purity']:.4f} nmi={per_seed[seed]['nmi']:.4f} ari={per_seed[seed]['ari']:.4f}", flush=True)

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
    stability = {f"ari_seed0_vs_seed{s}": float(adjusted_rand_score(labels[SEEDS[0]], labels[s])) for s in SEEDS[1:]}

    result = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "representation_sha256_file": str(spec.path.with_suffix(".npy.json")),
              "k": K, "n_init": 3, "seeds": list(SEEDS), "family_order": names,
              "per_seed": per_seed, "per_family_seed0": per_family, "stability_across_seeds": stability}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))

    # Human-readable table for the reference seed.
    lines = ["cluster | size | " + " | ".join(names), "---|---|" + "---|" * len(names)]
    for c in range(K):
        lines.append(f"{c} | {cont[c].sum()} | " + " | ".join(str(v) for v in cont[c]))
    md = [f"# {EXP}", "", f"Panel: {result['panel']} filtered, {result['rows']} rows. Representation: {spec.name}. K = {K}, unit vectors, n_init = 3.", "",
          f"## Agreement with the {len(names)} families", "", "seed | purity | NMI | ARI", "---|---|---|---"]
    md += [f"{s} | {per_seed[s]['purity']:.4f} | {per_seed[s]['nmi']:.4f} | {per_seed[s]['ari']:.4f}" for s in SEEDS]
    md += ["", "Stability (ARI of seed 0 vs others): " + ", ".join(f"{v:.3f}" for v in stability.values()), "",
           "## Contingency table, seed 0 (rows = clusters, columns = families)", ""] + lines
    md += ["", "## Per family (seed 0)", "", "family | n | clusters where it is the majority | share of family in those clusters | largest single-cluster share | clusters holding >= 5% of it", "---|---|---|---|---|---"]
    md += [f"{n} | {v['n']} | {v['clusters_where_majority']} | {v['share_in_own_clusters']:.3f} | {v['largest_single_cluster_share']:.3f} | {v['n_clusters_holding_ge_5pct']}" for n, v in per_family.items()]
    (out_dir / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
