"""Anatomy of the clustering router: what does K-means on the embedding partition?

Outcome-free partition (K-means on unit vectors, K in {7, 64, 256}) followed by a
descriptive read-out with outcomes:

  * per cluster: family composition (majority family share, entropy over families);
  * "pure" clusters (majority family >= 90%) vs "merged" clusters;
  * for merged clusters: which families are merged, and whether the cluster's
    routed model (argmax of the cluster's mean reward) is also the best model of
    EACH member family inside that cluster ("informed merge") or only of the
    dominant family ("swamped");
  * within a pure cluster: does the cluster's model differ from its family's
    modal model, and if so does the cluster gain over the family model on its
    own queries (sub-family specialisation) or lose (noise)?

Writes results/anatomy/<panel>/<rep>__K<k>.json and a summary CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, build_targets, load_panel  # noqa: E402
from rb.reps import list_reps, load_rep, safe_name  # noqa: E402

OUT = ANALYSIS_ROOT / "results/anatomy"


def entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def analyse(x: np.ndarray, t, families: np.ndarray, k: int, seed: int = 0) -> dict:
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(u)
    r = np.nan_to_num(t.reward, nan=0.0)
    n_fam = int(t.dataset_index.max()) + 1
    fam_mean = np.vstack([r[t.dataset_index == g].mean(0) for g in range(n_fam)])
    fam_best = fam_mean.argmax(1)                    # family modal model (full-panel, descriptive)
    clusters = []
    for c in range(k):
        rows = np.flatnonzero(labels == c)
        if len(rows) == 0:
            continue
        comp = np.bincount(t.dataset_index[rows], minlength=n_fam)
        maj = int(comp.argmax())
        share = float(comp[maj] / len(rows))
        cmean = r[rows].mean(0)
        c_model = int(cmean.argmax())
        members = [g for g in range(n_fam) if comp[g] >= max(10, 0.05 * len(rows))]
        per_member = {}
        for g in members:
            rg = rows[t.dataset_index[rows] == g]
            best_here = int(r[rg].mean(0).argmax())
            per_member[str(families[g])] = {
                "n": int(len(rg)), "best_model_in_cluster": best_here,
                "cluster_model_reward": float(r[rg, c_model].mean()),
                "family_model_reward": float(r[rg, fam_best[g]].mean()),
                "best_in_cluster_reward": float(r[rg, best_here].mean()),
            }
        clusters.append({
            "cluster": c, "n": int(len(rows)), "majority_family": str(families[maj]), "majority_share": share,
            "entropy": entropy(comp), "pure": share >= 0.9, "cluster_model": c_model,
            "cluster_model_is_family_model": bool(c_model == fam_best[maj]),
            "gain_vs_family_model_on_own_queries": float(r[rows, c_model].mean() - r[rows, fam_best[t.dataset_index[rows]]].mean()),
            "members": per_member,
        })
    # Contingency table (clusters x families) and agreement with the family labels.
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    contingency = np.zeros((k, n_fam), dtype=int)
    for c in range(k):
        contingency[c] = np.bincount(t.dataset_index[labels == c], minlength=n_fam)
    agreement = {
        "nmi_vs_family": float(normalized_mutual_info_score(t.dataset_index, labels)),
        "ari_vs_family": float(adjusted_rand_score(t.dataset_index, labels)),
        "purity_vs_family": float(contingency.max(1).sum() / contingency.sum()),
        "contingency_rows_clusters_cols_families": contingency.tolist(),
        "family_order": [str(f) for f in families],
    }
    df = pd.DataFrame(clusters)
    pure = df[df.pure]
    merged = df[~df.pure]
    w = df.n / df.n.sum()
    summary = {
        "k": k, "n_clusters_nonempty": int(len(df)),
        "query_weighted_majority_share": float((df.majority_share * w).sum()),
        "frac_queries_in_pure_clusters": float(pure.n.sum() / df.n.sum()),
        "frac_pure_clusters_using_family_model": float(pure.cluster_model_is_family_model.mean()) if len(pure) else float("nan"),
        "pure_clusters_deviating": int((~pure.cluster_model_is_family_model).sum()) if len(pure) else 0,
        "pure_deviating_mean_gain": float(pure[~pure.cluster_model_is_family_model].gain_vs_family_model_on_own_queries.mean()) if len(pure) and (~pure.cluster_model_is_family_model).any() else float("nan"),
        "pure_deviating_frac_positive_gain": float((pure[~pure.cluster_model_is_family_model].gain_vs_family_model_on_own_queries > 0).mean()) if len(pure) and (~pure.cluster_model_is_family_model).any() else float("nan"),
        "merged_clusters": int(len(merged)), "frac_queries_in_merged_clusters": float(merged.n.sum() / df.n.sum()) if len(merged) else 0.0,
    }
    # Merged clusters: informed merge = cluster model is the best model for every member family inside the cluster.
    informed, swamped, pairs = 0, 0, {}
    for _, row in merged.iterrows():
        mem = row["members"]
        if len(mem) < 2:
            continue
        ok = all(v["best_model_in_cluster"] == row["cluster_model"] for v in mem.values())
        informed += ok
        swamped += (not ok)
        key = " + ".join(sorted(mem))
        pairs[key] = pairs.get(key, 0) + 1
    summary.update({"merged_informed": informed, "merged_swamped": swamped, "merged_family_combinations": pairs,
                    "nmi_vs_family": agreement["nmi_vs_family"], "ari_vs_family": agreement["ari_vs_family"],
                    "purity_vs_family": agreement["purity_vs_family"]})
    return {"summary": summary, "agreement_with_families": agreement, "clusters": clusters}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="routerbench-0shot")
    p.add_argument("--reps", nargs="*", default=None)
    p.add_argument("--ks", nargs="*", type=int, default=[7, 64, 256])
    args = p.parse_args()
    panel = load_panel(args.panel)
    t = build_targets(panel)
    families = np.asarray(panel.datasets)
    rows = []
    for spec in list_reps(args.panel):
        if args.reps and spec.name not in args.reps:
            continue
        x = load_rep(spec, panel.n, panel.keep)
        for k in args.ks:
            out = OUT / args.panel / f"{safe_name(spec.name)}__K{k}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            res = analyse(x, t, families, k)
            res["provenance"] = panel.provenance()
            out.write_text(json.dumps(res, indent=1, default=float))
            s = res["summary"]
            rows.append({"rep": spec.name, **{kk: v for kk, v in s.items() if kk != "merged_family_combinations"}})
            print(f"{spec.name} K={k}: pure_q={s['frac_queries_in_pure_clusters']:.3f} pure_use_family_model={s['frac_pure_clusters_using_family_model']:.3f} "
                  f"pure_dev={s['pure_clusters_deviating']} gain={s['pure_deviating_mean_gain']:+.4f} merged={s['merged_clusters']} "
                  f"informed={s['merged_informed']} swamped={s['merged_swamped']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"summary_{args.panel}.csv", index=False)


if __name__ == "__main__":
    main()
