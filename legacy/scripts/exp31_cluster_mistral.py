"""exp31: can ANY partition of the embedding isolate regions where Mistral-7B is worth choosing?
K-means on unit GTE (train fold of each split) at K in {7, 64, 256, 1024, 4096}; for each cluster: Mistral's solve rate on the training rows, the cluster router's
choice (argmax of the cluster's mean utility) per lambda; on test rows: share of queries assigned to clusters whose Mistral solve rate is >= 0.5 / 0.7 / 0.9,
Mistral's realised solve rate on those, share of test queries the cluster router sends to Mistral, and the cluster router's utility vs the cell lookup.
Reference partitions: the (family, topic) cells and the true difficulty strata (count of models solving the query; not available at test time, an upper bound
for any partition that tracks difficulty).  Writes results/oracle_gap/exp31_cluster_mistral/."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import split  # noqa: E402
import os
SEEDS = tuple(int(v) for v in os.environ.get("E4_SEEDS", "0,1,2,3,4").split(","))    # E4_SEEDS overrides the default 5 seeds (hedge runs)
from rb.reps import find_rep, load_rep  # noqa: E402
OUT = RESULTS_ROOT / "oracle_gap" / ("exp31_cluster_mistral" + os.environ.get("E4_OUT_SUFFIX", "")); KS = (N_FAM, 64, 256, 1024, 4096); LAMS = (10.0, 50.0, 100.0, 500.0, 1000.0); MIN_CELL = 50; THR = (0.5, 0.7, 0.9)


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    models = [str(m).split("/")[-1] for m in panel.models]; M = len(models); mi = models.index(CHEAP_MODEL)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0); Y = (r >= SUCCESS_THRESHOLD).astype(np.float64); ns = Y.sum(1)
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    cell_str = topic_str(panel); _, cell = np.unique(cell_str, return_inverse=True)
    res = {}
    def evaluate(lab_tr, lab_te, tr, te, name):
        d = res.setdefault(name, {"n_clusters_nonempty": [], "share_test_in_clusters_rate_ge": {str(t): [] for t in THR}, "mistral_rate_on_those": {str(t): [] for t in THR},
                                  "share_sent_to_mistral": {f"{l:g}": [] for l in LAMS}, "utility": {f"{l:g}": [] for l in LAMS}, "cell_lookup_utility": {f"{l:g}": [] for l in LAMS}, "max_cluster_rate_min50": []})
        labs = np.unique(lab_tr); rate = {k: Y[tr][lab_tr == k, mi].mean() for k in labs}; size = {k: int((lab_tr == k).sum()) for k in labs}
        d["n_clusters_nonempty"].append(int(len(labs))); d["max_cluster_rate_min50"].append(float(max([rate[k] for k in labs if size[k] >= 50] or [0])))
        rate_te = np.array([rate.get(k, Y[tr][:, mi].mean()) for k in lab_te])
        for t in THR:
            m = rate_te >= t; d["share_test_in_clusters_rate_ge"][str(t)].append(float(m.mean())); d["mistral_rate_on_those"][str(t)].append(float(Y[te][m, mi].mean()) if m.any() else float("nan"))
        for lam in LAMS:
            Ul = r - lam * c; Utr, Ute = Ul[tr], Ul[te]; glob = Utr.mean(0).argmax(); choice = {k: (Utr[lab_tr == k].mean(0).argmax() if size[k] >= 1 else glob) for k in labs}
            ch = np.array([choice.get(k, glob) for k in lab_te]); d["share_sent_to_mistral"][f"{lam:g}"].append(float((ch == mi).mean())); d["utility"][f"{lam:g}"].append(float(Ute[np.arange(len(te)), ch].mean()))
            ftab = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)]); chc = ftab[fam[te]].copy()
            for g in np.unique(cell[te]):
                rows = cell[tr] == g
                if rows.sum() >= MIN_CELL: chc[cell[te] == g] = Utr[rows].mean(0).argmax()
            d["cell_lookup_utility"][f"{lam:g}"].append(float(Ute[np.arange(len(te)), chc].mean()))
    seeds_done = []
    def write_results():
        OUT.mkdir(parents=True, exist_ok=True); ms = lambda v: float(np.nanmean(v))
        out = {**panel.provenance(), "seeds_done": seeds_done, "experiment": "exp31_cluster_mistral", "partitions": {k: {"n_clusters": ms(d["n_clusters_nonempty"]), "max_cluster_mistral_rate_min50rows": ms(d["max_cluster_rate_min50"]),
                "share_test_in_clusters_rate_ge": {t: ms(v) for t, v in d["share_test_in_clusters_rate_ge"].items()}, "mistral_rate_on_those": {t: ms(v) for t, v in d["mistral_rate_on_those"].items()},
                "share_sent_to_mistral": {l: ms(v) for l, v in d["share_sent_to_mistral"].items()}, "utility": {l: ms(v) for l, v in d["utility"].items()}, "cell_lookup_utility": {l: ms(v) for l, v in d["cell_lookup_utility"].items()}} for k, d in res.items()}}
        (OUT / "result.json").write_text(json.dumps(out, indent=1))
        md = ["# exp31 — can any partition of the embedding isolate regions where Mistral-7B is worth choosing?", "", "K-means on unit GTE fitted on the training fold of each split; test rows assigned to the nearest centroid. Splits covered by this file: " + ", ".join(map(str, seeds_done)) + " (checkpointed after each); means over these splits. Mistral's overall solve rate: " + f"{Y[:, mi].mean():.3f}.", "",
              "partition | clusters | max Mistral solve rate in a cluster (>= 50 rows) | share of test rows in clusters with rate >= 0.5 / 0.7 / 0.9 | Mistral's realised solve rate on those | share of test rows the partition router sends to Mistral at λ = 10 / 50 / 100 / 500 / 1000 | router utility vs cell lookup at λ = 100 / 1000", "---|---|---|---|---|---|---"]
        for k, d in out["partitions"].items():
            md.append(f"{k} | {d['n_clusters']:.0f} | {d['max_cluster_mistral_rate_min50rows']:.2f} | " + " / ".join(f"{d['share_test_in_clusters_rate_ge'][str(t)]:.3f}" for t in THR) + " | " + " / ".join(f"{d['mistral_rate_on_those'][str(t)]:.2f}" for t in THR) + " | " + " / ".join(f"{d['share_sent_to_mistral'][f'{l:g}']:.3f}" for l in LAMS) + f" | {d['utility']['100']:.4f} vs {d['cell_lookup_utility']['100']:.4f} / {d['utility']['1000']:.4f} vs {d['cell_lookup_utility']['1000']:.4f}")
        (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
        print(f"[checkpoint] results written for seeds {seeds_done}", flush=True)

    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te)
        for K in KS:
            km = (KMeans(n_clusters=K, n_init=3, random_state=seed) if K <= 256 else MiniBatchKMeans(n_clusters=K, n_init=3, random_state=seed, batch_size=4096)).fit(u[tr])
            evaluate(km.labels_, km.predict(u[te]), tr, te, f"kmeans_K{K}"); print(f"seed {seed} K={K} done", flush=True)
        evaluate(cell[tr], cell[te], tr, te, "cells_family_topic"); evaluate(ns[tr].astype(int), ns[te].astype(int), tr, te, "true_difficulty_strata (not available at test time)")
        seeds_done.append(seed); write_results()


if __name__ == "__main__":
    main()
