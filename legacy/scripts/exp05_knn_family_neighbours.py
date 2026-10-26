"""exp05: same-family fraction among the k nearest neighbours (cosine, GTE), k in {10, 100, 1000}. Outcome-free.

See experiments/exp05_knn_family_neighbours.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp05_knn_family_neighbours"
KS = (10, 100, 1000)


def main() -> None:
    panel = load_panel(PANEL)
    spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep)
    u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names)
    n = panel.n; kmax = max(KS)
    same = {k: np.zeros(n, dtype=np.float32) for k in KS}
    foreign = {k: np.zeros((F, F), dtype=np.int64) for k in KS}   # [family of query, family of neighbour] counts, non-family only
    chunk = 2048
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sims = u[s:e] @ u.T
        sims[np.arange(e - s), np.arange(s, e)] = -np.inf          # exclude self
        top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
        top_s = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_s, axis=1)
        top = np.take_along_axis(top, order, axis=1)                  # sorted by similarity, descending
        nf = fam[top]
        qf = fam[s:e][:, None]
        for k in KS:
            m = nf[:, :k] == qf
            same[k][s:e] = m.mean(1)
            bad = ~m
            for qi in range(F):
                rows = np.flatnonzero(fam[s:e] == qi)
                if len(rows):
                    foreign[k][qi] += np.bincount(nf[rows, :k][bad[rows]], minlength=F)
        print(f"{e}/{n}", flush=True)
    share = np.bincount(fam, minlength=F) / n
    res = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "ks": list(KS), "family_order": names, "by_k": {}}
    md = [f"# {EXP}", "", f"Panel: {res['panel']} filtered, {n} rows. {spec.name}, cosine. Neighbours among all other rows.", ""]
    for k in KS:
        per_fam = {}
        for j, name in enumerate(names):
            m = fam == j; nj = int(m.sum())
            per_fam[name] = {"n": nj, "mean_same_family": float(same[k][m].mean()), "median": float(np.median(same[k][m])),
                             "chance": float(share[j]), "ceiling": float(min(k, nj - 1) / k),
                             "top_foreign": {names[i]: int(v) for i, v in sorted(enumerate(foreign[k][j]), key=lambda t: -t[1])[:2] if v > 0}}
        q = np.quantile(same[k], [0.05, 0.25, 0.5, 0.75, 0.95])
        res["by_k"][str(k)] = {"mean_same_family": float(same[k].mean()), "quantiles_5_25_50_75_95": q.tolist(),
                               "chance_query_weighted": float((share[fam]).mean()), "per_family": per_fam}
        md += [f"## k = {k}: mean same-family fraction = {same[k].mean():.4f} (chance {float((share[fam]).mean()):.3f}); per-query quantiles 5/25/50/75/95 = {', '.join(f'{v:.3f}' for v in q)}", "",
               "family | n | mean same-family | median | chance | ceiling | where the foreign neighbours go", "---|---|---|---|---|---|---"]
        md += [f"{nm} | {v['n']} | {v['mean_same_family']:.3f} | {v['median']:.3f} | {v['chance']:.3f} | {v['ceiling']:.3f} | {v['top_foreign']}" for nm, v in per_fam.items()]
        md.append("")
    out = RESULTS_ROOT / "geometry_tests" / EXP; out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=1)); (out / "result.md").write_text("\n".join(md) + "\n")
    np.savez_compressed(out / "per_query_same_family.npz", **{f"k{k}": same[k] for k in KS}, family=fam)
    print("\n".join(md))


if __name__ == "__main__":
    main()
