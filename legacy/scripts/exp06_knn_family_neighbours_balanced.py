"""exp06: exp05 on a family-balanced subsample (400 per family, MT-Bench excluded), k in {10, 100, 399}. Outcome-free.

See experiments/exp06_knn_family_neighbours_balanced.yaml.
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

EXP = "exp06_knn_family_neighbours_balanced"
KS = (10, 100, 399)
PER_FAMILY = 400
EXCLUDE = ("mtbench",)   # RouterBench; on other panels: every family with fewer than PER_FAMILY rows (small_families)
DRAWS = (0, 1, 2, 3, 4)


def same_family_fractions(u: np.ndarray, fam: np.ndarray, ks, n_fam: int):
    """Per query: fraction of its k nearest neighbours (cosine, self excluded) in the same family; plus foreign counts."""
    n, kmax = len(u), max(ks)
    same = {k: np.zeros(n, dtype=np.float32) for k in ks}
    foreign = {k: np.zeros((n_fam, n_fam), dtype=np.int64) for k in ks}
    sims = u @ u.T
    sims[np.arange(n), np.arange(n)] = -np.inf
    top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)
    nf = fam[top]
    for k in ks:
        m = nf[:, :k] == fam[:, None]
        same[k] = m.mean(1)
        for qi in range(n_fam):
            rows = np.flatnonzero(fam == qi)
            foreign[k][qi] += np.bincount(nf[rows, :k][~m[rows]], minlength=n_fam)
    return same, foreign


def main() -> None:
    panel = load_panel(PANEL)
    spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep)
    u_full = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    fam_all = panel.dataset_index(); names_all = panel.datasets
    excluded = EXCLUDE if PANEL.startswith("routerbench") else small_families(panel, PER_FAMILY)
    keep_fam = [j for j, nm in enumerate(names_all) if nm not in excluded]
    names = [names_all[j] for j in keep_fam]; F = len(names); remap = {j: i for i, j in enumerate(keep_fam)}

    per_draw_same = {k: [] for k in KS}; per_draw_fam = {k: [] for k in KS}; foreign_tot = {k: np.zeros((F, F), dtype=np.int64) for k in KS}
    for d in DRAWS:
        rng = np.random.default_rng(d)
        rows = np.concatenate([rng.choice(np.flatnonzero(fam_all == j), size=PER_FAMILY, replace=False) for j in keep_fam]); rows.sort()
        fam = np.array([remap[j] for j in fam_all[rows]])
        same, foreign = same_family_fractions(u_full[rows], fam, KS, F)
        for k in KS:
            per_draw_same[k].append(same[k]); per_draw_fam[k].append(np.array([same[k][fam == j].mean() for j in range(F)])); foreign_tot[k] += foreign[k]
        print(f"draw {d} done", flush=True)

    res = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "ks": list(KS), "family_order": names,
           "per_family_rows": PER_FAMILY, "excluded": list(excluded), "draws": list(DRAWS), "by_k": {}}
    md = [f"# {EXP}", "", f"Panel: {res['panel']} filtered; balanced subsample {PER_FAMILY} per family, {list(excluded)} excluded, {PER_FAMILY * F} rows per draw, {len(DRAWS)} draws. {spec.name}, cosine; neighbours among the other rows of the draw.", ""]
    for k in KS:
        allv = np.concatenate(per_draw_same[k]); pf = np.vstack(per_draw_fam[k]); draw_means = [float(v.mean()) for v in per_draw_same[k]]
        q = np.quantile(allv, [0.05, 0.25, 0.5, 0.75, 0.95])
        per_fam = {}
        for j, nm in enumerate(names):
            per_fam[nm] = {"mean_same_family": float(pf[:, j].mean()), "range_over_draws": [float(pf[:, j].min()), float(pf[:, j].max())],
                           "chance": 1.0 / F, "ceiling": float(min(k, PER_FAMILY - 1) / k),
                           "top_foreign": {names[i]: int(v) for i, v in sorted(enumerate(foreign_tot[k][j]), key=lambda t: -t[1])[:2] if v > 0}}
        res["by_k"][str(k)] = {"mean_same_family": float(allv.mean()), "draw_means": draw_means, "quantiles_5_25_50_75_95": q.tolist(), "chance": 1.0 / F, "per_family": per_fam}
        md += [f"## k = {k}: mean same-family fraction = {allv.mean():.4f} (draws {min(draw_means):.4f}–{max(draw_means):.4f}; chance {1/F:.3f}); per-query quantiles 5/25/50/75/95 = {', '.join(f'{v:.3f}' for v in q)}", "",
               "family | mean same-family | range over draws | chance | ceiling | where the foreign neighbours go (all draws)", "---|---|---|---|---|---"]
        md += [f"{nm} | {v['mean_same_family']:.3f} | {v['range_over_draws'][0]:.3f}–{v['range_over_draws'][1]:.3f} | {v['chance']:.3f} | {v['ceiling']:.3f} | {v['top_foreign']}" for nm, v in per_fam.items()]
        md.append("")
    out = RESULTS_ROOT / "geometry_tests" / EXP; out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=1)); (out / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
