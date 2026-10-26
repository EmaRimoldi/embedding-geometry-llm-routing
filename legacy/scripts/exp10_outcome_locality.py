"""exp10: outcome agreement with kNN neighbours vs random same-family vs best-possible same-family neighbours.

See experiments/exp10_outcome_locality.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp10_outcome_locality"
KS = (5, 10, 40, 100)
DRAWS = 5


def main() -> None:
    panel = load_panel(PANEL)
    spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep)
    u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float32), nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); n = panel.n; kmax = max(KS)
    diff = 1.0 - r.mean(1)
    dis = (r.max(1) - r.min(1)) > 1e-9
    rng = np.random.default_rng(0)

    # kNN neighbours over all rows (self excluded)
    nb_knn = np.zeros((n, kmax), dtype=np.int64)
    for s in range(0, n, 2048):
        e = min(n, s + 2048); sims = u[s:e] @ u.T; sims[np.arange(e - s), np.arange(s, e)] = -np.inf
        top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
        nb_knn[s:e] = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)
    # best-possible neighbours within family by outcome agreement; random same-family draws
    nb_best = np.zeros((n, kmax), dtype=np.int64); nb_rand = np.zeros((DRAWS, n, kmax), dtype=np.int64)
    for g in range(F):
        rows = np.flatnonzero(fam == g); R = r[rows]
        for s in range(0, len(rows), 1024):
            e = min(len(rows), s + 1024)
            agree = 1.0 - np.abs(R[s:e, None, :] - R[None, :, :]).mean(2)          # (chunk, n_g)
            agree[np.arange(e - s), np.arange(s, e)] = -np.inf
            kk = min(kmax, len(rows) - 1)
            top = np.argpartition(-agree, kk - 1, axis=1)[:, :kk]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(agree, top, axis=1), axis=1), axis=1)
            nb_best[rows[s:e], :kk] = rows[top]
            if kk < kmax:
                nb_best[rows[s:e], kk:] = rows[top[:, :1]]                             # tiny family: pad with its best
        for d in range(DRAWS):
            for i, row in enumerate(rows):
                pool = np.delete(rows, i) if len(rows) - 1 >= kmax else np.delete(rows, i)
                nb_rand[d, row] = rng.choice(pool, size=kmax, replace=len(pool) < kmax)

    def agreement(nb):  # (n, k) -> per-query mean agreement over the k neighbours
        return (1.0 - np.abs(r[:, None, :] - r[nb]).mean(2)).mean(1)

    res = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "ks": list(KS), "family_order": names, "by_k": {}}
    md = [f"# {EXP}", "", f"Panel: {res['panel']} filtered, {n} rows; {spec.name}, cosine. Agreement = mean over 11 models of 1 − |r_i − r_j|.", ""]
    for k in KS:
        a_knn = agreement(nb_knn[:, :k]); a_best = agreement(nb_best[:, :k])
        a_rand = np.mean([agreement(nb_rand[d, :, :k]) for d in range(DRAWS)], axis=0)
        d_knn = diff[nb_knn[:, :k]].mean(1); d_rand = np.mean([diff[nb_rand[d, :, :k]].mean(1) for d in range(DRAWS)], axis=0)
        out = {}
        for subset, mask in (("all", np.ones(n, bool)), ("disagreement", dis)):
            gap = (a_knn[mask].mean() - a_rand[mask].mean()) / max(a_best[mask].mean() - a_rand[mask].mean(), 1e-9)
            dd = (a_knn - a_rand)[mask]; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(1000)]
            out[subset] = {"knn": float(a_knn[mask].mean()), "random_same_family": float(a_rand[mask].mean()), "best_possible": float(a_best[mask].mean()),
                           "gap_closed": float(gap), "knn_minus_random": float(dd.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                           "difficulty_spearman_knn": float(spearmanr(diff[mask], d_knn[mask]).correlation),
                           "difficulty_spearman_random": float(spearmanr(diff[mask], d_rand[mask]).correlation),
                           "per_family": {names[g]: {"knn": float(a_knn[mask & (fam == g)].mean()), "random": float(a_rand[mask & (fam == g)].mean()),
                                                     "best": float(a_best[mask & (fam == g)].mean())} for g in range(F) if (mask & (fam == g)).sum() > 0}}
        res["by_k"][str(k)] = out
        for subset in ("all", "disagreement"):
            o = out[subset]
            md += [f"## k = {k}, {subset}: knn {o['knn']:.4f} | random same-family {o['random_same_family']:.4f} | best possible {o['best_possible']:.4f} | gap closed {o['gap_closed']:+.3f} | knn − random {o['knn_minus_random']:+.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] | difficulty ρ knn {o['difficulty_spearman_knn']:.3f} vs random {o['difficulty_spearman_random']:.3f}", "",
                   "family | knn | random | best", "---|---|---|---"]
            md += [f"{nm} | {v['knn']:.4f} | {v['random']:.4f} | {v['best']:.4f}" for nm, v in o["per_family"].items()]
            md.append("")
    out_dir = RESULTS_ROOT / "outcome_locality" / EXP; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(res, indent=1)); (out_dir / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
