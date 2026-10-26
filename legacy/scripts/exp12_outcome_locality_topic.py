"""exp12: exp11 with random / best-possible references matched on (family, topic) where a topic label exists (MMLU subjects; MT-Bench partial categories).

See experiments/exp12_outcome_locality_topic.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp12_outcome_locality_topic"
KS = (5, 10, 40, 100)
DRAWS = 5


def main() -> None:
    panel = load_panel(PANEL)
    spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep)
    u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float32), nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); n = panel.n; kmax = max(KS)
    cell_name = topic_str(panel, "record_prefix")          # RouterBench: "mmlu-anatomy", "mtbench-math", "hellaswag"; LLMRouterBench: released topics
    cell_names = sorted(set(cell_name)); cell = np.array([cell_names.index(c_) for c_ in cell_name])
    print(f"cells: {len(cell_names)} (families {F})", flush=True)
    diff = 1.0 - r.mean(1)
    dis = (r.max(1) - r.min(1)) > 1e-9
    def one_split(seed):
        rng = np.random.default_rng(seed)
        tr, te = split(n, fam, seed); tr = np.sort(tr); te = np.sort(te)
        fam_tr, fam_te = fam[tr], fam[te]; cell_tr, cell_te = cell[tr], cell[te]
        # kNN neighbours of test rows among training rows
        nb_knn = np.zeros((len(te), kmax), dtype=np.int64)
        for s0 in range(0, len(te), 2048):
            e0 = min(len(te), s0 + 2048); sims = u[te[s0:e0]] @ u[tr].T
            top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
            nb_knn[s0:e0] = tr[np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)]
        nb_best = np.zeros((len(te), kmax), dtype=np.int64); nb_rand = np.zeros((DRAWS, len(te), kmax), dtype=np.int64)
        avail = np.zeros(len(te), dtype=np.int64)      # training rows available in the query's reference cell (caps k for random/best)
        for g in range(len(cell_names)):
            trg = tr[cell_tr == g]; teg = np.flatnonzero(cell_te == g); Rtr = r[trg]
            if len(teg) == 0 or len(trg) == 0:
                continue
            for s0 in range(0, len(teg), 1024):
                e0 = min(len(teg), s0 + 1024)
                agree = 1.0 - np.abs(r[te[teg[s0:e0]]][:, None, :] - Rtr[None, :, :]).mean(2)
                kk = min(kmax, len(trg))
                top = np.argpartition(-agree, kk - 1, axis=1)[:, :kk]
                top = np.take_along_axis(top, np.argsort(-np.take_along_axis(agree, top, axis=1), axis=1), axis=1)
                nb_best[teg[s0:e0], :kk] = trg[top]
                avail[teg[s0:e0]] = kk
            for d in range(DRAWS):
                # independent draw, per test row, of min(kmax, |cell|) training rows of the same cell, without replacement
                kk = min(kmax, len(trg))
                keys = rng.random((len(teg), len(trg)))
                nb_rand[d, teg, :kk] = trg[np.argpartition(keys, kk - 1, axis=1)[:, :kk]] if kk < len(trg) else trg[np.argsort(keys, axis=1)]
        def agreement(nb, counts=None):
            a = 1.0 - np.abs(r[te][:, None, :] - r[nb]).mean(2)          # (n_te, k)
            if counts is None:
                return a.mean(1)
            m = np.arange(a.shape[1])[None, :] < counts[:, None]
            return (a * m).sum(1) / counts
        out_k = {}
        for k in KS:
            cnt = np.minimum(k, avail)
            a_knn = agreement(nb_knn[:, :k]); a_best = agreement(nb_best[:, :k], cnt)
            a_rand = np.mean([agreement(nb_rand[d, :, :k], cnt) for d in range(DRAWS)], axis=0)
            mk = np.arange(k)[None, :] < cnt[:, None]
            d_knn = diff[nb_knn[:, :k]].mean(1); d_rand = np.mean([(diff[nb_rand[d, :, :k]] * mk).sum(1) / cnt for d in range(DRAWS)], axis=0)
            out_k[k] = {"queries_with_fewer_than_k_available": int((avail < k).sum())}
            for subset, mask in (("all", np.ones(len(te), bool)), ("disagreement", dis[te])):
                out_k[k][subset] = {"knn": float(a_knn[mask].mean()), "random": float(a_rand[mask].mean()), "best": float(a_best[mask].mean()),
                                    "knn_minus_random": float((a_knn - a_rand)[mask].mean()),
                                    "rho_knn": float(spearmanr(diff[te][mask], d_knn[mask]).correlation), "rho_random": float(spearmanr(diff[te][mask], d_rand[mask]).correlation),
                                    "per_family": {names[g]: {"knn": float(a_knn[mask & (fam_te == g)].mean()), "random": float(a_rand[mask & (fam_te == g)].mean()), "best": float(a_best[mask & (fam_te == g)].mean())} for g in range(F) if (mask & (fam_te == g)).sum() > 0}}
        return out_k

    per_seed = {seed: one_split(seed) for seed in SEEDS}
    res = {**panel.provenance(), "experiment": EXP, "representation": spec.name, "ks": list(KS), "seeds": list(SEEDS), "family_order": names, "by_k": {}}
    md = [f"# {EXP}", "", f"Panel: {res['panel']} filtered, {n} rows; 5 stratified 70/30 splits (seeds 0–4), neighbours from training rows, scored on test rows; values = mean [min, max] over splits. {spec.name}, cosine.", ""]
    def agg(k, subset, key):
        v = [per_seed[sd][k][subset][key] for sd in SEEDS]; return float(np.mean(v)), float(min(v)), float(max(v))
    for k in KS:
        res["by_k"][str(k)] = {subset: {key: agg(k, subset, key) for key in ("knn", "random", "best", "knn_minus_random", "rho_knn", "rho_random")} for subset in ("all", "disagreement")}
        res["by_k"][str(k)]["per_family_all"] = {nm: {key: float(np.mean([per_seed[sd][k]["all"]["per_family"][nm][key] for sd in SEEDS])) for key in ("knn", "random", "best")} for nm in names}
        for subset in ("all", "disagreement"):
            a = res["by_k"][str(k)][subset]
            gap = (a["knn"][0] - a["random"][0]) / max(a["best"][0] - a["random"][0], 1e-9)
            md += [f"## k = {k}, {subset}: knn {a['knn'][0]:.4f} [{a['knn'][1]:.4f},{a['knn'][2]:.4f}] | random same-family {a['random'][0]:.4f} | best possible {a['best'][0]:.4f} | gap closed {gap:+.3f} | knn − random {a['knn_minus_random'][0]:+.4f} [{a['knn_minus_random'][1]:+.4f},{a['knn_minus_random'][2]:+.4f}] | difficulty ρ knn {a['rho_knn'][0]:.3f} vs random {a['rho_random'][0]:.3f}", ""]
        md += ["family (all, mean over splits) | knn | random | best", "---|---|---|---"]
        md += [f"{nm} | {v['knn']:.4f} | {v['random']:.4f} | {v['best']:.4f}" for nm, v in res["by_k"][str(k)]["per_family_all"].items()]
        md.append("")
    out_dir = RESULTS_ROOT / "outcome_locality" / EXP; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(res, indent=1)); (out_dir / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
