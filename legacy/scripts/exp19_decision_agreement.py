"""exp19: do kNN and the (family, topic) lookup choose the same model?  See experiments/exp19_decision_agreement.yaml."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp19_decision_agreement"; OUT = RESULTS_ROOT / "outcome_locality" / EXP
KS = (5, 10, 40, 100); DRAWS = 3; MIN_CELL = 50; N_BOOT = 1000
REPORT_LAMS = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight")


def main() -> None:
    panel = load_panel(PANEL); spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep); u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); n = panel.n
    cell_str = topic_str(panel)
    _, cell = np.unique(cell_str, return_inverse=True)

    def lookup_choice(groups, tr, te, U, fallback=None):
        Utr = U[tr]; gtr, gte_ = groups[tr], groups[te]
        ch = fallback.copy() if fallback is not None else np.full(len(te), Utr.mean(0).argmax())
        for g in np.unique(gte_):
            rows = gtr == g
            if rows.sum() >= (MIN_CELL if fallback is not None else 1): ch[gte_ == g] = Utr[rows].mean(0).argmax()
        return ch

    def random_sets(groups, tr, te, rng, k):
        gtr, gte_ = groups[tr], groups[te]; sets = np.zeros((DRAWS, len(te), k), dtype=np.int64); cnt = np.zeros(len(te), dtype=np.int64)
        for g in np.unique(gte_):
            trg = tr[gtr == g]; teg = np.flatnonzero(gte_ == g); kk = min(k, len(trg)); cnt[teg] = kk
            for d in range(DRAWS):
                keys = rng.random((len(teg), len(trg)))
                sets[d, teg, :kk] = trg[np.argpartition(keys, kk - 1, axis=1)[:, :kk]] if kk < len(trg) else trg[np.argsort(keys, axis=1)]
        return sets, cnt

    def set_choice(sets, cnt, U):
        m = (np.arange(sets.shape[2])[None, :] < cnt[:, None])[:, :, None]
        return [((U[sets[d]] * m).sum(1) / np.maximum(cnt, 1)[:, None]).argmax(1) for d in range(sets.shape[0])]

    def agg(v, fte, mode):
        return float(v.mean()) if mode == "per_query" else float(np.mean([v[fte == g].mean() for g in range(F) if (fte == g).any()]))

    def boot(v, fte, mode, rng):
        if mode == "per_query":
            vals = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
        else:
            groups = [np.flatnonzero(fte == g) for g in range(F)]; groups = [g for g in groups if len(g)]
            vals = [np.mean([v[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    stats = {}   # (lam, k, stat, agg) -> list over seeds ; plus split-0 CIs and per-family
    def put(key, val): stats.setdefault(key, []).append(val)
    ci = {}; perfam = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed); tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte = fam[te]
        nb = tr[NearestNeighbors(n_neighbors=max(KS), metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
        rsets = {k: random_sets(cell, tr, te, rng, k) for k in KS}
        for lam in LAMBDAS:
            U = r - lam * c; Ute = U[te]; idx = np.arange(len(te))
            ch_fam = lookup_choice(fam, tr, te, U); ch_cell = lookup_choice(cell, tr, te, U, fallback=ch_fam)
            cell_vs_fam = (ch_cell == ch_fam).astype(float)
            for mode in AGGS: put((lam, "cell_vs_family_agree", mode), agg(cell_vs_fam, fte, mode))
            for k in KS:
                ch_knn = U[nb[:, :k]].mean(1).argmax(1)
                a_knn = (ch_knn == ch_cell).astype(float); a_knnf = (ch_knn == ch_fam).astype(float)
                a_rand = np.mean([(ch == ch_cell).astype(float) for ch in set_choice(*rsets[k], U)], axis=0)
                dis = ch_knn != ch_cell; du = np.where(dis, Ute[idx, ch_knn] - Ute[idx, ch_cell], 0.0)   # per-query utility difference (0 where they agree)
                for mode in AGGS:
                    put((lam, k, "knn_vs_cell_agree", mode), agg(a_knn, fte, mode)); put((lam, k, "rand_vs_cell_agree", mode), agg(a_rand, fte, mode))
                    put((lam, k, "knn_minus_rand_agree", mode), agg(a_knn - a_rand, fte, mode)); put((lam, k, "knn_vs_family_agree", mode), agg(a_knnf, fte, mode))
                    put((lam, k, "disagree_share", mode), agg(dis.astype(float), fte, mode)); put((lam, k, "utility_knn_minus_cell_all_rows", mode), agg(du, fte, mode))
                    if seed == SEEDS[0]:
                        ci[(lam, k, "knn_minus_rand_agree", mode)] = boot(a_knn - a_rand, fte, mode, np.random.default_rng(0))
                        ci[(lam, k, "utility_knn_minus_cell_all_rows", mode)] = boot(du, fte, mode, np.random.default_rng(0))
                if lam in REPORT_LAMS:
                    for g in range(F):
                        m = fte == g
                        for stat, v in (("knn_vs_cell_agree", a_knn), ("rand_vs_cell_agree", a_rand), ("knn_vs_family_agree", a_knnf), ("cell_vs_family_agree", cell_vs_fam), ("disagree_share", dis.astype(float)), ("utility_knn_minus_cell_all_rows", du)):
                            perfam.setdefault((lam, k, stat, names[g]), []).append(float(v[m].mean()))
        print(f"seed {seed} done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    js = {"|".join(map(str, k)): {"mean": float(np.mean(v)), "sd": float(np.std(v)), "ci95_split0": ci.get(k)} for k, v in stats.items()}
    jf = {"|".join(map(str, k)): float(np.mean(v)) for k, v in perfam.items()}
    (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": EXP, "ks": list(KS), "draws": DRAWS, "min_cell": MIN_CELL, "n_boot": N_BOOT, "stats": js, "per_family": jf}, indent=1))

    def f(key, pm=True):
        v = stats[key]; return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if pm else f"{np.mean(v):+.4f}"
    def fci(key): a, b = ci[key]; return f"[{a:+.4f}, {b:+.4f}]"
    md = [f"# {EXP}", "", "Agreement = share of test queries on which two routers choose the same model. Cell = (family, topic) lookup with family fallback (< 50 train rows). Random-k = k random training rows of the same cell (3 draws). Mean ± sd over 5 splits; CI = bootstrap on split 0 (query-level for per query; family-stratified for equal weight). Δutility = utility(kNN choice) − utility(cell choice), averaged over ALL test rows (zero where they agree), so it is the net effect of kNN's deviations from the lookup.", ""]
    for mode in AGGS:
        for k in KS:
            md += [f"## {mode}, k = {k}", "", "λ | kNN–cell agree | random-k–cell agree | kNN − random (CI) | kNN–family agree | cell–family agree | disagree share | Δutility kNN−cell (CI)", "---|---|---|---|---|---|---|---"]
            for lam in LAMBDAS:
                md.append(f"{lam:g} | {f((lam, k, 'knn_vs_cell_agree', mode))} | {f((lam, k, 'rand_vs_cell_agree', mode))} | {f((lam, k, 'knn_minus_rand_agree', mode), False)} {fci((lam, k, 'knn_minus_rand_agree', mode))} | "
                          f"{f((lam, k, 'knn_vs_family_agree', mode))} | {f((lam, 'cell_vs_family_agree', mode))} | {f((lam, k, 'disagree_share', mode))} | {f((lam, k, 'utility_knn_minus_cell_all_rows', mode), False)} {fci((lam, k, 'utility_knn_minus_cell_all_rows', mode))}")
            md.append("")
    md += ["## Per family, k = 100 (mean over splits)", ""]
    for stat, label in (("knn_vs_cell_agree", "kNN–cell agreement"), ("rand_vs_cell_agree", "random-100–cell agreement"), ("cell_vs_family_agree", "cell–family agreement"), ("utility_knn_minus_cell_all_rows", "Δutility kNN−cell over all rows")):
        md += [f"### {label}", "", "λ | " + " | ".join(names), "---|" + "---|" * F]
        for lam in REPORT_LAMS:
            md.append(f"{lam:g} | " + " | ".join((f"{np.mean(perfam[(lam, 100, stat, g)]):+.4f}" if stat.startswith("utility") else f"{np.mean(perfam[(lam, 100, stat, g)]):.3f}") for g in names))
        md.append("")
    (OUT / "result.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
