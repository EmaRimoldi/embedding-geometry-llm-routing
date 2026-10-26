"""exp17: within-family permutation test — does the topic label carry routing information beyond the family?

See experiments/exp17_topic_significance.yaml.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402

EXP = "exp17_topic_significance"
LAMS = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0)
N_PERM = 200; K = 100; DRAWS = 3; MIN_CELL = 50


def main() -> None:
    panel = load_panel(PANEL)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); n = panel.n
    topic_arr = topic_str(panel)
    _, topic = np.unique(topic_arr, return_inverse=True)
    has_topic = np.isin(panel.dataset_ids, list(TOPIC_FAMILIES))
    scopes = {TOPIC_FAMILIES[0]: panel.dataset_ids == TOPIC_FAMILIES[0], TOPIC_FAMILIES[1]: panel.dataset_ids == TOPIC_FAMILIES[1], "both": has_topic}

    def lookup_choice(cell_all, tr, te, U):
        """best model per cell on training rows, family fallback below MIN_CELL; returns choice for test rows."""
        Utr = U[tr]; ftab = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)]); ch = ftab[fam[te]].copy()
        ctr, cte = cell_all[tr], cell_all[te]
        for g in np.unique(cte):
            rows = ctr == g
            if rows.sum() >= MIN_CELL: ch[cte == g] = Utr[rows].mean(0).argmax()
        return ch

    def random_sets(cell_all, tr, te, rng):
        ctr, cte = cell_all[tr], cell_all[te]; sets = np.zeros((DRAWS, len(te), K), dtype=np.int64); cnt = np.zeros(len(te), dtype=np.int64)
        for g in np.unique(cte):
            trg = tr[ctr == g]; teg = np.flatnonzero(cte == g)
            if len(trg) == 0: continue
            kk = min(K, len(trg)); cnt[teg] = kk
            for d in range(DRAWS):
                keys = rng.random((len(teg), len(trg))); sets[d, teg, :kk] = trg[np.argpartition(keys, kk - 1, axis=1)[:, :kk]] if kk < len(trg) else trg[np.argsort(keys, axis=1)]
        return sets, cnt

    def rand_choice(sets, cnt, U):
        m = np.arange(K)[None, :] < cnt[:, None]; outs = []
        for d in range(sets.shape[0]):
            mu = (U[sets[d]] * m[:, :, None]).sum(1) / np.maximum(cnt, 1)[:, None]; outs.append(mu.argmax(1))
        return outs

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed); tr, te = split(n, fam, seed); tr = np.sort(tr); te = np.sort(te)
        # permuted topic labels (within family), fixed across λ for this seed
        perms = []
        for pi in range(N_PERM):
            pt = topic.copy()
            for g in range(F):
                rows = np.flatnonzero(fam == g)
                if has_topic[rows[0]]: pt[rows] = topic[rows][rng.permutation(len(rows))]
            perms.append(pt)
        fam_sets, fam_cnt = random_sets(fam, tr, te, rng); top_sets, top_cnt = random_sets(topic, tr, te, rng)
        perm_sets = [random_sets(pt, tr, te, rng) for pt in perms[:40]]          # neighbourhood statistic: 40 permutations (cost)
        for lam in LAMS:
            U = r - lam * c; Ute = U[te]; idx = np.arange(len(te))
            ch_fam = lookup_choice(fam, tr, te, U); ch_top = lookup_choice(topic, tr, te, U)
            u_fam = Ute[idx, ch_fam]; u_top = Ute[idx, ch_top]
            null_lookup = np.array([Ute[idx, lookup_choice(pt, tr, te, U)] for pt in perms])            # (N_PERM, n_te)
            rf = np.mean([Ute[idx, ch] for ch in rand_choice(fam_sets, fam_cnt, U)], axis=0)
            rt = np.mean([Ute[idx, ch] for ch in rand_choice(top_sets, top_cnt, U)], axis=0)
            null_rand = np.array([np.mean([Ute[idx, ch] for ch in rand_choice(s_, c_, U)], axis=0) for s_, c_ in perm_sets])
            for scope, smask in scopes.items():
                m = smask[te]
                key = (scope, lam); d = results.setdefault(key, {"lookup_obs": [], "lookup_null": [], "rand_obs": [], "rand_null": [], "lookup_diff_q": None, "rand_diff_q": None})
                d["lookup_obs"].append(float((u_top - u_fam)[m].mean())); d["lookup_null"].extend(((null_lookup[:, m] - u_fam[m]).mean(1)).tolist())
                d["rand_obs"].append(float((rt - rf)[m].mean())); d["rand_null"].extend(((null_rand[:, m] - rf[m]).mean(1)).tolist())
                if seed == SEEDS[0]:
                    d["lookup_diff_q"] = (u_top - u_fam)[m]; d["rand_diff_q"] = (rt - rf)[m]
        print(f"seed {seed} done", flush=True)

    rng = np.random.default_rng(0); out = {**panel.provenance(), "experiment": EXP, "lambdas": list(LAMS), "n_perm_lookup": N_PERM, "n_perm_neighbourhood": 40, "k": K, "min_cell": MIN_CELL, "table": {}}
    md = [f"# {EXP}", "", "Within-family permutation test of the topic label (MMLU subjects; HellaSwag source). Gain = utility with the topic cells minus utility with the family only, on test rows of the topic families, per query. Observed = mean over 5 splits; null = the same statistic with topic labels permuted within family (lookup: 200 perms × 5 splits; neighbourhood: 40 × 5); p = fraction of null draws ≥ observed; CI = paired bootstrap over test queries, split 0.", ""]
    for stat, label in (("lookup", "(1) topic lookup − family lookup"), ("rand", "(2) random-100 same-topic − random-100 same-family")):
        md += [f"## {label}", "", "scope | λ | observed gain | bootstrap 95% CI (split 0) | null mean | null 97.5th pct | p-value", "---|---|---|---|---|---|---"]
        for scope in scopes:
            for lam in LAMS:
                d = results[(scope, lam)]; obs = float(np.mean(d[f"{stat}_obs"])); null = np.asarray(d[f"{stat}_null"])
                q = d[f"{stat}_diff_q"]; bs = [q[rng.integers(0, len(q), len(q))].mean() for _ in range(1000)]
                p = float((null >= obs).mean())
                out["table"][f"{stat}|{scope}|{lam:g}"] = {"observed": obs, "ci95_split0": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "null_mean": float(null.mean()), "null_p975": float(np.percentile(null, 97.5)), "p_value": p, "n_null": int(len(null))}
                md.append(f"{scope} | {lam:g} | {obs:+.4f} | [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] | {null.mean():+.4f} | {np.percentile(null, 97.5):+.4f} | {p:.3f}")
        md.append("")
    d_ = RESULTS_ROOT / "outcome_locality" / EXP; d_.mkdir(parents=True, exist_ok=True)
    (d_ / "result.json").write_text(json.dumps(out, indent=1)); (d_ / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
