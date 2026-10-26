"""exp42: every non-semantic proxy computable from the outcome table, predicted from the frozen
reader and used as a LOOKUP KEY (the shape that worked for difficulty in exp36/exp41).

Pipeline, per split (5 stratified 70/30 splits):
  1. Targets Y, all computed from the panel's own reward/cost table, no text:
       n_solved, mean_reward, max_reward, reward_std, disagreement (max-min), n_acceptable
       (models within 0.05 of the best reward), log_mean_cost, log_cheapest_solver_cost,
       cheap_solves (does the pool's cheapest model solve it), oracle_utility at lam 10/100/1000,
       log token count (surface control), plus the per-model success and per-model log cost vectors.
  2. One multi-output ridge on [family one-hot, standardised reader state] predicts all of them;
     training-row values are cross-fitted (5 folds) so index and query are both out of sample.
  3. Each predicted scalar is cut into Q per-family quantile bins (Q in {2,3,5}); the per-model
     predictions give one extra categorical key: the argmax of predicted utility at the current lam.
  4. Keys = family; (family, topic); (family, bin(p)); (family, bin(p1), bin(p2)); (family, topic,
     bin(p)); (family, predicted-argmax). Cells under MIN_CELL training rows fall back to the family.
  5. Every key is a lookup: best model per cell on training rows, applied to test rows, at the
     15-lam grid, both aggregations.
  6. HONEST HEADLINE: for each lam the key is chosen on an inner 80/20 split of the TRAINING rows,
     then reported on the test rows; the full grid is kept as exploratory. Paired bootstrap on
     split 0 for the selected key and the exploratory best, and a within-family permutation null
     (bin labels shuffled inside each family) for the selected key.
Writes results[_root]/difficulty_llm/exp42_proxy_keys/."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, lengths_features, topic_str  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split, summarise  # noqa: E402
from exp36_augmented_knn import boot, crossfit  # noqa: E402

EXP = "exp42_proxy_keys"; OUT = RESULTS_ROOT / "difficulty_llm" / EXP
QS = (2, 3, 5); MIN_CELL = 50; REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight"); N_PERM = 200; EPS = 0.05


def build_targets(r, c, panel):
    """Every scalar computable from the outcome table (plus the surface length control)."""
    S = (r >= SUCCESS_THRESHOLD).astype(np.float64); M = r.shape[1]
    cheapest = int(np.median(c, axis=0).argmin())
    solver_cost = np.where(S > 0, c, np.nan)
    with np.errstate(all="ignore"):
        cheap_solver = np.nanmin(solver_cost, axis=1)
    cheap_solver = np.where(np.isfinite(cheap_solver), cheap_solver, c.max(axis=1))
    L, _ = lengths_features(panel)
    cols = {
        "n_solved": S.sum(1), "mean_reward": r.mean(1), "max_reward": r.max(1), "reward_std": r.std(1),
        "disagreement": r.max(1) - r.min(1), "n_acceptable": (r >= r.max(1, keepdims=True) - EPS).sum(1),
        "log_mean_cost": np.log(c.mean(1) + 1e-9), "log_cheapest_solver_cost": np.log(cheap_solver + 1e-9),
        "cheap_solves": S[:, cheapest], "oracle_u_10": (r - 10 * c).max(1), "oracle_u_100": (r - 100 * c).max(1),
        "oracle_u_1000": (r - 1000 * c).max(1), "log_tokens": L[:, 0],
    }
    names = list(cols); Y = np.column_stack([cols[k] for k in names])
    Y = np.column_stack([Y, S, np.log(c + 1e-9)])          # per-model success, per-model log cost
    return Y.astype(np.float64), names, M, cheapest


def cell_choice(cell_tr, cell_te, fam_tr, fam_te, Utr, F, n_cells):
    """Lookup: best model per cell on training rows; cells under MIN_CELL fall back to the family."""
    Sm = sparse.csr_matrix((np.ones(len(cell_tr)), (np.arange(len(cell_tr)), cell_tr)), shape=(len(cell_tr), n_cells))
    sums = Sm.T @ Utr; counts = np.asarray(Sm.sum(0)).ravel()
    fam_sums = np.zeros((F, Utr.shape[1])); fam_counts = np.zeros(F)
    np.add.at(fam_sums, fam_tr, Utr); np.add.at(fam_counts, fam_tr, 1.0)
    fam_best = (fam_sums / np.maximum(fam_counts, 1)[:, None]).argmax(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / np.maximum(counts, 1)[:, None]
    best = means.argmax(1); ok = counts >= MIN_CELL
    choice = np.where(ok[cell_te], best[cell_te], fam_best[fam_te])
    return choice.astype(int)


def qbins(v_tr, v_te, fam_tr, fam_te, F, Q):
    btr = np.zeros(len(v_tr), int); bte = np.zeros(len(v_te), int)
    for g in range(F):
        mt = fam_tr == g
        if not mt.any():
            continue
        q = np.quantile(v_tr[mt], np.linspace(0, 1, Q + 1)[1:-1])
        btr[mt] = np.searchsorted(q, v_tr[mt]); bte[fam_te == g] = np.searchsorted(q, v_te[fam_te == g])
    return btr, bte


def codes(arrays):
    """Combine integer label arrays into a dense 0..K-1 code (train and test coded together)."""
    key = np.zeros(len(arrays[0][0]) + len(arrays[0][1]), dtype=np.int64)
    for a_tr, a_te in arrays:
        a = np.concatenate([a_tr, a_te]); key = key * (int(a.max()) + 1) + a
    uniq, inv = np.unique(key, return_inverse=True)
    n_tr = len(arrays[0][0]); return inv[:n_tr], inv[n_tr:], len(uniq)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", default="qwen2.5-7b-instruct"); ap.add_argument("--layer", type=int, default=21); ap.add_argument("--pool", default="last"); ap.add_argument("--max-pairs", type=int, default=10000); a = ap.parse_args()
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    Y, pnames, M, cheapest = build_targets(r, c, panel); P = len(pnames)
    topics_all = np.array([str(t) for t in topic_str(panel)], dtype=object); _, topic_code = np.unique(topics_all, return_inverse=True)
    HID = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / a.reader
    h = np.load(HID / f"L{a.layer}_{a.pool}.npy", mmap_mode="r"); h = h[panel.keep] if panel.keep is not None and h.shape[0] == len(panel.keep) else h; h = np.asarray(h, dtype=np.float32)
    print(f"panel {PANEL}: n={n} F={F} M={M} proxies={P} reader={a.reader} L{a.layer}_{a.pool}", flush=True)
    per_split = {}; readout = {}; pq_store = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
        dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        sc = StandardScaler().fit(h[tr])
        Ptr, Pte = crossfit(np.asarray(sc.transform(h), dtype=np.float32), Y, onehot, tr, te, fam, seed)
        Ptr = np.asarray(Ptr); Pte = np.asarray(Pte)
        if seed == SEEDS[0]:
            for j, nm in enumerate(pnames):
                floor = np.array([Y[tr][ftr == g, j].mean() for g in range(F)])[fte]
                sse = ((Y[te][:, j] - Pte[:, j]) ** 2).sum(); sse0 = ((Y[te][:, j] - floor) ** 2).sum()
                readout[nm] = float(1 - sse / max(sse0, 1e-12))
        print(f"seed {seed}: predictions ready ({time.time() - t0:.0f}s)" + (f"; readout R2beyondfam {readout}" if seed == SEEDS[0] else ""), flush=True)
        # --- build all candidate keys
        keyset = {}
        keyset["family"] = (ftr * 0 + ftr, fte, F) if False else codes([(ftr, fte)])
        keyset["family+topic"] = codes([(ftr, fte), (topic_code[tr], topic_code[te])])
        bins = {}
        for j, nm in enumerate(pnames):
            for Q in QS:
                bt, be = qbins(Ptr[:, j], Pte[:, j], ftr, fte, F, Q); bins[(nm, Q)] = (bt, be)
                keyset[f"family+{nm}/Q{Q}"] = codes([(ftr, fte), (bt, be)])
                keyset[f"family+topic+{nm}/Q{Q}"] = codes([(ftr, fte), (topic_code[tr], topic_code[te]), (bt, be)])
        pairs = list(itertools.combinations(pnames, 2))[: a.max_pairs]
        for p1, p2 in pairs:
            for Q in (2, 3):
                keyset[f"family+{p1}/Q{Q}+{p2}/Q{Q}"] = codes([(ftr, fte), bins[(p1, Q)], bins[(p2, Q)]])
        # predicted-argmax key is lambda-dependent; built inside the lambda loop
        rhat = Ptr[:, P:P + M], Pte[:, P:P + M]; chat = np.exp(Ptr[:, P + M:]), np.exp(Pte[:, P + M:])
        inner_tr, inner_te = split(len(tr), ftr, 100 + seed); inner_tr, inner_te = np.sort(inner_tr), np.sort(inner_te)
        print(f"seed {seed}: {len(keyset)} keys built ({time.time() - t0:.0f}s)", flush=True)
        res = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            ks = dict(keyset)
            am_tr = (rhat[0] - lam * chat[0]).argmax(1); am_te = (rhat[1] - lam * chat[1]).argmax(1)
            ks["family+predicted_argmax"] = codes([(ftr, fte), (am_tr, am_te)])
            ks["predicted_argmax_only"] = codes([(am_tr, am_te)])
            # inner selection on the training rows
            inner_scores = {}
            for name, (ctr, cte, nc) in ks.items():
                ch = cell_choice(ctr[inner_tr], ctr[inner_te], ftr[inner_tr], ftr[inner_te], Utr[inner_tr], F, nc)
                inner_scores[name] = float(U[tr[inner_te]][np.arange(len(inner_te)), ch].mean())
            selected = max(inner_scores, key=inner_scores.get)
            for name, (ctr, cte, nc) in ks.items():
                ch = cell_choice(ctr, cte, ftr, fte, Utr, F, nc)
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); pq = s.pop("_per_query_utility")
                res.setdefault(name, {})[str(lam)] = {"all": s["all"], "per_family": s["per_family"], "n_cells": int(nc)}
                if seed == SEEDS[0] and lam in REPORT: pq_store[(name, lam)] = pq
            res.setdefault("_selected", {})[str(lam)] = {"key": selected, "inner_score": inner_scores[selected]}
            if lam in REPORT: print(f"  seed {seed} lam={lam:g}: selected '{selected}' ({time.time() - t0:.0f}s)", flush=True)
        per_split[seed] = res
        if seed == SEEDS[0]:
            perm = {}
            for lam in REPORT:
                sel = res["_selected"][str(lam)]["key"]
                if "/Q" not in sel: continue
                U = r - lam * c; Utr = U[tr]; base = res[sel][str(lam)]["all"]["per_query"]["utility"]; fam_only = res["family"][str(lam)]["all"]["per_query"]["utility"]
                nulls = []
                nm_q = [t for t in sel.split("+") if "/Q" in t]
                for d in range(N_PERM):
                    rng = np.random.default_rng(1000 + d); parts = [(ftr, fte)]
                    if "topic" in sel: parts.append((topic_code[tr], topic_code[te]))
                    for t_ in nm_q:
                        nm, Q = t_.split("/Q"); bt, be = bins[(nm, int(Q))]; bt2 = bt.copy(); be2 = be.copy()
                        for g in range(F):
                            mt = ftr == g; me = fte == g
                            bt2[mt] = bt[mt][rng.permutation(mt.sum())]; be2[me] = be[me][rng.permutation(me.sum())]
                        parts.append((bt2, be2))
                    ctr, cte, nc = codes(parts)
                    ch = cell_choice(ctr, cte, ftr, fte, Utr, F, nc)
                    nulls.append(float((r[te][np.arange(len(te)), ch] - lam * c[te][np.arange(len(te)), ch]).mean()))
                nulls = np.array(nulls)
                perm[str(lam)] = {"key": sel, "observed_gain_over_family": base - fam_only, "null_mean_gain": float(nulls.mean() - fam_only),
                                  "null_p975_gain": float(np.percentile(nulls, 97.5) - fam_only), "p_value": float((nulls >= base).mean()), "n_perm": N_PERM}
                print(f"  perm lam={lam:g} key={sel}: observed {base - fam_only:+.4f}, null {nulls.mean() - fam_only:+.4f}, p={perm[str(lam)]['p_value']:.3f}", flush=True)
            per_split["_perm_split0"] = perm
    # ---- aggregate
    methods = [k for k in per_split[SEEDS[0]] if not k.startswith("_")]
    def agg(name, lam, mode):
        return float(np.mean([per_split[s][name][str(lam)]["all"][mode]["utility"] for s in SEEDS]))
    table = {str(lam): {name: {mode: agg(name, lam, mode) for mode in AGGS} for name in methods} for lam in LAMBDAS}
    selected = {str(lam): [per_split[s]["_selected"][str(lam)]["key"] for s in SEEDS] for lam in LAMBDAS}
    boots = {}
    rng = np.random.default_rng(0); tr0, te0 = split(n, fam, SEEDS[0]); te0 = np.sort(te0); fte0 = fam[te0]
    for lam in REPORT:
        sel = max(set(selected[str(lam)]), key=selected[str(lam)].count)
        top = sorted((nm for nm in methods if nm != "family"), key=lambda nm: -table[str(lam)][nm]["per_query"])[:3]
        for nm in dict.fromkeys([sel] + top):
            if (nm, lam) in pq_store and ("family", lam) in pq_store:
                boots[f"{nm} - family @ λ={lam:g}"] = boot(pq_store[(nm, lam)] - pq_store[("family", lam)], fte0, F, rng)
        if ("family+topic", lam) in pq_store and (sel, lam) in pq_store:
            boots[f"{sel} - family+topic @ λ={lam:g}"] = boot(pq_store[(sel, lam)] - pq_store[("family+topic", lam)], fte0, F, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": EXP, "reader": a.reader, "layer": a.layer, "pool": a.pool, "proxies": pnames, "n_keys": len(methods),
           "readout_r2_beyond_family_split0": readout, "selected_key_per_lambda": selected, "by_lambda": table,
           "bootstrap_split0": boots, "permutation_split0": per_split.get("_perm_split0", {}), "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# {EXP} — non-semantic proxies from the outcome table as lookup keys ({panel.name})", "",
          f"Every proxy is predicted from the frozen {a.reader} (L{a.layer}, {a.pool}) by a multi-output ridge on family + reader state, cross-fitted on training rows; predictions are cut into per-family quantile bins and used as lookup cells (cells under {MIN_CELL} training rows fall back to the family). {len(methods)} keys, 5 splits, 15 λ. The headline is the key chosen per λ on an inner 80/20 split of the TRAINING rows; the rest of the grid is exploratory and subject to multiple comparisons.", "",
          "## How well each proxy is read from the reader (R² beyond the family mean, split 0)", "", "proxy | R²", "---|---"]
    md += [f"{k} | {v:+.3f}" for k, v in sorted(readout.items(), key=lambda kv: -kv[1])]
    for mode in AGGS:
        md += ["", f"## Utility of the inner-selected key, {mode}", "", "λ | selected key (mode over splits) | selected | family lookup | (family, topic) lookup | best key in the grid (exploratory)", "---|---|---|---|---|---"]
        for lam in REPORT:
            sel = max(set(selected[str(lam)]), key=selected[str(lam)].count)
            bestk = max((nm for nm in methods), key=lambda nm: table[str(lam)][nm][mode])
            md.append(f"{lam:g} | {sel} | {table[str(lam)][sel][mode]:.4f} | {table[str(lam)]['family'][mode]:.4f} | {table[str(lam)]['family+topic'][mode]:.4f} | {bestk} {table[str(lam)][bestk][mode]:.4f}")
    md += ["", "## Top 12 keys per λ (exploratory, per query, mean over splits)", ""]
    for lam in REPORT:
        top = sorted(methods, key=lambda nm: -table[str(lam)][nm]["per_query"])[:12]
        md += [f"### λ = {lam:g} (family {table[str(lam)]['family']['per_query']:.4f}, family+topic {table[str(lam)]['family+topic']['per_query']:.4f})", "", "key | per query | equal weight", "---|---|---"]
        md += [f"{nm} | {table[str(lam)][nm]['per_query']:.4f} | {table[str(lam)][nm]['per_family_equal_weight']:.4f}" for nm in top] + [""]
    md += ["## Paired bootstraps, split 0 (per query / equal weight)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    if per_split.get("_perm_split0"):
        md += ["", "## Within-family permutation null of the selected key (split 0, bin labels shuffled inside each family)", "", "λ | key | observed gain over family | null mean | null 97.5th | p", "---|---|---|---|---|---"]
        for lam, v in per_split["_perm_split0"].items():
            md.append(f"{float(lam):g} | {v['key']} | {v['observed_gain_over_family']:+.4f} | {v['null_mean_gain']:+.4f} | {v['null_p975_gain']:+.4f} | {v['p_value']:.3f}")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
