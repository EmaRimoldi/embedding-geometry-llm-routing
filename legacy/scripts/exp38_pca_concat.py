"""exp38: PCA the semantic embedding first, then concatenate difficulty and cost proxies (PI, 2026-09-16).

Semantic part S in {full GTE, PCA-32/64/128/256 of GTE (fitted on training rows), whitened PCA-64}, unit-normalised.
Blocks B (cross-fitted on training rows, out of sample on test rows, standardised, / sqrt(d)):
  diff (1)        ridge-predicted solve count from Qwen2.5-7B-Instruct layer 21 last token
  permodel (M)    ridge-predicted success of each pool model
  all             diff + permodel + prompt surprise scalars + vocabulary-projection prediction + token counts
  cost (M)        ridge-predicted log cost of each pool model on this query (the cost proxy)
  diff+cost, permodel+cost, all+cost
Space z = [S, w * B]; w in a wide grid chosen per (lambda, k) on an inner 80/20 split of the training rows.
Routers: kNN k in {10, 100} on every (S, B) and on every S alone; family lookup, (family, topic) cell lookup,
best single model, oracle.  5 splits, 15-lambda grid, both aggregations, AIQ, paired bootstraps on split 0
against full-GTE kNN and against the same-S kNN.  Also the explained-variance curve of PCA on GTE.
Writes results[_root]/difficulty_llm/exp38_pca_concat/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, lengths_features, topic_str  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, aiq, split, summarise  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp36_augmented_knn import boot, cell_lookup, crossfit, topk, unit  # noqa: E402

EXP = "exp38_pca_concat"; OUT = RESULTS_ROOT / "difficulty_llm" / EXP
KS = (10, 100); W = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0); REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight")
SEMS = ("full", "pca32", "pca64", "pca128", "pca256", "pca64w"); BLOCKS = ("diff", "permodel", "all", "cost", "diff+cost", "permodel+cost", "all+cost")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", default="qwen2.5-7b-instruct"); ap.add_argument("--layer", type=int, default=21); ap.add_argument("--pool", default="last"); a = ap.parse_args()
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); M = panel.m; onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0); solved = (r >= SUCCESS_THRESHOLD).astype(np.float32); y = solved.sum(1); logc = np.log(c + 1e-6).astype(np.float32)
    g = unit(load_rep(find_rep(PANEL, REP), n, panel.keep))
    HID = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / a.reader
    def load(p):
        arr = np.load(p, mmap_mode="r"); arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr; assert arr.shape[0] == n; return np.asarray(arr, dtype=np.float32)
    h = load(HID / f"L{a.layer}_{a.pool}.npy"); lens = load(HID / "logit_lens_last.npy")
    st = np.load(ANALYSIS_ROOT / "features" / PANEL / "prompt" / "qwen7b_hidden" / "prompt_logprob_stats.npz")["stats"]; surprise = (st[panel.keep] if panel.keep is not None and st.shape[0] == len(panel.keep) else st).astype(np.float32)
    surface, _ = lengths_features(panel); surface = surface.astype(np.float32); topics = np.array([str(t) for t in topic_str(panel)], dtype=object)
    per = {}; boots = {}; chosen_w = {}; evr = None
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte, ftr = fam[te], fam[tr]; dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        raw = {"diff": crossfit(h, y, onehot, tr, te, fam, seed), "permodel": crossfit(h, solved, onehot, tr, te, fam, seed), "lens": crossfit(lens, y, onehot, tr, te, fam, seed), "cost": crossfit(h, logc, onehot, tr, te, fam, seed),
               "surprise": (surprise[tr], surprise[te]), "surface": (surface[tr], surface[te])}
        def std(parts):
            btr = np.column_stack([np.asarray(raw[k][0]).reshape(len(tr), -1) for k in parts]); bte = np.column_stack([np.asarray(raw[k][1]).reshape(len(te), -1) for k in parts])
            s = StandardScaler().fit(btr); d = btr.shape[1]; return (s.transform(btr) / np.sqrt(d)).astype(np.float32), (s.transform(bte) / np.sqrt(d)).astype(np.float32)
        B = {"diff": std(["diff"]), "permodel": std(["permodel"]), "all": std(["diff", "permodel", "surprise", "lens", "surface"]), "cost": std(["cost"])}
        for k in ("diff", "permodel", "all"):
            B[k + "+cost"] = (np.column_stack([B[k][0], B["cost"][0]]) / np.sqrt(2), np.column_stack([B[k][1], B["cost"][1]]) / np.sqrt(2))
        S = {"full": (g[tr], g[te])}
        pca_full = PCA(512, random_state=0).fit(g[tr])
        if seed == SEEDS[0]:
            cum = np.cumsum(pca_full.explained_variance_ratio_); evr = {str(k): float(cum[k - 1]) for k in (8, 16, 32, 64, 128, 256, 512)}
        for k in (32, 64, 128, 256):
            S[f"pca{k}"] = (unit(pca_full.transform(g[tr])[:, :k]), unit(pca_full.transform(g[te])[:, :k]))
        pw = PCA(64, whiten=True, random_state=0).fit(g[tr]); S["pca64w"] = (unit(pw.transform(g[tr])), unit(pw.transform(g[te])))
        print(f"seed {seed}: blocks and semantics ready ({time.time() - t0:.0f}s); PCA cumulative variance {evr}", flush=True)
        itr, ite = split(len(tr), ftr, 100 + seed); itr, ite = np.sort(itr), np.sort(ite)
        spaces = {f"{s}|plain": (S[s][0], S[s][1]) for s in SEMS}
        for s in SEMS:
            for b in BLOCKS:
                for w in W: spaces[f"{s}|{b}@w{w:g}"] = (np.column_stack([S[s][0], w * B[b][0]]), np.column_stack([S[s][1], w * B[b][1]]))
        inner_nb = {name: topk(vtr[ite], vtr[itr], max(KS)) for name, (vtr, vte) in spaces.items() if "@w" in name}
        outer_nb = {name: topk(vte, vtr, max(KS)) for name, (vtr, vte) in spaces.items()}
        print(f"seed {seed}: neighbours ready ({time.time() - t0:.0f}s)", flush=True)
        res = {}; pq = {}; curves = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            choices = {"family_lookup": np.array([Utr[ftr == gi].mean(0).argmax() for gi in range(F)])[fte], "cell_lookup": cell_lookup(topics[tr], topics[te], ftr, fte, Utr, F), "best_single": np.full(len(te), int(Utr.mean(0).argmax())), "oracle": U[te].argmax(1)}
            for k in KS:
                for s in SEMS:
                    choices[f"{s}|plain_knn{k}"] = Utr[outer_nb[f"{s}|plain"][:, :k]].mean(1).argmax(1)
                    for b in BLOCKS:
                        iu = {w: float(U[tr[ite]][np.arange(len(ite)), U[tr[itr]][inner_nb[f"{s}|{b}@w{w:g}"][:, :k]].mean(1).argmax(1)].mean()) for w in W}; w = max(iu, key=iu.get)
                        chosen_w.setdefault(f"{s}|{b}_knn{k}", {}).setdefault(str(lam), []).append(w); choices[f"{s}|{b}_knn{k}"] = Utr[outer_nb[f"{s}|{b}@w{w:g}"][:, :k]].mean(1).argmax(1)
            for name, ch in choices.items():
                sm = summarise(ch, r[te], c[te], lam, fte, dis, F); pq[(name, lam)] = sm.pop("_per_query_utility"); res.setdefault(name, {})[str(lam)] = sm
                curves.setdefault(name, []).append([sm["all"]["per_query"]["cost"], sm["all"]["per_query"]["reward"]])
        max_cost = max(np.array(v)[:, 0].max() for v in curves.values()); per[seed] = {"results": res, "aiq": {name: aiq(np.array(v), max_cost) for name, v in curves.items()}}
        if seed == SEEDS[0]:
            rng = np.random.default_rng(0)
            for lam in REPORT:
                for k in KS:
                    for s in SEMS:
                        if s != "full": boots[f"{s}|plain_knn{k} - full|plain_knn{k} @ λ={lam:g}"] = boot(pq[(f"{s}|plain_knn{k}", lam)] - pq[(f"full|plain_knn{k}", lam)], fte, F, rng)
                        for b in BLOCKS:
                            boots[f"{s}|{b}_knn{k} - full|plain_knn{k} @ λ={lam:g}"] = boot(pq[(f"{s}|{b}_knn{k}", lam)] - pq[(f"full|plain_knn{k}", lam)], fte, F, rng)
                            if s != "full": boots[f"{s}|{b}_knn{k} - {s}|plain_knn{k} @ λ={lam:g}"] = boot(pq[(f"{s}|{b}_knn{k}", lam)] - pq[(f"{s}|plain_knn{k}", lam)], fte, F, rng)
                    boots[f"best of (S,B) knn100 - cell_lookup @ λ={lam:g}"] = None
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    boots = {k: v for k, v in boots.items() if v is not None}
    OUT.mkdir(parents=True, exist_ok=True); methods = list(per[SEEDS[0]]["results"])
    def agg(name, lam, mode):
        v = [per[s]["results"][name][str(lam)]["all"][mode]["utility"] for s in SEEDS]; return float(np.mean(v))
    out = {**panel.provenance(), "experiment": EXP, "reader": a.reader, "layer": a.layer, "pool": a.pool, "semantics": list(SEMS), "blocks": list(BLOCKS), "weights": list(W), "ks": list(KS), "pca_cumulative_explained_variance_split0": evr,
           "by_lambda": {str(lam): {name: {mode: {key: float(np.mean([per[s]["results"][name][str(lam)]["all"][mode][key] for s in SEEDS])) for key in ("utility", "reward", "cost")} for mode in AGGS} for name in methods} for lam in LAMBDAS},
           "per_family": {str(lam): {name: {names[gi]: float(np.mean([per[s]["results"][name][str(lam)]["per_family"][gi]["utility"] for s in SEEDS])) for gi in range(F)} for name in methods} for lam in REPORT},
           "aiq": {name: float(np.mean([per[s]["aiq"][name] for s in SEEDS])) for name in methods}, "bootstrap_split0": boots, "chosen_w": chosen_w, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# {EXP} — PCA of the semantic embedding, then concatenation of difficulty / cost proxies ({panel.name})", "",
          f"S ∈ {SEMS}; blocks ∈ {BLOCKS}; z = [S, w·B]; w ∈ {W} chosen per (λ, k) on an inner split; kNN k ∈ {KS}; 5 splits, mean; CIs = paired bootstraps on split 0. PCA cumulative explained variance of GTE (split 0 training rows): {evr}.", ""]
    for mode in AGGS:
        for k in KS:
            md += [f"## Utility, {mode}, kNN-{k}", "", "S \\ B | plain | " + " | ".join(BLOCKS), "---|---|" + "---|" * len(BLOCKS)]
            for lam in REPORT:
                md.append(f"**λ = {lam:g}** (family lookup {agg('family_lookup', lam, mode):.4f}, cell lookup {agg('cell_lookup', lam, mode):.4f}, oracle {agg('oracle', lam, mode):.4f}) | | " + " | ".join("" for _ in BLOCKS))
                for s in SEMS:
                    md.append(f"{s} | {agg(f'{s}|plain_knn{k}', lam, mode):.4f} | " + " | ".join(f"{agg(f'{s}|{b}_knn{k}', lam, mode):.4f}" for b in BLOCKS))
            md.append("")
    md += ["## AIQ (mean over splits)", "", "method | AIQ", "---|---"] + [f"{name} | {out['aiq'][name]:.4f}" for name in ["family_lookup", "cell_lookup"] + [f"{s}|plain_knn100" for s in SEMS] + [f"{s}|{b}_knn100" for s in SEMS for b in BLOCKS]] + [""]
    md += ["## Paired bootstraps, split 0 (per query / equal weight, 95% CI), kNN-100 only", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        if "knn100" in k_: md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    md += ["", "## Chosen weights (mode over splits), kNN-100", "", "space | " + " | ".join(f"λ={lam:g}" for lam in REPORT), "---|" + "---|" * len(REPORT)]
    for s in SEMS:
        for b in BLOCKS:
            key = f"{s}|{b}_knn100"; md.append(f"{key} | " + " | ".join(f"{max(set(chosen_w[key][str(lam)]), key=chosen_w[key][str(lam)].count):g}" for lam in REPORT))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
