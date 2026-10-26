"""exp34: kNN with a format-augmented index (RouterBench, results/augmentation_mc).

Index variants, per split (5 stratified 70/30 splits, seeds 0-4):
  orig   the training rows' prompt embeddings (the usual kNN);
  aug    orig + the augmented copies of the TRAINING rows only (each copy carries its source row's
         reward and cost vector; copies of test rows are never indexed);
  dup    control: orig with every training row repeated as many times as it has augmented copies
         (same index size and same per-row weight as "aug", no format bridging).
Queries: the original test prompts (what a deployed router receives).  Test-time augmentation
("tta"): the test query's own copies are used as extra query views against the "aug" index; the
per-model mean utilities of the views' neighbourhoods are averaged, then argmax (no view of a test
row is ever in the index).  k in {5, 10, 40, 100}; the family lookup, best single model and the
oracle as references.  Both aggregations at every lambda; paired bootstraps on split 0 (aug vs
orig, tta vs orig, aug vs family lookup); per-family utilities; neighbourhood diagnostics (share of
augmented neighbours, distinct source rows, same-family share) for the MC families.
Writes results/augmentation_mc/exp34_augmented_knn/{result.json, result.md}."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, aiq, split, summarise  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

AUG = ANALYSIS_ROOT / "results" / "augmentation_mc" / "panel"
OUT = ANALYSIS_ROOT / "results" / "augmentation_mc" / "exp34_augmented_knn"
KS = (5, 10, 40, 100); REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight"); N_BOOT = 1000


def unit(a):
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def topk(q, index, k):
    nb = np.zeros((len(q), k), dtype=np.int64); sims_top = np.zeros((len(q), k), dtype=np.float32)
    for s in range(0, len(q), 1024):
        sims = q[s:s + 1024] @ index.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        nb[s:s + 1024] = np.take_along_axis(top, order, axis=1); sims_top[s:s + 1024] = np.take_along_axis(sims, nb[s:s + 1024], axis=1)
    return nb, sims_top


def boot(d, fte, F, rng):
    q = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    groups = [np.flatnonzero(fte == g) for g in range(F)]; groups = [g for g in groups if len(g)]
    f = [np.mean([d[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
    return {"per_query": {"point": float(d.mean()), "ci95": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))]},
            "per_family_equal_weight": {"point": float(np.mean([d[g].mean() for g in groups])), "ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]}}


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); M = panel.m
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    x = unit(load_rep(find_rep(PANEL, REP), n, panel.keep))
    xa = unit(np.load(AUG / "gte-qwen2-7b.npy", mmap_mode="r").astype(np.float32))
    meta = json.load(open(AUG / "meta.json")); src = np.array([m["source_row"] for m in meta]); tgt = np.array([m["target_format"] for m in meta])
    assert len(src) == len(xa)
    n_copies = np.bincount(src, minlength=n)
    mc = [names.index(f) for f in ("mmlu", "arc-challenge", "hellaswag", "winogrande") if f in names]
    per = {}; boots = {}; diag = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte = fam[te]; dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        in_tr = np.zeros(n, bool); in_tr[tr] = True
        aug_rows = np.flatnonzero(in_tr[src])                                 # copies of training rows only
        index = {"orig": (x[tr], tr, np.zeros(len(tr), bool)),
                 "aug": (np.vstack([x[tr], xa[aug_rows]]), np.concatenate([tr, src[aug_rows]]), np.concatenate([np.zeros(len(tr), bool), np.ones(len(aug_rows), bool)])),
                 "dup": (x[np.repeat(tr, 1 + n_copies[tr])], np.repeat(tr, 1 + n_copies[tr]), np.zeros(int((1 + n_copies[tr]).sum()), bool))}
        nbs = {}
        for name, (vecs, owner, is_aug) in index.items():
            nb, sims = topk(x[te], vecs, max(KS)); nbs[name] = (owner[nb], is_aug[nb], sims)
        # test-time augmentation views (copies of test rows) against the aug index
        vecs, owner, is_aug = index["aug"]
        te_pos = {int(t): j for j, t in enumerate(te)}
        views = [[] for _ in te]
        for a in np.flatnonzero(~in_tr[src]):
            views[te_pos[int(src[a])]].append(a)
        view_rows = np.array([a for v in views for a in v]); view_of = np.array([j for j, v in enumerate(views) for _ in v])
        nb_v, _ = topk(xa[view_rows], vecs, max(KS)) if len(view_rows) else (np.zeros((0, max(KS)), np.int64), None)
        own_v = owner[nb_v] if len(view_rows) else nb_v
        # diagnostics on the aug index (k = 100, MC families)
        own, isa, _ = nbs["aug"]
        d = {}
        for g in mc:
            m = fte == g
            d[names[g]] = {"share_augmented_neighbours_k100": float(isa[m].mean()), "distinct_source_rows_k100": float(np.mean([len(set(row)) for row in own[m]])),
                           "same_family_share_k100_aug": float((fam[own[m]] == g).mean()), "same_family_share_k100_orig": float((fam[nbs["orig"][0][m]] == g).mean())}
        diag[str(seed)] = d
        res = {}; pq = {}; curves = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            choices = {"family_lookup": np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)])[fte], "best_single": np.full(len(te), int(Utr.mean(0).argmax())), "oracle": U[te].argmax(1)}
            for name in index:
                own, _, _ = nbs[name]
                for k in KS:
                    choices[f"{name}_knn{k}"] = U[own[:, :k]].mean(1).argmax(1)
            for k in KS:
                mean_u = U[nbs["aug"][0][:, :k]].mean(1)                       # original view against the aug index
                cnt = np.ones(len(te))
                if len(view_rows):
                    mv = U[own_v[:, :k]].mean(1); np.add.at(mean_u, view_of, mv); np.add.at(cnt, view_of, 1.0)
                choices[f"tta_knn{k}"] = (mean_u / cnt[:, None]).argmax(1)
            for name, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); pq[(name, lam)] = s.pop("_per_query_utility"); res.setdefault(name, {})[str(lam)] = s
                curves.setdefault(name, []).append([s["all"]["per_query"]["cost"], s["all"]["per_query"]["reward"]])
        max_cost = max(np.array(v)[:, 0].max() for v in curves.values())
        per[seed] = {"results": res, "aiq": {name: aiq(np.array(v), max_cost) for name, v in curves.items()}}
        if seed == SEEDS[0]:
            rng = np.random.default_rng(0)
            for lam in REPORT:
                for k in (10, 100):
                    boots[f"aug_knn{k} - orig_knn{k} @ λ={lam:g}"] = boot(pq[(f"aug_knn{k}", lam)] - pq[(f"orig_knn{k}", lam)], fte, F, rng)
                    boots[f"tta_knn{k} - orig_knn{k} @ λ={lam:g}"] = boot(pq[(f"tta_knn{k}", lam)] - pq[(f"orig_knn{k}", lam)], fte, F, rng)
                    boots[f"dup_knn{k} - orig_knn{k} @ λ={lam:g}"] = boot(pq[(f"dup_knn{k}", lam)] - pq[(f"orig_knn{k}", lam)], fte, F, rng)
                    boots[f"aug_knn{k} - family_lookup @ λ={lam:g}"] = boot(pq[(f"aug_knn{k}", lam)] - pq[("family_lookup", lam)], fte, F, rng)
        print(f"seed {seed} done; aug index {len(index['aug'][1])} rows, views {len(view_rows)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    methods = list(per[SEEDS[0]]["results"])
    def agg(name, lam, subset, mode, key):
        v = [per[s]["results"][name][str(lam)][subset][mode][key] for s in SEEDS]; return float(np.mean(v)), float(min(v)), float(max(v))
    out = {**panel.provenance(), "experiment": "exp34_augmented_knn", "representation": REP, "augmentation": str(AUG), "n_augmented": int(len(src)), "ks": list(KS), "seeds": list(SEEDS), "methods": methods,
           "by_lambda": {str(lam): {name: {subset: {mode: {key: agg(name, lam, subset, mode, key) for key in ("utility", "reward", "cost")} for mode in AGGS} for subset in ("all", "disagreement")} for name in methods} for lam in LAMBDAS},
           "per_family": {str(lam): {name: {names[g]: float(np.mean([per[s]["results"][name][str(lam)]["per_family"][g]["utility"] for s in SEEDS])) for g in range(F)} for name in methods} for lam in REPORT},
           "aiq": {name: [per[s]["aiq"][name] for s in SEEDS] for name in methods}, "bootstrap_split0": boots, "diagnostics_aug_index": diag}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# exp34 — kNN with a format-augmented index ({len(src):,} augmented texts, results/augmentation_mc)", "",
          "Index: orig = training prompts; aug = orig + augmented copies of the training rows (each with its source's rewards/costs); dup = orig with every training row repeated 1 + #copies times (size-matched control, no format bridging). Queries = original test prompts; tta = the test query and its own copies as views against the aug index (views never indexed). 5 splits; mean [min, max]; CIs = paired bootstraps, split 0.", ""]
    for mode in AGGS:
        md += [f"## Utility, {mode} (all test rows)", "", "λ | family lookup | " + " | ".join(f"orig k{k} | aug k{k} | dup k{k} | tta k{k}" for k in (10, 100)) + " | best single | oracle", "---|---|" + "---|" * 8 + "---|---"]
        for lam in REPORT:
            cells = [f"{agg('family_lookup', lam, 'all', mode, 'utility')[0]:.4f}"]
            for k in (10, 100):
                cells += [f"{agg(f'{v}_knn{k}', lam, 'all', mode, 'utility')[0]:.4f}" for v in ("orig", "aug", "dup", "tta")]
            cells += [f"{agg('best_single', lam, 'all', mode, 'utility')[0]:.4f}", f"{agg('oracle', lam, 'all', mode, 'utility')[0]:.4f}"]
            md.append(f"{lam:g} | " + " | ".join(cells))
        md.append("")
    md += ["## AIQ (mean over splits)", "", "method | AIQ", "---|---"] + [f"{name} | {np.mean(out['aiq'][name]):.4f}" for name in methods] + [""]
    md += ["## Per-family utility (mean over splits), multiple-choice families", ""]
    for lam in REPORT:
        md += [f"### λ = {lam:g}", "", "method | " + " | ".join(names[g] for g in mc), "---|" + "---|" * len(mc)]
        for name in ["family_lookup", "orig_knn10", "aug_knn10", "dup_knn10", "tta_knn10", "orig_knn100", "aug_knn100", "dup_knn100", "tta_knn100"]:
            md.append(f"{name} | " + " | ".join(f"{out['per_family'][str(lam)][name][names[g]]:.4f}" for g in mc))
        md.append("")
    md += ["## Paired bootstraps, split 0 (per query / equal weight, 95% CI)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    md += ["", "## Neighbourhood diagnostics on the aug index (k = 100, mean over splits)", "", "family | share of augmented neighbours | distinct source rows among 100 | same-family share (aug index) | same-family share (orig index)", "---|---|---|---|---"]
    for g in mc:
        f = names[g]; md.append(f"{f} | " + " | ".join(f"{np.mean([diag[str(s)][f][key] for s in SEEDS]):.3f}" for key in ("share_augmented_neighbours_k100", "distinct_source_rows_k100", "same_family_share_k100_aug", "same_family_share_k100_orig")))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
