"""exp43: enrich the routing space until the signal the semantic embedding loses comes back.

Two changes over exp36/exp42, both aimed at the same goal:
  (a) better proxy predictions: instead of one reader layer, every frozen representation on the
      cluster is used as a predictor (Qwen2.5-7B-Instruct L14/L21/L28 last, Qwen2.5-Coder-1.5B
      L22/L25 last, Qwen3-Embedding-0.6B L14/L19 last, the GTE embedding itself, the vocabulary
      projection, the prompt surprise scalars, token counts); each big source is reduced by PCA
      (fitted on training rows, no outcome used) and the pieces are concatenated into one design
      matrix for a multi-output ridge, cross-fitted on the training rows;
  (b) the predictions are used as the SPACE, not only as extra coordinates: kNN is run on the
      predicted-proxy vector alone, on the predicted per-model utility vector at the current lambda,
      and on the semantic embedding joined with them at an inner-selected weight.
Targets = exp42's proxies (everything computable from the outcome table) plus the per-model success
and per-model log cost vectors.  Routers compared, at the 15-lambda grid, 5 splits, both
aggregations: family lookup, (family, topic) lookup, the inner-selected proxy-bin lookup, plain GTE
kNN, proxy-space kNN, predicted-utility-space kNN, joined-space kNN, the direct argmax router on
predicted per-model utility, best single model and the oracle.  Every choice that could be tuned
(the key, the weight, k) is selected on an inner 80/20 split of the TRAINING rows.
Writes results[_root]/difficulty_llm/exp43_ensemble_space/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, topic_str  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split, summarise  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp36_augmented_knn import boot, crossfit, topk, unit  # noqa: E402
from exp42_proxy_keys import build_targets, cell_choice, codes, qbins  # noqa: E402

EXP = "exp43_ensemble_space"
NPCA = 128; QS = (3, 5); KS = (10, 40, 100); W = (0.25, 0.5, 1.0, 2.0, 4.0)
REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight")
SOURCES = [("qwen2.5-7b-instruct", "L14_last"), ("qwen2.5-7b-instruct", "L21_last"), ("qwen2.5-7b-instruct", "L28_last"),
           ("qwen2.5-coder-1.5b-instruct", "L22_last"), ("qwen2.5-coder-1.5b-instruct", "L25_last"),
           ("qwen3-embedding-0.6b", "L14_last"), ("qwen3-embedding-0.6b", "L19_last")]


def main() -> None:
    global NPCA, OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--baseline-layer", default="L21_last")
    ap.add_argument("--raw", default="", help="sources kept at full dimension instead of PCA, e.g. 'qwen2.5-7b-instruct:L21_last'")
    ap.add_argument("--npca", type=int, default=128); ap.add_argument("--tag", default="")
    a = ap.parse_args(); raw_keep = {x for x in a.raw.replace(",", ";").split(";") if x}
    NPCA = a.npca
    OUT = RESULTS_ROOT / "difficulty_llm" / (EXP + (("_" + a.tag) if a.tag else ""))
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names)
    onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    Y, pnames, M, _ = build_targets(r, c, panel); P = len(pnames)
    topics_all = np.array([str(t) for t in topic_str(panel)], dtype=object); _, topic_code = np.unique(topics_all, return_inverse=True)
    HROOT = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden"

    def load_arr(p):
        arr = np.load(p, mmap_mode="r"); arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr
        return np.asarray(arr, dtype=np.float32)

    big, used = {}, []
    for mdl, lay in SOURCES:
        f = HROOT / mdl / f"{lay}.npy"
        if f.exists() and (HROOT / mdl / "meta.json").exists():
            big[f"{mdl}:{lay}"] = load_arr(f); used.append(f"{mdl}:{lay}")
    gte = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); big["gte"] = gte; used.append("gte")
    small = {}
    ll = HROOT / "qwen2.5-7b-instruct" / "logit_lens_last.npy"
    if ll.exists(): small["logit_lens"] = load_arr(ll)
    sp = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "qwen7b_hidden" / "prompt_logprob_stats.npz"
    if sp.exists():
        st = np.load(sp)["stats"]; small["surprise"] = (st[panel.keep] if panel.keep is not None and st.shape[0] == len(panel.keep) else st).astype(np.float32)
    baseline = big.get(f"qwen2.5-7b-instruct:{a.baseline_layer}")
    print(f"panel {PANEL}: n={n} F={F} M={M} proxies={P}; sources {used} + {list(small)}", flush=True)

    per_split = {}; readout = {}; pq_store = {}; sel_log = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
        dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        parts = []
        for k, x in big.items():
            sc = StandardScaler().fit(x[tr]); xs = sc.transform(x).astype(np.float32)
            if k in raw_keep or NPCA <= 0:                     # kept at full dimension (npca 0 = no reduction anywhere)
                parts.append(xs); continue
            p = PCA(min(NPCA, x.shape[1]), svd_solver="randomized", random_state=0).fit(xs[tr])
            parts.append(p.transform(xs).astype(np.float32))
        for k, x in small.items():
            parts.append(StandardScaler().fit(x[tr]).transform(x).astype(np.float32))
        X = np.column_stack(parts); print(f"seed {seed}: design matrix {X.shape} ({time.time() - t0:.0f}s)", flush=True)
        Ptr, Pte = crossfit(X, Y, onehot, tr, te, fam, seed); Ptr = np.asarray(Ptr); Pte = np.asarray(Pte)
        Btr = Bte = None
        if baseline is not None:
            Btr, Bte = crossfit(baseline, Y, onehot, tr, te, fam, seed); Btr = np.asarray(Btr); Bte = np.asarray(Bte)
        if seed == SEEDS[0]:
            for j, nm in enumerate(pnames):
                floor = np.array([Y[tr][ftr == g, j].mean() for g in range(F)])[fte]; sse0 = ((Y[te][:, j] - floor) ** 2).sum()
                readout[nm] = {"ensemble": float(1 - ((Y[te][:, j] - Pte[:, j]) ** 2).sum() / max(sse0, 1e-12))}
                if Bte is not None: readout[nm]["single_reader"] = float(1 - ((Y[te][:, j] - Bte[:, j]) ** 2).sum() / max(sse0, 1e-12))
            print("readout:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in readout.items()}), flush=True)
        # ---- spaces
        gtr, gte_u = unit(gte[tr]), unit(gte[te])
        ps = StandardScaler().fit(Ptr); prox_tr, prox_te = unit(ps.transform(Ptr)), unit(ps.transform(Pte))
        rhat = (Ptr[:, P:P + M], Pte[:, P:P + M]); chat = (np.exp(Ptr[:, P + M:]), np.exp(Pte[:, P + M:]))
        spaces = {"gte": (gtr, gte_u), "proxy": (prox_tr, prox_te)}
        for w in W: spaces[f"gte+proxy@w{w:g}"] = (np.column_stack([gtr, w * prox_tr]), np.column_stack([gte_u, w * prox_te]))
        nb = {nm: topk(v[1], v[0], max(KS)) for nm, v in spaces.items()}
        inner_tr, inner_te = split(len(tr), ftr, 100 + seed); inner_tr, inner_te = np.sort(inner_tr), np.sort(inner_te)
        nb_in = {nm: topk(v[0][inner_te], v[0][inner_tr], max(KS)) for nm, v in spaces.items()}
        # ---- deployability: the family and topic parts of a key are themselves predicted at inference
        sc_g = StandardScaler().fit(gte[tr]); gs_tr, gs_te = sc_g.transform(gte[tr]), sc_g.transform(gte[te])
        clf_f = LogisticRegression(C=1.0, max_iter=300, n_jobs=-1).fit(gs_tr, ftr)
        pf_tr, pf_te = clf_f.predict(gs_tr), clf_f.predict(gs_te)
        clf_t = LogisticRegression(C=1.0, max_iter=300, n_jobs=-1).fit(gs_tr, topic_code[tr])
        pt_tr, pt_te = clf_t.predict(gs_tr), clf_t.predict(gs_te)
        acc = {"family_accuracy_test": float((pf_te == fte).mean()), "topic_cell_accuracy_test": float((pt_te == topic_code[te]).mean())}
        sel_log.setdefault("_label_recovery", []).append(acc)
        print(f"seed {seed}: label recovery from the embedding {acc} ({time.time() - t0:.0f}s)", flush=True)
        # ---- lookup keys from the ensemble predictions
        keyset = {"family": codes([(ftr, fte)]), "family+topic": codes([(ftr, fte), (topic_code[tr], topic_code[te])]),
                  "DEPLOY::family": codes([(pf_tr, pf_te)]), "DEPLOY::family+topic": codes([(pf_tr, pf_te), (pt_tr, pt_te)])}
        for j, nm in enumerate(pnames):
            for Q in QS:
                bt, be = qbins(Ptr[:, j], Pte[:, j], ftr, fte, F, Q)
                keyset[f"family+{nm}/Q{Q}"] = codes([(ftr, fte), (bt, be)])
                keyset[f"family+topic+{nm}/Q{Q}"] = codes([(ftr, fte), (topic_code[tr], topic_code[te]), (bt, be)])
                # fully deployable: predicted family and predicted topic, predicted proxy bin
                bt_d, be_d = qbins(Ptr[:, j], Pte[:, j], pf_tr, pf_te, F, Q)
                keyset[f"DEPLOY::family+{nm}/Q{Q}"] = codes([(pf_tr, pf_te), (bt_d, be_d)])
                # unattainable ceiling: the TRUE value of the proxy as the cell key
                ct, ce = qbins(Y[tr][:, j], Y[te][:, j], ftr, fte, F, Q)
                keyset[f"CEILING::family+{nm}TRUE/Q{Q}"] = codes([(ftr, fte), (ct, ce)])
        print(f"seed {seed}: {len(keyset)} keys, {len(spaces)} spaces ({time.time() - t0:.0f}s)", flush=True)
        res = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            uhat_tr = rhat[0] - lam * chat[0]; uhat_te = rhat[1] - lam * chat[1]
            us = StandardScaler().fit(uhat_tr); utr, ute = unit(us.transform(uhat_tr)), unit(us.transform(uhat_te))
            nb_u = topk(ute, utr, max(KS)); nb_u_in = topk(utr[inner_te], utr[inner_tr], max(KS))
            choices = {"best_single": np.full(len(te), int(Utr.mean(0).argmax())), "oracle": U[te].argmax(1),
                       "direct_argmax_router": uhat_te.argmax(1)}
            # inner selection: best lookup key, best (space, k)
            inner = {}
            for nm, (ctr, cte, nc) in keyset.items():
                ch = cell_choice(ctr[inner_tr], ctr[inner_te], ftr[inner_tr], ftr[inner_te], Utr[inner_tr], F, nc)
                inner[f"lookup::{nm}"] = float(U[tr[inner_te]][np.arange(len(inner_te)), ch].mean())
            for nm in spaces:
                for k in KS:
                    ch = Utr[inner_tr][nb_in[nm][:, :k]].mean(1).argmax(1)
                    inner[f"knn::{nm}::{k}"] = float(U[tr[inner_te]][np.arange(len(inner_te)), ch].mean())
            for k in KS:
                ch = Utr[inner_tr][nb_u_in[:, :k]].mean(1).argmax(1)
                inner[f"knn::uhat::{k}"] = float(U[tr[inner_te]][np.arange(len(inner_te)), ch].mean())
            sel_lookup = max((x for x in inner if x.startswith("lookup::") and "CEILING" not in x), key=inner.get)
            sel_deploy = max((x for x in inner if x.startswith("lookup::DEPLOY")), key=inner.get)
            sel_knn = max((x for x in inner if x.startswith("knn::")), key=inner.get)
            sel_any = max((x for x in inner if "CEILING" not in x), key=inner.get)
            sel_log.setdefault(str(lam), []).append({"lookup": sel_lookup, "knn": sel_knn, "any": sel_any, "deploy": sel_deploy})
            # evaluate everything on the test rows
            for nm, (ctr, cte, nc) in keyset.items():
                choices[f"lookup::{nm}"] = cell_choice(ctr, cte, ftr, fte, Utr, F, nc)
            for nm in spaces:
                for k in KS: choices[f"knn::{nm}::{k}"] = Utr[nb[nm][:, :k]].mean(1).argmax(1)
            for k in KS: choices[f"knn::uhat::{k}"] = Utr[nb_u[:, :k]].mean(1).argmax(1)
            for nm, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); pq = s.pop("_per_query_utility")
                res.setdefault(nm, {})[str(lam)] = {"all": s["all"], "per_family": s["per_family"]}
                if seed == SEEDS[0] and lam in REPORT: pq_store[(nm, lam)] = pq
            res.setdefault("_selected", {})[str(lam)] = {"lookup": sel_lookup, "knn": sel_knn, "any": sel_any, "deploy": sel_deploy}
            if lam in REPORT: print(f"  seed {seed} lam={lam:g}: lookup={sel_lookup} knn={sel_knn} ({time.time() - t0:.0f}s)", flush=True)
        per_split[seed] = res
    methods = [k for k in per_split[SEEDS[0]] if not k.startswith("_")]
    def agg(nm, lam, mode):
        return float(np.mean([per_split[s][nm][str(lam)]["all"][mode]["utility"] for s in SEEDS]))
    table = {str(lam): {nm: {mode: agg(nm, lam, mode) for mode in AGGS} for nm in methods} for lam in LAMBDAS}
    boots = {}; rng = np.random.default_rng(0); _, te0 = split(n, fam, SEEDS[0]); te0 = np.sort(te0); fte0 = fam[te0]
    for lam in REPORT:
        picks = [p for p in sel_log[str(lam)] if isinstance(p, dict) and "lookup" in p]
        sel_l = max({p["lookup"] for p in picks}, key=lambda x: sum(p["lookup"] == x for p in picks))
        sel_k = max({p["knn"] for p in picks}, key=lambda x: sum(p["knn"] == x for p in picks))
        for nm in dict.fromkeys([sel_l, sel_k, "knn::proxy::100", "knn::gte::100", "direct_argmax_router"]):
            if (nm, lam) in pq_store:
                boots[f"{nm} - family @ λ={lam:g}"] = boot(pq_store[(nm, lam)] - pq_store[("lookup::family", lam)], fte0, F, rng)
                boots[f"{nm} - gte_knn100 @ λ={lam:g}"] = boot(pq_store[(nm, lam)] - pq_store[("knn::gte::100", lam)], fte0, F, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": EXP, "tag": a.tag, "raw_sources": sorted(raw_keep), "sources": used + list(small), "n_pca": NPCA, "proxies": pnames,
           "readout_split0": readout, "selected_per_lambda": sel_log, "label_recovery": sel_log.get("_label_recovery", []),
           "by_lambda": table, "bootstrap_split0": boots, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# {EXP} — ensemble of frozen readers, proxies as the routing space ({panel.name})", "",
          f"Predictors: {', '.join(used + list(small))}, each big source reduced by PCA-{NPCA} on the training rows, concatenated, one multi-output ridge cross-fitted on the training rows. Every tunable choice is selected on an inner 80/20 split of the training rows. 5 splits, 15 λ.", "",
          "## Proxy read-out (R² beyond the family mean, split 0): ensemble vs the single reader layer", "", "proxy | ensemble | single reader | Δ", "---|---|---|---"]
    for k, v in sorted(readout.items(), key=lambda kv: -kv[1]["ensemble"]):
        s_ = v.get("single_reader"); md.append(f"{k} | {v['ensemble']:+.3f} | " + (f"{s_:+.3f} | {v['ensemble'] - s_:+.3f}" if s_ is not None else "— | —"))
    for mode in AGGS:
        md += ["", f"## Utility, {mode}", "", "router | " + " | ".join(f"λ={l:g}" for l in REPORT), "---|" + "---|" * len(REPORT)]
        base = ["lookup::family", "lookup::family+topic", "knn::gte::100", "knn::proxy::10", "knn::proxy::100", "knn::uhat::10", "knn::uhat::100", "direct_argmax_router", "best_single", "oracle"]
        for nm in base:
            if nm in methods: md.append(f"{nm} | " + " | ".join(f"{table[str(l)][nm][mode]:.4f}" for l in REPORT))
        md.append("")
        md += ["inner-selected per λ (DEPLOY = family and topic parts of the key also predicted at inference):", "",
               "λ | selected lookup | utility | selected deployable lookup | utility | selected kNN | utility", "---|---|---|---|---|---|---"]
        for lam in REPORT:
            picks = [p for p in sel_log[str(lam)] if isinstance(p, dict) and "lookup" in p]
            sel_l = max({p["lookup"] for p in picks}, key=lambda x: sum(p["lookup"] == x for p in picks))
            sel_k = max({p["knn"] for p in picks}, key=lambda x: sum(p["knn"] == x for p in picks))
            sel_d = max({p["deploy"] for p in picks}, key=lambda x: sum(p["deploy"] == x for p in picks))
            md.append(f"{lam:g} | {sel_l} | {table[str(lam)][sel_l][mode]:.4f} | {sel_d} | {table[str(lam)][sel_d][mode]:.4f} | {sel_k} | {table[str(lam)][sel_k][mode]:.4f}")
        md += ["", "Predicted vs true cell label, and the unattainable ceiling of each proxy (per query, λ = 0 / 100):", "",
               "proxy | predicted-bin cell | TRUE-bin cell (ceiling) | gap", "---|---|---|---"]
        for nm in pnames:
            for Q in (5,):
                a_ = f"lookup::family+{nm}/Q{Q}"; b_ = f"lookup::CEILING::family+{nm}TRUE/Q{Q}"
                if a_ in methods and b_ in methods:
                    md.append(f"{nm} (Q{Q}) | {table['0.0'][a_][mode]:.4f} / {table['100.0'][a_][mode]:.4f} | {table['0.0'][b_][mode]:.4f} / {table['100.0'][b_][mode]:.4f} | {table['0.0'][b_][mode] - table['0.0'][a_][mode]:+.4f} / {table['100.0'][b_][mode] - table['100.0'][a_][mode]:+.4f}")
    md += ["", "## Top 10 routers per λ (exploratory, per query)", ""]
    for lam in REPORT:
        top = sorted(methods, key=lambda nm: -table[str(lam)][nm]["per_query"])[:10]
        md += [f"### λ = {lam:g}", "", "router | per query | equal weight", "---|---|---"] + [f"{nm} | {table[str(lam)][nm]['per_query']:.4f} | {table[str(lam)][nm]['per_family_equal_weight']:.4f}" for nm in top] + [""]
    md += ["## Paired bootstraps, split 0 (per query / equal weight)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
