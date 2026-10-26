"""exp20: recover the (family, topic) cell from the embedding with the exp08 family classifier, and route on the predicted cell.
See experiments/exp20_topic_recovery.yaml.   Usage: --seed s  (one job per seed) ; --aggregate (after all seeds)."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp20_topic_recovery"; OUT = RESULTS_ROOT / "outcome_locality" / EXP
MIN_CELL = 50; K = 100; N_BOOT = 1000; AGGS = ("per_query", "per_family_equal_weight")
METHODS = ("lookup:true_cell", "lookup:predicted_cell", "lookup:predicted_family", "knn100", "oracle")


def load():
    panel = load_panel(PANEL); x = load_rep(find_rep(PANEL, REP), panel.n, panel.keep).astype(np.float32)
    bow = load_rep(find_rep(PANEL, "computed/hash-bow@prompt"), panel.n, panel.keep).astype(np.float32)
    cell_str = topic_str(panel)
    cell_names, cell = np.unique(cell_str, return_inverse=True)
    fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]
    cell_fam = np.array([fam[cell == c_][0] for c_ in range(len(cell_names))])          # family of each cell
    return panel, x, bow, fam, names, cell, [str(c_) for c_ in cell_names], cell_fam


def run_seed(seed: int) -> None:
    t0 = time.time(); panel, x, bow, fam, names, cell, cell_names, cell_fam = load(); F = len(names); n = panel.n
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte, cte = fam[te], cell[te]
    # classifiers
    sc = StandardScaler().fit(x[tr]); clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x[tr]), cell[tr])
    pred = clf.predict(sc.transform(x[te])); print(f"gte classifier fit {time.time() - t0:.0f}s, acc {np.mean(pred == cte):.4f}", flush=True)
    scb = StandardScaler().fit(bow[tr]); clfb = LogisticRegression(C=1.0, max_iter=3000).fit(scb.transform(bow[tr]), cell[tr])
    predb = clfb.predict(scb.transform(bow[te])); print(f"bow classifier fit {time.time() - t0:.0f}s, acc {np.mean(predb == cte):.4f}", flush=True)
    scf = StandardScaler().fit(x[tr]); fam_pred = LogisticRegression(C=1.0, max_iter=3000).fit(scf.transform(x[tr]), fam[tr]).predict(scf.transform(x[te]))
    # majority cell within the true family (reference for the topic part)
    maj = {g: np.bincount(cell[tr][fam[tr] == g]).argmax() for g in range(F)}; pred_maj = np.array([maj[g] for g in fte])
    def cls_metrics(p):
        m = {"cell_accuracy": float(np.mean(p == cte)), "implied_family_accuracy": float(np.mean(cell_fam[p] == fte)),
             "cell_accuracy_per_family": {names[g]: float(np.mean(p[fte == g] == cte[fte == g])) for g in range(F)},
             "per_cell_recall": {cell_names[c_]: float(np.mean(p[cte == c_] == c_)) for c_ in np.unique(cte)}}
        mm = fte == names.index(TOPIC_FAMILIES[0]); conf = collections.Counter((cell_names[a], cell_names[b]) for a, b in zip(cte[mm], p[mm]) if a != b)
        m["top_mmlu_confusions"] = [[a, b, k] for (a, b), k in conf.most_common(10)]; return m
    cls = {"gte": cls_metrics(pred), "bow": cls_metrics(predb), "majority_cell_in_true_family": cls_metrics(pred_maj),
           "family_classifier_accuracy": float(np.mean(fam_pred == fte))}
    # routing
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    nb = tr[NearestNeighbors(n_neighbors=K, metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
    def lookup(groups_tr, groups_te, U, fallback=None):
        Utr = U[tr]; ch = fallback.copy() if fallback is not None else np.full(len(te), Utr.mean(0).argmax())
        for g in np.unique(groups_te):
            rows = groups_tr == g
            if rows.sum() >= (MIN_CELL if fallback is not None else 1): ch[groups_te == g] = Utr[rows].mean(0).argmax()
        return ch
    idx = np.arange(len(te)); util = np.zeros((len(LAMBDAS), len(METHODS), len(te)))
    for li, lam in enumerate(LAMBDAS):
        U = r - lam * c; Ute = U[te]
        ch_f_true = lookup(fam[tr], fte, U); ch_f_pred = lookup(fam[tr], fam_pred, U)
        chs = {"lookup:true_cell": lookup(cell[tr], cte, U, fallback=ch_f_true), "lookup:predicted_cell": lookup(cell[tr], pred, U, fallback=lookup(fam[tr], cell_fam[pred], U)),
               "lookup:predicted_family": ch_f_pred, "knn100": U[nb].mean(1).argmax(1)}
        for mi, m in enumerate(METHODS): util[li, mi] = Ute.max(1) if m == "oracle" else Ute[idx, chs[m]]
    def agg(v, mode): return float(v.mean()) if mode == "per_query" else float(np.mean([v[fte == g].mean() for g in range(F)]))
    routing = {f"{lam:g}|{m}|{mode}": agg(util[li, mi], mode) for li, lam in enumerate(LAMBDAS) for mi, m in enumerate(METHODS) for mode in AGGS}
    routing.update({f"{lam:g}|{m}|family:{names[g]}": float(util[li, mi][fte == g].mean()) for li, lam in enumerate(LAMBDAS) for mi, m in enumerate(METHODS) for g in range(F)})
    (OUT / "per_seed").mkdir(parents=True, exist_ok=True)
    (OUT / "per_seed" / f"seed{seed}.json").write_text(json.dumps({"seed": seed, "classifier": cls, "routing": routing, "n_test": int(len(te)), "seconds": time.time() - t0}, indent=1))
    if seed == SEEDS[0]: np.savez_compressed(OUT / "seed0_utilities.npz", util=util, fam_te=fte, methods=np.array(METHODS), lambdas=np.array(LAMBDAS))
    print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)


def aggregate() -> None:
    panel, x, bow, fam, names, cell, cell_names, cell_fam = load(); F = len(names)
    per = {s: json.loads((OUT / "per_seed" / f"seed{s}.json").read_text()) for s in SEEDS if (OUT / "per_seed" / f"seed{s}.json").exists()}
    seeds = sorted(per); z = np.load(OUT / "seed0_utilities.npz"); util, fte = z["util"], z["fam_te"]; rng = np.random.default_rng(0)
    def boot(v, mode):
        if mode == "per_query": vals = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
        else:
            groups = [np.flatnonzero(fte == g) for g in range(F)]; vals = [np.mean([v[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]) for _ in range(N_BOOT)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    mi = {m: i for i, m in enumerate(METHODS)}
    ci = {f"{lam:g}|{a}-{b}|{mode}": boot(util[li, mi[a]] - util[li, mi[b]], mode) for li, lam in enumerate(LAMBDAS) for a, b in (("lookup:predicted_cell", "lookup:true_cell"), ("lookup:predicted_cell", "knn100"), ("lookup:predicted_cell", "lookup:predicted_family")) for mode in AGGS}
    def ms(key): v = [per[s]["routing"][key] for s in seeds]; return float(np.mean(v)), float(np.std(v))
    def cm(path):
        v = [per[s]["classifier"] for s in seeds]
        for p_ in path: v = [d[p_] for d in v]
        return float(np.mean(v)), float(np.std(v))
    md = [f"# {EXP}", "", f"Seeds {seeds}. Classifier = exp08 method 3 recipe (standardised input, multinomial logistic, C = 1) over the 64 (family, topic) cells. Mean ± sd over splits; CI = bootstrap on split 0 (query-level / family-stratified).", "",
          "## Cell classifier", "", f"input | cell accuracy | implied family accuracy | cell acc. in {TOPIC_FAMILIES[0]} | cell acc. in {TOPIC_FAMILIES[1]}", "---|---|---|---|---"]
    for inp, lab in (("gte", "GTE"), ("bow", "hashed bag-of-words (control)"), ("majority_cell_in_true_family", "majority cell given the true family (reference)")):
        a, b, m_, h_ = cm([inp, "cell_accuracy"]), cm([inp, "implied_family_accuracy"]), cm([inp, "cell_accuracy_per_family", TOPIC_FAMILIES[0]]), cm([inp, "cell_accuracy_per_family", TOPIC_FAMILIES[1]])
        md.append(f"{lab} | {a[0]:.4f} ± {a[1]:.4f} | {b[0]:.4f} ± {b[1]:.4f} | {m_[0]:.4f} ± {m_[1]:.4f} | {h_[0]:.4f} ± {h_[1]:.4f}")
    fa = cm(["family_classifier_accuracy"]); md += [f"family-only classifier (exp08 method 3) | – | {fa[0]:.4f} ± {fa[1]:.4f} | – | –", ""]
    rec_ = {c_: np.mean([per[s]["classifier"]["gte"]["per_cell_recall"][c_] for s in seeds]) for c_ in per[seeds[0]]["classifier"]["gte"]["per_cell_recall"]}
    mm = sorted([(v, k) for k, v in rec_.items() if k.startswith(TOPIC_FAMILIES[0])])
    md += ["MMLU subjects, GTE recall (mean over splits): lowest 8 — " + ", ".join(f"{k} {v:.2f}" for v, k in mm[:8]) + "; highest 5 — " + ", ".join(f"{k} {v:.2f}" for v, k in mm[-5:]), "",
           "Most frequent MMLU confusions (GTE, split 0, true → predicted, count): " + "; ".join(f"{a} → {b} ({k})" for a, b, k in per[seeds[0]]["classifier"]["gte"]["top_mmlu_confusions"]), ""]
    for mode in AGGS:
        md += [f"## Routing, {mode}", "", "λ | " + " | ".join(METHODS) + " | pred−true cell (CI) | pred cell − kNN-100 (CI) | pred cell − pred family (CI)", "---|" + "---|" * (len(METHODS) + 3)]
        for lam in LAMBDAS:
            row = [f"{ms(f'{lam:g}|{m}|{mode}')[0]:.4f}" for m in METHODS]
            for a, b in (("lookup:predicted_cell", "lookup:true_cell"), ("lookup:predicted_cell", "knn100"), ("lookup:predicted_cell", "lookup:predicted_family")):
                d = ms(f"{lam:g}|{a}|{mode}")[0] - ms(f"{lam:g}|{b}|{mode}")[0]; lo, hi = ci[f"{lam:g}|{a}-{b}|{mode}"]; row.append(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            md.append(f"{lam:g} | " + " | ".join(row))
        md.append("")
    md += ["## Per family (mean over splits): predicted-cell lookup − true-cell lookup / predicted-cell lookup − kNN-100", "", "λ | " + " | ".join(names), "---|" + "---|" * F]
    for lam in (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0):
        md.append(f"{lam:g} | " + " | ".join(f"{ms(f'{lam:g}|lookup:predicted_cell|family:{g}')[0] - ms(f'{lam:g}|lookup:true_cell|family:{g}')[0]:+.4f} / {ms(f'{lam:g}|lookup:predicted_cell|family:{g}')[0] - ms(f'{lam:g}|knn100|family:{g}')[0]:+.4f}" for g in names))
    (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": EXP, "seeds": seeds, "per_seed": per, "ci95_split0": ci}, indent=1))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--aggregate", action="store_true"); a = ap.parse_args()
    aggregate() if a.aggregate else run_seed(a.seed)
