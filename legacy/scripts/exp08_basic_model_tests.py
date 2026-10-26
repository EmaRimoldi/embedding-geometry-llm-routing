"""exp08: kNN vs best-model-per-family (true label) vs classify-then-best-model, utility fit per λ.

See experiments/exp08_basic_model_tests.yaml for every setting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp08_basic_model_tests"
OUT = RESULTS_ROOT / "basic_model_tests"
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1e3, 1e4]
KS = (5, 10, 40, 100)
SEEDS = (0, 1, 2, 3, 4)
TEST_FRAC = 0.3
METHODS = ["best_model_per_family", "classify_per_family_best_model"] + [f"knn{k}" for k in KS]
REFS = ["best_single", "oracle"]


def aiq(points: np.ndarray, max_cost: float) -> float:
    """RouterBench AIQ: area under the non-decreasing convex hull of (cost, reward) points, / max_cost."""
    pts = points[np.argsort(points[:, 0])]
    hull = []
    for c, r in pts:  # upper envelope, non-decreasing in reward
        if hull and r < hull[-1][1]:
            continue
        hull.append([c, r])
    hull = np.array(hull)
    if hull[-1, 0] < max_cost:
        hull = np.vstack([hull, [max_cost, hull[-1, 1]]])
    return float(np.trapezoid(hull[:, 1], hull[:, 0]) / max_cost)


def summarise(choice: np.ndarray, r: np.ndarray, c: np.ndarray, lam: float, fam: np.ndarray, dis: np.ndarray, n_fam: int) -> dict:
    rr = r[np.arange(len(choice)), choice]; cc = c[np.arange(len(choice)), choice]; uu = rr - lam * cc
    def agg(mask):
        per_q = {"utility": float(uu[mask].mean()), "reward": float(rr[mask].mean()), "cost": float(cc[mask].mean())}
        per_f = {k: float(np.mean([v[mask & (fam == g)].mean() for g in range(n_fam) if (mask & (fam == g)).sum() > 0]))
                 for k, v in (("utility", uu), ("reward", rr), ("cost", cc))}
        return {"per_query": per_q, "per_family_equal_weight": per_f, "n": int(mask.sum())}
    return {"all": agg(np.ones_like(dis)), "disagreement": agg(dis),
            "per_family": {int(g): {"utility": float(uu[fam == g].mean()), "reward": float(rr[fam == g].mean()), "cost": float(cc[fam == g].mean())} for g in range(n_fam)},
            "_per_query_utility": uu}


def main() -> None:
    panel = load_panel(PANEL)
    spec = find_rep(PANEL, REP)
    x = load_rep(spec, panel.n, panel.keep)
    u_unit = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); M = panel.m
    dis_all = (r.max(1) - r.min(1)) > 1e-9

    per_seed = {}
    boot = {}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(panel.n), test_size=TEST_FRAC, random_state=seed, stratify=fam)
        fam_tr, fam_te = fam[tr], fam[te]; dis = dis_all[te]
        # family classifier (λ-independent)
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x[tr]), fam_tr)
        fam_pred = clf.predict(sc.transform(x[te]))
        cls = {"accuracy": float((fam_pred == fam_te).mean()),
               "per_family_recall": {names[g]: float((fam_pred[fam_te == g] == g).mean()) for g in range(F)},
               "confusion_rows_true_cols_pred": confusion_matrix(fam_te, fam_pred, labels=list(range(F))).tolist()}
        # kNN neighbours (λ-independent): top-100 training rows by cosine for each test row
        kmax = max(KS); nb = np.zeros((len(te), kmax), dtype=np.int64)
        for s in range(0, len(te), 2048):
            sims = u_unit[te[s:s + 2048]] @ u_unit[tr].T
            top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)
            nb[s:s + 2048] = top
        # sweep λ
        res_seed = {m: {} for m in METHODS + REFS}; curves = {m: [] for m in METHODS + REFS}; zero = []
        pq = {}
        for lam in LAMBDAS:
            U = r - lam * c
            Utr = U[tr]
            table = np.array([Utr[fam_tr == g].mean(0).argmax() for g in range(F)])      # best model per family (train)
            best_single = int(Utr.mean(0).argmax())
            choices = {
                "best_model_per_family": table[fam_te],
                "classify_per_family_best_model": table[fam_pred],
                "best_single": np.full(len(te), best_single),
                "oracle": U[te].argmax(1),
            }
            for k in KS:
                choices[f"knn{k}"] = Utr[nb[:, :k]].mean(1).argmax(1)
            for m, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fam_te, dis, F)
                pq[(m, lam)] = s.pop("_per_query_utility")
                res_seed[m][str(lam)] = s
                curves[m].append([s["all"]["per_query"]["cost"], s["all"]["per_query"]["reward"]])
            zero.append([[c[te][:, j].mean(), r[te][:, j].mean()] for j in range(M)])
        zero_pts = np.array(zero[0])                                                   # single-model points do not depend on λ
        max_cost = max([np.array(v)[:, 0].max() for v in curves.values()] + [zero_pts[:, 0].max()])
        aiqs = {m: aiq(np.array(v), max_cost) for m, v in curves.items()}; aiqs["zero_router"] = aiq(zero_pts, max_cost)
        per_seed[seed] = {"classifier": cls, "results": res_seed, "curves": curves, "zero_router_points": zero_pts.tolist(), "aiq": aiqs,
                          "n_train": int(len(tr)), "n_test": int(len(te))}
        if seed == SEEDS[0]:
            rng = np.random.default_rng(0)
            for lam in (0.0, 10.0, 100.0):
                for m in ("knn10", "classify_per_family_best_model"):
                    d = pq[(m, lam)] - pq[("best_model_per_family", lam)]
                    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                    boot[f"{m} minus best_model_per_family @ lambda={lam:g}"] = {"point": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
                    dd = d[dis]; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(1000)]
                    boot[f"{m} minus best_model_per_family @ lambda={lam:g} [disagreement]"] = {"point": float(dd.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print(f"seed {seed}: classifier acc={cls['accuracy']:.4f}  AIQ " + " ".join(f"{m}={v:.4f}" for m, v in aiqs.items()), flush=True)

    # ---- write per-method folders and a summary
    prov = panel.provenance()
    def agg_over_seeds(m, lam, subset, aggr, key):
        vals = [per_seed[s]["results"][m][str(lam)][subset][aggr][key] for s in SEEDS]
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))
    for m in METHODS:
        folder = "knn" if m.startswith("knn") else m
        d = OUT / folder; d.mkdir(parents=True, exist_ok=True)
        out = {**prov, "experiment": EXP, "method": m, "lambdas": LAMBDAS, "seeds": list(SEEDS), "family_order": names,
               "by_lambda": {str(lam): {subset: {aggr: {key: agg_over_seeds(m, lam, subset, aggr, key) for key in ("utility", "reward", "cost")}
                                                for aggr in ("per_query", "per_family_equal_weight")} for subset in ("all", "disagreement")} for lam in LAMBDAS},
               "aiq_per_seed": [per_seed[s]["aiq"][m] for s in SEEDS], "curves_per_seed": {str(s): per_seed[s]["curves"][m] for s in SEEDS}}
        if m == "classify_per_family_best_model":
            out["classifier_per_seed"] = {str(s): per_seed[s]["classifier"] for s in SEEDS}
        (d / f"{m}.json").write_text(json.dumps(out, indent=1))
    refs = {**prov, "experiment": EXP, "references": {m: {str(lam): {subset: {aggr: {key: agg_over_seeds(m, lam, subset, aggr, key) for key in ("utility", "reward", "cost")}
                                                                                 for aggr in ("per_query", "per_family_equal_weight")} for subset in ("all", "disagreement")} for lam in LAMBDAS} for m in REFS},
            "aiq_per_seed": {m: [per_seed[s]["aiq"][m] for s in SEEDS] for m in REFS + ["zero_router"]},
            "zero_router_points_seed0": per_seed[SEEDS[0]]["zero_router_points"], "bootstrap_seed0": boot}
    (OUT / "references.json").write_text(json.dumps(refs, indent=1))

    md = [f"# {EXP} — summary", "", f"Panel: {prov['panel']} filtered, {prov['rows']} rows; 5 random 70/30 splits stratified by family (seeds 0–4); values = mean [min, max] over splits.",
          f"Representation: {spec.name}. Utility u_λ = r − λ·c (cost in dollars), fitted per λ. Methods: pooled kNN (k = {list(KS)}), best model per family with the TRUE label, best model per family with the PREDICTED label (multinomial logistic regression on GTE).", ""]
    md += ["## Family classifier (method 3)", "", "seed | accuracy | " + " | ".join(names), "---|---|" + "---|" * F]
    md += [f"{s} | {per_seed[s]['classifier']['accuracy']:.4f} | " + " | ".join(f"{per_seed[s]['classifier']['per_family_recall'][n]:.3f}" for n in names) for s in SEEDS]
    md += ["", "## AIQ over the 15-λ sweep (RouterBench score; per-query cost/reward points; mean [min, max] over splits)", "", "method | AIQ", "---|---"]
    for m in METHODS + REFS + ["zero_router"]:
        v = [per_seed[s]["aiq"][m] for s in SEEDS]; md.append(f"{m} | {np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]")
    for subset in ("all", "disagreement"):
        for aggr in ("per_query", "per_family_equal_weight"):
            md += ["", f"## Mean utility on test — subset: {subset}; aggregation: **{aggr}**", "", "λ | " + " | ".join(METHODS + REFS), "---|" + "---|" * (len(METHODS) + len(REFS))]
            for lam in LAMBDAS:
                cells = []
                for m in METHODS + REFS:
                    mu, lo, hi = agg_over_seeds(m, lam, subset, aggr, "utility"); cells.append(f"{mu:.4f} [{lo:.4f},{hi:.4f}]")
                md.append(f"{lam:g} | " + " | ".join(cells))
    md += ["", "## Reward and cost at λ = 0 and λ = 100 (all test rows, per-query)", "", "method | reward@0 | cost@0 | reward@100 | cost@100", "---|---|---|---|---"]
    for m in METHODS + REFS:
        r0 = agg_over_seeds(m, 0.0, "all", "per_query", "reward")[0]; c0 = agg_over_seeds(m, 0.0, "all", "per_query", "cost")[0]
        r1 = agg_over_seeds(m, 100.0, "all", "per_query", "reward")[0]; c1 = agg_over_seeds(m, 100.0, "all", "per_query", "cost")[0]
        md.append(f"{m} | {r0:.4f} | {c0:.5f} | {r1:.4f} | {c1:.5f}")
    md += ["", "## Paired bootstrap, split 0 (mean per-query utility difference, 95% CI)", "", "comparison | point | CI", "---|---|---"]
    md += [f"{k} | {v['point']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]" for k, v in boot.items()]
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
