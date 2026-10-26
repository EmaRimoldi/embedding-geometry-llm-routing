"""exp09 aggregation: turn the stored MLP predictions into the same tables as exp08 and append rows to SUMMARY.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, aiq, split, summarise  # noqa: E402
from rb.reps import find_rep  # noqa: E402

BASE = RESULTS_ROOT / "basic_model_tests"
PRED = BASE / "mlp" / "predictions"
ARCHS = ("mlp100", "mlp100x100")


def main() -> None:
    panel = load_panel(PANEL); spec = find_rep(PANEL, REP)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); M = panel.m
    dis_all = (r.max(1) - r.min(1)) > 1e-9
    refs = json.loads((BASE / "references.json").read_text())
    per_seed = {}; boot = {}; missing = []
    for seed in SEEDS:
        tr, te = split(panel.n, fam, seed); fam_te = fam[te]; dis = dis_all[te]
        res = {a: {} for a in ARCHS}; curves = {a: [] for a in ARCHS}; pq = {}
        for li, lam in enumerate(LAMBDAS):
            f = PRED / f"seed{seed}_lam{li}.npz"
            if not f.exists():
                missing.append(f.name); continue
            z = np.load(f); assert np.array_equal(z["test_idx"], te)
            for a in ARCHS:
                ch = z[a].argmax(1)
                s = summarise(ch, r[te], c[te], lam, fam_te, dis, F); pq[(a, lam)] = s.pop("_per_query_utility")
                res[a][str(lam)] = s; curves[a].append([s["all"]["per_query"]["cost"], s["all"]["per_query"]["reward"]])
        zero_pts = np.array([[c[te][:, j].mean(), r[te][:, j].mean()] for j in range(M)])
        max_cost = max([np.array(v)[:, 0].max() for v in curves.values() if v] + [zero_pts[:, 0].max()])
        per_seed[seed] = {"results": res, "curves": curves, "aiq": {a: aiq(np.array(v), max_cost) for a, v in curves.items() if len(v) == len(LAMBDAS)}}
        if seed == SEEDS[0] and not missing:
            # paired bootstrap vs best_model_per_family: recompute the lookup's per-query utility on this split
            rng = np.random.default_rng(0)
            for lam in (0.0, 10.0, 100.0):
                U = r - lam * c; table = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)])
                ref = summarise(table[fam_te], r[te], c[te], lam, fam_te, dis, F)["_per_query_utility"]
                for a in ARCHS:
                    d = pq[(a, lam)] - ref; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                    boot[f"{a} minus best_model_per_family @ lambda={lam:g}"] = {"point": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    if missing:
        print("missing predictions:", missing); return
    prov = panel.provenance()
    def agg(a, lam, subset, aggr, key):
        v = [per_seed[s]["results"][a][str(lam)][subset][aggr][key] for s in SEEDS]; return float(np.mean(v)), float(min(v)), float(max(v))
    for a in ARCHS:
        out = {**prov, "experiment": "exp09_mlp", "method": a, "lambdas": LAMBDAS, "seeds": list(SEEDS), "family_order": names,
               "by_lambda": {str(lam): {subset: {aggr: {key: agg(a, lam, subset, aggr, key) for key in ("utility", "reward", "cost")}
                                               for aggr in ("per_query", "per_family_equal_weight")} for subset in ("all", "disagreement")} for lam in LAMBDAS},
               "aiq_per_seed": [per_seed[s]["aiq"][a] for s in SEEDS], "curves_per_seed": {str(s): per_seed[s]["curves"][a] for s in SEEDS},
               "bootstrap_seed0": {k: v for k, v in boot.items() if k.startswith(a)}}
        (BASE / "mlp" / f"{a}.json").write_text(json.dumps(out, indent=1))
    md = ["", "# exp09 — MLP (RouterBench architecture, fitted on u_λ per λ, raw GTE input)", "",
          "## AIQ over the 15-λ sweep (mean [min, max] over splits)", "", "method | AIQ", "---|---"]
    for a in ARCHS:
        v = [per_seed[s]["aiq"][a] for s in SEEDS]; md.append(f"{a} | {np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]")
    for subset in ("all", "disagreement"):
        for aggr in ("per_query", "per_family_equal_weight"):
            md += ["", f"## Mean utility on test — subset: {subset}; aggregation: **{aggr}**", "", "λ | " + " | ".join(ARCHS), "---|" + "---|" * len(ARCHS)]
            for lam in LAMBDAS:
                md.append(f"{lam:g} | " + " | ".join("{:.4f} [{:.4f},{:.4f}]".format(*agg(a, lam, subset, aggr, "utility")) for a in ARCHS))
    md += ["", "## Reward and cost (all test rows, per-query)", "", "method | reward@0 | cost@0 | reward@100 | cost@100", "---|---|---|---|---"]
    for a in ARCHS:
        md.append(f"{a} | {agg(a, 0.0, 'all', 'per_query', 'reward')[0]:.4f} | {agg(a, 0.0, 'all', 'per_query', 'cost')[0]:.5f} | {agg(a, 100.0, 'all', 'per_query', 'reward')[0]:.4f} | {agg(a, 100.0, 'all', 'per_query', 'cost')[0]:.5f}")
    md += ["", "## Paired bootstrap, split 0 (per-query utility difference vs best_model_per_family, 95% CI)", "", "comparison | point | CI", "---|---|---"]
    md += [f"{k} | {v['point']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]" for k, v in boot.items()]
    with (BASE / "SUMMARY.md").open("a") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
