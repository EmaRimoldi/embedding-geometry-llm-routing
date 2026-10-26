"""Print every number the paper's routing tables and the LLMRouterBench appendix need at the rescaled lambda grids.
RB grid {0, 1, 10, 50, 100, 500}; LRB grid {0, .5, 1, 2, 5, 10}. All values are the saved five-split means of the
existing experiments (exp08 basic tests, exp64 baselines, exp15/exp20/exp28/exp32/exp33 chapters); the LLM-cell
lookup rows at LRB lambdas not stored by exp33 come from exp14 (same protocol; identical at the shared lambdas)
and the predicted-LLM-cell row from exp16. Read-only. Writes results/paper_numbers_lambda_rescale/paper_numbers.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

RB = C.E4 / "results"; LRB = C.E4 / "llmrouterbench" / "results" / "performance-cost"
GRID = {"RB": [0.0, 1.0, 10.0, 50.0, 100.0, 500.0], "LRB": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]}
AGG = {"qw": "per_query", "eq": "per_family_equal_weight"}


def J(p): return json.load(open(p))


def bl(j, lam, agg):
    d = j["by_lambda"]; k = next((k for k in (str(float(lam)), f"{lam:g}", str(lam)) if k in d), None)
    if k is None: return None
    v = d[k]["all"][AGG[agg]]["utility"]; return float(v[0] if isinstance(v, list) else v)


def rows(panel, root, base64_dir):
    g = GRID[panel]; out = {}
    ref = J(root / "basic_model_tests" / "references.json")["references"]
    for lab, key in (("oracle", "oracle"), ("best single model", "best_single")):
        out[lab] = {a: [bl({"by_lambda": ref[key]}, l, a) for l in g] for a in AGG}
    for lab, p in (("family lookup", root / "basic_model_tests" / "best_model_per_family" / "best_model_per_family.json"), ("predicted-family lookup", root / "basic_model_tests" / "classify_per_family_best_model" / "classify_per_family_best_model.json"),
                   ("kNN-100", root / "basic_model_tests" / "knn" / "knn100.json"), ("kNN-40", root / "basic_model_tests" / "knn" / "knn40.json"), ("kNN-10", root / "basic_model_tests" / "knn" / "knn10.json"), ("kNN-5", root / "basic_model_tests" / "knn" / "knn5.json"),
                   ("MLP (100)", root / "basic_model_tests" / "mlp" / "mlp100.json"), ("MLP (100,100)", root / "basic_model_tests" / "mlp" / "mlp100x100.json"), ("regularized MLP (100)", root / "basic_model_tests" / "mlp_regularised" / "mlp100.json"), ("regularized MLP (100,100)", root / "basic_model_tests" / "mlp_regularised" / "mlp100x100.json"),
                   ("K-means utility, K=64, top 1", base64_dir / "kmeans64_utility_top1" / "result.json"), ("K-means utility, K=64, top 3", base64_dir / "kmeans64_utility_top3" / "result.json"), ("RouteLLM-MF, two models", base64_dir / "routellm_mf_train_selected_threshold" / "result.json"),
                   ("GraphRouter, performance first", base64_dir / "graphrouter_performance_first" / "result.json"), ("GraphRouter, balance", base64_dir / "graphrouter_balance" / "result.json"), ("GraphRouter, cost first", base64_dir / "graphrouter_cost_first" / "result.json")):
        if not p.exists(): out[lab] = None; continue
        j = J(p)
        if "by_lambda" in j: out[lab] = {a: [bl(j, l, a) for l in g] for a in AGG}
        else:  # mlp files may store methods per file
            out[lab] = None
    for lab, p in (("Avengers-Pro, K=64, top 1, upper bound", base64_dir / "avengerspro_k64_top1" / "result.json"), ("Avengers-Pro, K=64, top 3, upper bound", base64_dir / "avengerspro_k64_top3" / "result.json")):
        if not p.exists(): continue
        a_ = J(p)["by_alpha"]; out[lab] = {"qw": [max(float(rec["utility_by_lambda_per_query"][str(float(l))]) for rec in a_) for l in g], "eq": [max(float(rec.get("utility_by_lambda_equal_weight", rec.get("utility_by_lambda_per_family_equal_weight", {})).get(str(float(l)), float("nan"))) for rec in a_) for l in g]}
    # cell lookup (released cells on both panels) from exp15 locality at k=100; predicted cell from exp20
    e15 = J(root / "outcome_locality" / "exp15_locality_lambda_topic" / "result.json")["results"]; ks = list(e15[list(e15)[0]].keys())[0]
    out["family/topic lookup (released cells)"] = {"qw": [float(e15[str(float(l))][ks]["all"]["cell_lookup"]["utility_per_query"][0]) for l in g], "eq": [float(e15[str(float(l))][ks]["all"]["cell_lookup"]["utility_per_family"][0]) for l in g]}
    e20 = J(root / "outcome_locality" / "exp20_topic_recovery" / "result.json")["per_seed"]
    def e20v(l, m, a): return float(np.mean([e20[s]["routing"][f"{l:g}|lookup:{m}|{AGG[a]}"] for s in e20]))
    out["predicted-cell lookup (released cells)"] = {a: [e20v(l, "predicted_cell", a) for l in g] for a in AGG}
    return out


def main():
    d = C.RESULTS / "paper_numbers_lambda_rescale"; d.mkdir(parents=True, exist_ok=True); md = ["# Paper numbers at the rescaled lambda grids (saved five-split means; read-only)", ""]
    rb = rows("RB", RB, RB / "routerbench_baselines_exp64" / "baselines"); lrb = rows("LRB", LRB, LRB / "baselines")
    # LRB all-topics rows: exp33 at stored lambdas + exp14/exp16 at the new ones
    e33 = J(LRB / "outcome_locality" / "exp33_llm_topics" / "result.json")["A_utility"]; e14 = J(C.RESULTS / "exp14_lrb_topic_labels" / "result.json")["results"]
    p16 = C.RESULTS / "exp16_lrb_predicted_cell" / "result.json"; e16 = J(p16)["results"] if p16.exists() else None
    def e33v(l, m, a): k = f"{l:g}"; return float(e33[k][f"{m}|{AGG[a]}"]) if k in e33 else None
    lrb["family/topic lookup, all topics (exp33 where stored)"] = {a: [e33v(l, "llm", a) for l in GRID["LRB"]] for a in AGG}
    lrb["family/topic lookup, all topics (exp14, all lambdas)"] = {"qw": [e14["lookup:llm"][f"{l:g}"]["qw"][0] for l in GRID["LRB"]], "eq": [e14["lookup:llm"][f"{l:g}"]["macro"][0] for l in GRID["LRB"]]}
    lrb["family/topic lookup, released labels (exp14)"] = {"qw": [e14["lookup:released"][f"{l:g}"]["qw"][0] for l in GRID["LRB"]], "eq": [e14["lookup:released"][f"{l:g}"]["macro"][0] for l in GRID["LRB"]]}
    lrb["predicted-cell lookup, all topics (exp33 where stored)"] = {a: [e33v(l, "llm_predicted", a) for l in GRID["LRB"]] for a in AGG}
    if e16: lrb["predicted-cell lookup, all topics (exp16, all lambdas)"] = {"qw": [e16["predicted_cell"][f"{l:g}"]["qw"] for l in GRID["LRB"]], "eq": [e16["predicted_cell"][f"{l:g}"]["macro"] for l in GRID["LRB"]]}
    for name, tab, g in (("RouterBench", rb, GRID["RB"]), ("LLMRouterBench", lrb, GRID["LRB"])):
        for a, alab in (("qw", "per query"), ("eq", "equal weight over families")):
            md += [f"## {name} — {alab}", "", "router | " + " | ".join(f"λ={l:g}" for l in g), "---|" + "---|" * len(g)]
            for lab, v in tab.items():
                if v is None: md.append(f"{lab} | (not found)"); continue
                md.append(f"{lab} | " + " | ".join("n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}" for x in v[a]))
            md.append("")
    # appendix numbers (LRB)
    e28 = J(LRB / "outcome_locality" / "exp28_choice_by_lambda" / "result.json"); md += ["## LRB exp28 agreement (kNN-100 vs cell / family; predicted vs cell; cell vs family)", ""]
    for l in GRID["LRB"]: k = f"{l:g}"; md.append(f"- λ={k}: " + ", ".join(f"{m} {v:.3f}" for m, v in e28["agreement"][k].items()))
    models = e28["models"]; md += ["", "## LRB exp28 family-lookup model shares (top 3) and number of models used", ""]
    for l in GRID["LRB"]:
        k = f"{l:g}"; s = np.array(e28["share"]["family_lookup"][k]); o = np.argsort(-s)[:3]; md.append(f"- λ={k}: " + ", ".join(f"{str(models[i]).split('/')[-1]} {s[i]:.2f}" for i in o) + f"; models with share>0: {int((s > 0).sum())}")
    e19 = J(LRB / "outcome_locality" / "exp19_decision_agreement" / "result.json")["stats"]; md += ["", "## LRB exp19 (kNN-100 vs cell agreement; utility kNN minus cell, all rows)", ""]
    for l in GRID["LRB"]:
        k = f"{float(l)}"; keys = [x for x in e19 if x.startswith(k + "|100|")]
        md.append(f"- λ={l:g}: " + ", ".join(f"{x.split('|',2)[2]}={e19[x]['mean']:.3f}" for x in keys if x.split("|", 2)[2] in ("knn_vs_cell_agree|per_query", "rand_vs_cell_agree|per_query", "utility_knn_minus_cell_all_rows|per_query", "utility_knn_minus_cell_all_rows|per_family_equal_weight")))
    e32 = J(LRB / "generalization" / "exp32_lofo" / "result.json")["summary"]["prompt"]; md += ["", "## LRB exp32 held-out family (pooled per query [equal weight])", ""]
    for l in GRID["LRB"]:
        k = f"{l:g}"; md.append(f"- λ={k}: " + ", ".join(f"{m} {e32[k][m]['pooled_per_query']:.3f} [{e32[k][m]['equal_weight']:.3f}]" for m in ("knn100", "kmeans64", "classify_cell_lookup", "classify_family_lookup", "best_single_train", "oracle")))
    e15 = J(LRB / "outcome_locality" / "exp15_locality_lambda_topic" / "result.json")["results"]; md += ["", "## LRB exp15 locality at k=100 (released cells): kNN / random same-cell / cell lookup, per query [equal weight]", ""]
    for l in GRID["LRB"]:
        v = e15[str(float(l))]["100"]["all"]; md.append(f"- λ={l:g}: " + ", ".join(f"{m} {v[m]['utility_per_query'][0]:.3f} [{v[m]['utility_per_family'][0]:.3f}]" for m in ("knn", "random_same_cell", "cell_lookup", "family_lookup")))
    (d / "paper_numbers.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
